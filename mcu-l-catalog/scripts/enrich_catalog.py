#!/usr/bin/env python3
"""Flatten MCU capabilities and calculate transparent MCU-L selection scores.

Raw source values stay in device-variants.csv. This script creates derived,
query-friendly records without claiming that a calculated score is a benchmark.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable


MODEL_VERSION = "mcu-l-selection-index-v2"

# These Arm architecture profiles have no optional hardware FPU. An official
# record that identifies one of these cores therefore establishes an explicit
# negative without inferring from a sibling MCU.
ARCHITECTURE_NO_FPU = {
    "cortex-m0", "cortex-m0+", "cortex-m1", "cortex-m3", "cortex-m23",
}

EXACT_DEVICE_SOURCE_KINDS = {
    "cubemx_device_db", "microchip_atdf", "puya_device_header",
    "infineon_device_db", "espressif_idf_soc_caps",
    "hpmicro_product_selector", "renesas_product_selector", "ti_datasheet",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def quantity(feature: dict[str, str]) -> float | None:
    return number(feature.get("count")) if number(feature.get("count")) is not None else number(feature.get("n"))


def compact_number(value: float | None) -> int | float | str:
    if value is None:
        return ""
    return int(value) if value.is_integer() else value


def sum_features(
    features: list[dict[str, str]],
    *,
    types: set[str] = frozenset(),
    name_tokens: tuple[str, ...] = (),
    exclude_name_tokens: tuple[str, ...] = (),
) -> float | None:
    matched_features: list[dict[str, str]] = []
    lowered_types = {value.lower() for value in types}
    for feature in features:
        feature_type = feature.get("type", "").lower()
        name = feature.get("name", "").lower()
        type_match = feature_type in lowered_types
        name_match = bool(name_tokens) and any(token in name for token in name_tokens)
        if (type_match or name_match) and not any(token in name for token in exclude_name_tokens):
            matched_features.append(feature)
    # A CubeMX MCU XML describes the matched package/device.  Prefer its
    # explicit instance count (including an authoritative zero) over summary
    # PDSC features.  A family SVD is intentionally not preferred: it may list
    # the register superset for several package/memory variants.
    exact_device_features = [
        item for item in matched_features
        if item.get("source_kind") in EXACT_DEVICE_SOURCE_KINDS
    ]
    if exact_device_features:
        matched_features = exact_device_features
    total = 0.0
    found = False
    for feature in matched_features:
        value = quantity(feature)
        if value is not None:
            total += value
            found = True
    return total if found else None


def bit_widths(features: list[dict[str, str]], feature_types: set[str]) -> list[int]:
    result: set[int] = set()
    for feature in features:
        if feature.get("type", "").lower() not in {value.lower() for value in feature_types}:
            continue
        name = feature.get("name", "")
        # A bare number in a peripheral description is usually a channel
        # count or sample rate, not a resolution.  Require an explicit bit
        # suffix so e.g. "16 external channels" cannot become "16-bit ADC".
        for value in re.findall(r"(?<!\d)(8|10|12|14|16|24|32)\s*-?[Bb]it\b", name):
            result.add(int(value))
        raw = feature.get("m", "")
        for value in re.findall(r"\d+(?:\.\d+)?", raw):
            numeric = float(value)
            if numeric <= 32:
                result.add(int(numeric))
    return sorted(result)


def engineering_feature_text(features: list[dict[str, str]], types: set[str]) -> str:
    """Join only explicitly tagged engineering evidence records.

    Keeping this separate from the generic inventory prevents an unrelated
    clock, register, or package number from becoming an ADC/timer metric.
    """
    wanted = {value.lower() for value in types}
    return " | ".join(
        str(item.get("name") or "")
        for item in features
        if str(item.get("type") or "").lower() in wanted and item.get("name")
    )


def engineering_metrics(features: list[dict[str, str]]) -> dict[str, Any]:
    """Extract conservative, source-backed engineering metrics.

    Empty values mean that the official source did not publish a comparable
    value.  We intentionally do not convert a family-level clock or a GPIO
    electrical limit into a peripheral speed unless the evidence names that
    block and its unit.
    """
    adc_text = engineering_feature_text(features, {"ADCPerformance"})
    dac_text = engineering_feature_text(features, {"DACPerformance"})
    io_text = engineering_feature_text(features, {"IOSpeed"})
    timer_text = engineering_feature_text(features, {"TimerArchitecture"})
    memory_text = engineering_feature_text(features, {"MemoryArchitecture"})

    def max_rate(text: str, pattern: str) -> int | None:
        values: list[int] = []
        for match in re.finditer(pattern, text, re.IGNORECASE):
            unit = match.group("unit").lower().replace(" ", "")
            factors = {"gsps": 1_000_000_000, "msps": 1_000_000, "ksps": 1_000, "sps": 1}
            if unit in factors:
                values.append(round(float(match.group("value")) * factors[unit]))
        return max(values) if values else None

    adc_rate = max_rate(adc_text, r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>GSPS|MSPS|KSPS|SPS)")
    dac_rate = max_rate(dac_text, r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>GSPS|MSPS|KSPS|SPS)")
    io_rate = max_rate(io_text, r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>MSPS|KSPS|SPS)")
    if io_rate is None:
        io_values = [round(float(item.group("value")) * 1_000_000) for item in re.finditer(r"(?P<value>\d+(?:\.\d+)?)\s*MHz", io_text, re.IGNORECASE)]
        io_rate = max(io_values) if io_values else None

    timer_widths = sorted({int(value) for value in re.findall(r"(?<!\d)(8|16|24|32)\s*-?\s*bit(?=[^|]{0,80}(?:timer|tmr|counter))", timer_text, re.IGNORECASE)})
    timer_widths.extend(int(value) for value in re.findall(r"(?:timer|tmr|counter)[^|]{0,80}?(?<!\d)(8|16|24|32)\s*-?\s*bit", timer_text, re.IGNORECASE) if int(value) not in timer_widths)
    timer_widths = sorted(set(timer_widths))

    flash_lower = memory_text.lower()
    flash_bank_count = 2 if re.search(r"dual[- ]?bank|two[- ]?bank|双区", flash_lower) else 1 if re.search(r"single[- ]?bank|one[- ]?bank|单区", flash_lower) else None
    flash_wait_states = 0 if re.search(r"zero[- ]?wait|0[- ]?wait|零等待", flash_lower) else None
    wait_match = re.search(r"(?P<value>\d+)\s*[- ]?wait(?:ing)?\s*states?", flash_lower, re.IGNORECASE)
    if wait_match:
        flash_wait_states = int(wait_match.group("value"))
    flash_ecc = "yes" if re.search(r"(?:flash|nvm)[^|]{0,90}\becc\b|\becc\b[^|]{0,90}(?:flash|nvm)", flash_lower) else "unknown"
    cache = "yes" if re.search(r"(?:i-?cache|d-?cache|instruction cache|data cache|cache controller)", flash_lower) else "unknown"
    ram_ecc = "yes" if re.search(r"(?:ram|sram)[^|]{0,90}\becc\b|\becc\b[^|]{0,90}(?:ram|sram)", flash_lower) else "unknown"
    ram_types = sorted({value.upper() for value in re.findall(r"\b(?:ITCM|DTCM|TCM|CCM|AXI\s+SRAM|SRAM[12]|OCRAM|PSRAM)\b", flash_lower, re.IGNORECASE)})
    ram_exclusive = "yes" if re.search(r"(?:dedicated|private|exclusive|per[- ]core)[^|]{0,120}(?:ram|sram|memory)", flash_lower) else "unknown"
    return {
        "adc_sample_rate_hz": adc_rate,
        "dac_sample_rate_hz": dac_rate,
        "io_speed_hz": io_rate,
        "timer_engineering_width_bits": ";".join(str(value) for value in timer_widths),
        "flash_wait_states": flash_wait_states,
        "flash_bank_count": flash_bank_count,
        "flash_ecc_present": flash_ecc,
        "cache_present": cache,
        "ram_ecc_present": ram_ecc,
        "ram_architecture": ";".join(ram_types),
        "ram_exclusive_present": ram_exclusive,
    }


POWER_CURRENT_RE = re.compile(
    r"(?P<value>\d+(?:\.\d+)?)\s*(?P<prefix>[mnuμµ]?)\s*A"
    r"(?:\s*/\s*(?P<per>MHz|kHz))?",
    re.IGNORECASE,
)
POWER_WATT_RE = re.compile(
    r"(?P<value>\d+(?:\.\d+)?)\s*(?P<prefix>[mnuμµ]?)\s*W",
    re.IGNORECASE,
)
POWER_CLOCK_RE = re.compile(
    r"(?:at|@|\(|,|，|在)\s*(?P<value>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>GHz|MHz|kHz|Hz)",
    re.IGNORECASE,
)
POWER_VOLTAGE_RE = re.compile(
    r"(?:at|@|vdd|vcc|供电|电压)\s*[=:]?\s*"
    r"(?P<value>\d+(?:\.\d+)?)\s*V\b",
    re.IGNORECASE,
)
POWER_TEMPERATURE_RE = re.compile(
    r"(?P<value>-?\d+(?:\.\d+)?)\s*(?:°\s*)?(?:C|℃)\b",
    re.IGNORECASE,
)


def _power_mode(name: str) -> str:
    lowered = name.lower()
    if any(token in lowered for token in ("sleep", "standby", "stop", "lpm", "deep power-down", "low-power")):
        return "sleep"
    if any(token in lowered for token in ("active", "run", "operating")):
        return "run"
    return "other"


def _power_quality(name: str) -> str:
    lowered = name.lower()
    if re.search(r"\btyp(?:ical)?\b|典型", lowered, re.IGNORECASE):
        return "typical"
    if re.search(r"\bmax(?:imum)?\b|最大", lowered, re.IGNORECASE):
        return "maximum"
    if re.search(r"\bmin(?:imum)?\b|最小", lowered, re.IGNORECASE):
        return "minimum"
    return "unknown"


def _power_frequency_hz(value: str, unit: str) -> int:
    multipliers = {"ghz": 1_000_000_000, "mhz": 1_000_000, "khz": 1_000, "hz": 1}
    return round(float(value) * multipliers[unit.lower()])


def power_measurements(
    features: list[dict[str, str]],
    source_document_id: str,
    verification_status: str = "cmsis_pack_metadata",
) -> list[dict[str, Any]]:
    """Extract only source text that contains an explicit power unit.

    CMSIS feature records often contain mode counts or unitless power hints.
    Those are intentionally ignored here: a number without an explicit A/W
    unit cannot be compared as a current or power measurement.
    """
    result: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for feature in features:
        feature_type = str(feature.get("type") or "").strip().lower()
        if feature_type not in {"consumption", "powerother", "power"}:
            continue
        name = str(feature.get("name") or "").strip()
        if not name:
            continue
        match = POWER_CURRENT_RE.search(name) or POWER_WATT_RE.search(name)
        if not match:
            continue
        raw_value = float(match.group("value"))
        prefix = (match.groupdict().get("prefix") or "").replace("μ", "u").replace("µ", "u")
        current_match = POWER_CURRENT_RE.search(name)
        if current_match:
            unit = f"{prefix}A" if prefix else "A"
            per = current_match.groupdict().get("per")
            if per:
                unit = f"{unit}_per_{per}"
        else:
            unit = f"{prefix}W" if prefix else "W"
        conditions: dict[str, Any] = {}
        clock = POWER_CLOCK_RE.search(name)
        if clock:
            conditions["clock_hz"] = _power_frequency_hz(clock.group("value"), clock.group("unit"))
        voltage = POWER_VOLTAGE_RE.search(name)
        if voltage:
            conditions["voltage_v"] = float(voltage.group("value"))
        temperature = POWER_TEMPERATURE_RE.search(name)
        if temperature:
            conditions["temperature_c"] = float(temperature.group("value"))
        if not conditions:
            conditions["note"] = "条件未完整披露"
        record: dict[str, Any] = {
            "mode": _power_mode(name),
            "label": name,
            "value": compact_number(raw_value),
            "unit": unit,
            "typical_or_max": _power_quality(name),
            "conditions": conditions,
            "source_document_id": str(feature.get("source_document_id") or source_document_id or "unknown"),
            "verification_status": str(feature.get("verification_status") or verification_status),
        }
        key = (
            record["mode"], record["label"], record["value"], record["unit"],
            json.dumps(conditions, sort_keys=True, ensure_ascii=False),
        )
        if key not in seen:
            seen.add(key)
            result.append(record)
    return result


def bool_from_processors(processors: list[dict[str, str]], key: str) -> str:
    values = [item.get(key) for item in processors if item.get(key) not in (None, "")]
    if not values:
        return "unknown"
    normalized = [str(value).strip().lower().replace("-", "_").replace(" ", "_") for value in values]
    false_values = {"0", "false", "none", "no", "disabled", "not_present"}
    unknown_values = {"unknown", "unspecified", "na", "n/a"}
    known = [value for value in normalized if value not in unknown_values]
    if not known:
        return "unknown"
    return "yes" if any(value not in false_values and not value.startswith("no_") for value in known) else "no"


def fpu_from_processors(processors: list[dict[str, str]]) -> str:
    explicit = bool_from_processors(processors, "Dfpu")
    if explicit != "unknown":
        return explicit
    cores = [item.get("Dcore", "").strip().lower() for item in processors if item.get("Dcore")]
    if cores and all(core in ARCHITECTURE_NO_FPU for core in cores):
        return "no"
    return "unknown"


def processor_core_count(processors: list[dict[str, str]]) -> int | str:
    if not processors:
        return ""
    explicit = [number(item.get("DcoreCount")) for item in processors]
    explicit = [value for value in explicit if value is not None and value > 0]
    if explicit:
        return int(sum(explicit))
    descriptions = " ".join(item.get("DsourceDescription", "").lower() for item in processors)
    if "exact core count not stated" in descriptions:
        return ""
    return len(processors)


def core_rank(core_names: list[str]) -> float | None:
    ranks = {
        "avr8l": 8, "avr8": 12, "avr8x": 16,
        "cortex-m0": 18, "cortex-m0+": 24, "cortex-m1": 20,
        "cortex-m3": 44, "cortex-m4": 60, "cortex-m7": 82,
        "cortex-m23": 34, "cortex-m33": 66, "cortex-m35p": 72,
        "cortex-m52": 82, "cortex-m55": 90, "cortex-m85": 100,
    }
    matched = [ranks[name.lower()] for name in core_names if name.lower() in ranks]
    return max(matched) if matched else None


def log_score(value: float | None, low: float, high: float) -> float | None:
    if value is None or value <= 0:
        return None
    low_log = math.log2(low)
    high_log = math.log2(high)
    return max(0.0, min(100.0, 100.0 * (math.log2(value) - low_log) / (high_log - low_log)))


def weighted_score(parts: list[tuple[float | None, float]]) -> tuple[float | None, int]:
    available = [(value, weight) for value, weight in parts if value is not None]
    if not available:
        return None, 0
    total_weight = sum(weight for _, weight in available)
    full_weight = sum(weight for _, weight in parts)
    result = sum(value * weight for value, weight in available) / total_weight
    return result, round(100 * total_weight / full_weight)


def match_override(row: dict[str, str], override: dict[str, str]) -> bool:
    if override.get("manufacturer") and override["manufacturer"] != row["manufacturer"]:
        return False
    match_type = override.get("match_type", "")
    value = override.get("match_value", "")
    if match_type == "device":
        return fnmatch.fnmatchcase(row["device_name"], value)
    if match_type == "device_prefix":
        return row["device_name"].startswith(value)
    if match_type == "product_line":
        return row["product_line"] == value
    if match_type == "series":
        return row["series"] == value
    return False


def auto_features(
    features: list[dict[str, str]], verification_status: str = "cmsis_pack_metadata"
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accelerators: list[dict[str, Any]] = []
    special: list[dict[str, Any]] = []
    # These blocks are useful selection signals even when a vendor names them
    # only by the IP block acronym. Keep them source-backed; do not infer
    # support from the product family name.
    accelerator_types = {
        "npu", "crypto", "accelerator", "dma2d", "cordic", "fmac", "mdf", "dfsdm",
        "jpeg", "divas", "hsp_engine", "hsp1", "neochrom", "chrom-art", "gfx",
        "gpu", "openvg", "pdmic", "spdifrx", "otfdec", "ramecc", "flexramecc",
        "aes", "aesb", "pka", "crccu", "crcscan", "bsec", "hsm",
        "dmamux", "linkedlist", "lpbam", "lpbamqueue", "lpbamadc",
        "lpbamdac", "lpbami2c", "lpbamlptim", "lpbamlpuart", "lpbamspi",
    }
    accelerator_phrases = (
        "accelerator", "neural", "chrom-art",
        "dma2d", "graphics", "gpu", "openvg", "trigonometric", "math unit",
        "jpeg codec", "fft", "filter accelerator", "vector graphics", "neochrom",
        "pdm microphone", "spdif", "ecc", "on-the-fly decryption", "cache controller",
    )
    accelerator_token_pattern = re.compile(
        r"(?<![a-z0-9])(?:npu|tmu|cordic|fmac|crc|aes|sha|hash|pka)\d*(?![a-z0-9])",
        re.I,
    )
    special_types = {
        "rng", "touch", "camera", "lcd", "glcd", "display", "audio", "sai", "pdmic",
        "spdifrx", "application", "security", "vendorcapability", "mipi", "i3c",
        "canfd", "flexray", "ucpd", "usbpd", "usbss", "usbhs", "usb_otg_hs",
        "ethphy", "sqi", "psram", "rtc_ram", "pdec", "evsys", "ccl", "swpmi",
        "subghz", "ieee802154", "pio", "hsem", "icache", "dcache", "ramecc",
        "otfdec", "vrefbuf", "gtzc", "sau", "idau", "dts",
        "lpbamlpgpio", "lpbamcomp", "lpbamopamp", "lpbamvrefbuf", "lpbamqueue",
    }
    for feature in features:
        if quantity(feature) == 0:
            continue
        feature_type = feature.get("type", "")
        lowered_type = feature_type.lower()
        name = feature.get("name", "")
        descriptive = " ".join(value for value in (name, feature.get("n", ""), feature.get("m", "")) if value)
        record = {
            "name": descriptive or feature_type,
            "category": feature_type or "Other",
            "source_feature": feature,
            "verification_status": verification_status,
        }
        if (
            lowered_type in accelerator_types
            or any(word in descriptive.lower() for word in accelerator_phrases)
            or accelerator_token_pattern.search(descriptive)
        ):
            accelerators.append(record)
        elif lowered_type in special_types:
            special.append(record)
    return accelerators, special


PERIPHERAL_CATEGORIES = {
    "adc": "analog", "adcunits": "analog", "adc12": "analog", "adc16": "analog", "a/d": "analog",
    "dac": "analog", "d/a": "analog", "comp": "analog", "analogother": "analog",
    "tempsens": "analog", "touch": "analog", "ts": "analog",
    "timer": "timing", "timerother": "timing", "pwm": "timing", "wdt": "timing",
    "rtc": "timing",
    "i2c": "connectivity", "spi": "connectivity", "i2s": "connectivity",
    "uart": "connectivity", "usart": "connectivity", "lin": "connectivity",
    "can": "connectivity", "usbd": "connectivity", "usbh": "connectivity",
    "usbotg": "connectivity", "usb": "connectivity", "eth": "connectivity", "sdio": "connectivity",
    "ethphy": "connectivity",
    "sdiohost": "connectivity", "sdioslave": "connectivity", "i3c": "connectivity",
    "wifi": "wireless", "wifi6": "wireless", "bluetooth": "wireless",
    "ieee802154": "wireless",
    "mipi": "connectivity", "mpserial": "connectivity", "com": "connectivity",
    "comother": "connectivity", "audio": "display_multimedia",
    "camera": "display_multimedia", "lcd": "display_multimedia", "glcd": "display_multimedia",
    "extbus": "memory_bus", "dma": "memory_bus",
    "crypto": "security", "security": "security", "rng": "security", "npu": "accelerator",
    "coreother": "accelerator", "accelerator": "accelerator", "vendorcapability": "other",
    "rmt": "timing", "ledpwm": "timing", "mcpwm": "timing",
    "psram": "memory_bus", "rtc_ram": "memory_bus", "hall": "analog", "tof": "connectivity",
    "nvic": "system", "extint": "system", "pll": "clock", "xtal": "clock",
    "intrc": "clock", "clockother": "clock", "powermode": "power",
    "powerother": "power", "consumption": "power", "ios": "gpio", "i/o": "gpio",
    "gpiopadcount": "gpio", "gpioport": "gpio", "gpio": "gpio", "spiperipheraltotal": "connectivity",
    "pwr": "power", "rcc": "clock", "sys": "system", "debug": "system", "jtag": "system",
    "dwt": "system", "fpu": "accelerator", "mpu": "system", "cortex_m33": "system",
    # Advanced peripherals and vendor IP blocks found in CMSIS/device-db
    # records. They remain visible in the inventory even when no flat metric
    # exists for them.
    "dma2d": "accelerator", "cordic": "accelerator", "fmac": "accelerator",
    "mdf": "display_multimedia", "dfsdm": "display_multimedia", "jpeg": "display_multimedia",
    "sai": "display_multimedia", "pdmic": "display_multimedia", "spdifrx": "display_multimedia",
    "display": "display_multimedia", "mipi": "display_multimedia", "hdmi_cec": "display_multimedia",
    "canfd": "connectivity", "flexray": "connectivity", "ucpd": "connectivity",
    "usbpd": "connectivity", "usbss": "connectivity", "usbhs": "connectivity",
    "usb_otg_hs": "connectivity", "lin_uart": "connectivity", "linuart": "connectivity",
    "swpmi": "connectivity", "subghz": "wireless", "zigbee": "wireless",
    "ble": "wireless", "ieee802154": "wireless", "ethphy": "connectivity",
    "dmamux": "memory_bus", "dmachannels": "memory_bus", "linkedlist": "memory_bus",
    "mem2mem": "memory_bus", "psram": "memory_bus", "rtc_ram": "memory_bus",
    "icache": "memory_bus", "dcache": "memory_bus", "ramecc": "memory_bus",
    "flexramecc": "memory_bus", "otfdec": "security", "gtzc": "security", "sau": "security",
    "idau": "security", "aes": "security", "aesb": "security", "pka": "security",
    "crccu": "security", "crcscan": "security", "bsec": "security", "hsm": "security",
    "hsem": "system", "evsys": "system", "ccl": "system",
    "pdec": "timing", "qei": "timing", "pio": "accelerator", "hsp_engine": "accelerator",
    "hsp1": "accelerator", "divas": "accelerator", "gfx": "accelerator", "rcc": "clock",
    "vrefbuf": "analog", "sdadc": "analog", "bandgap": "analog", "zcd": "analog",
    "battery_protection": "power", "charger_detect": "power", "voltage_regulator": "power",
}


EXTERNAL_BUS_TOKEN_RE = re.compile(
    r"(?<![A-Z0-9])(?:OCTOSPI\d*|OCTOSPIM|OSPI\d*|XSPI\d*|XSPIM|"
    r"QUADSPI\d*|QSPI\d*|FLEXSPI\d*|HYPERBUS|EMIF\d*|EBI\d*|FMC\d*|FSMC\d*|SMC\d*|SQI\d*)(?![A-Z0-9])",
    re.IGNORECASE,
)


def external_bus_types(features: list[dict[str, str]]) -> list[str]:
    """Keep explicitly named external-memory interfaces as source evidence."""
    result: set[str] = set()
    for item in features:
        if item.get("type", "").lower() not in {"extbus", "comother", "other", "vendorcapability"}:
            continue
        source = " ".join(str(item.get(key) or "") for key in ("name", "n", "description"))
        result.update(match.group(0).upper() for match in EXTERNAL_BUS_TOKEN_RE.finditer(source))
    return sorted(result)


def peripheral_inventory(
    features: list[dict[str, str]], verification_status: str = "cmsis_pack_metadata"
) -> list[dict[str, Any]]:
    """Preserve every source-backed peripheral feature without inventing semantics."""
    result: list[dict[str, Any]] = []
    excluded = {"vcc", "temp", "memory", "memoryother", "application"}
    package_pattern = re.compile(r"^(?:l?qfp\d*|qfn\d*|bga|csp|sop|dip|lga|packageother)$", re.I)
    for feature in features:
        feature_type = (feature.get("type") or "").strip()
        lowered = feature_type.lower()
        if not feature_type or lowered in excluded or package_pattern.match(feature_type):
            continue
        raw_quantity = feature.get("count") if feature.get("count") not in (None, "") else feature.get("n", "")
        if str(raw_quantity).strip() == "0":
            continue
        record: dict[str, Any] = {
            "category": PERIPHERAL_CATEGORIES.get(lowered, "other"),
            "type": feature_type,
            "name": feature.get("name", ""),
            "verification_status": verification_status,
        }
        for key in ("count", "n", "m"):
            if feature.get(key) not in (None, ""):
                record[key] = feature[key]
        result.append(record)
    return result


def preferred_count(
    features: list[dict[str, str]],
    *,
    types: set[str],
    name_tokens: tuple[str, ...] = (),
) -> float | None:
    direct = sum_features(features, types=types)
    if direct is not None:
        return direct
    return sum_features(features, name_tokens=name_tokens)


def operating_voltage_range(features: list[dict[str, str]]) -> tuple[float | None, float | None]:
    """Extract an explicit MCU supply range from VCC/VDD source records."""
    for feature in features:
        if str(feature.get("type") or "").strip().lower() not in {"vcc", "vdd"}:
            continue
        lower = number(feature.get("n"))
        upper = number(feature.get("m"))
        if lower is None or upper is None:
            lower = number(feature.get("min") or feature.get("minimum") or feature.get("vmin"))
            upper = number(feature.get("max") or feature.get("maximum") or feature.get("vmax"))
        if lower is not None and upper is not None and 0 < lower <= upper <= 20:
            return lower, upper
    return None, None


def feature_present(
    features: list[dict[str, str]],
    *,
    types: set[str],
    name_tokens: tuple[str, ...] = (),
) -> str:
    lowered_types = {value.lower() for value in types}
    matched = [
        feature for feature in features
        if feature.get("type", "").lower() in lowered_types
        or any(token in feature.get("name", "").lower() for token in name_tokens)
    ]
    exact = [
        feature for feature in matched
        if feature.get("source_kind") in EXACT_DEVICE_SOURCE_KINDS
    ]
    if exact:
        matched = exact
    for feature in matched:
        feature_type = feature.get("type", "").lower()
        name = feature.get("name", "").lower()
        raw_quantity = feature.get("count") if feature.get("count") not in (None, "") else feature.get("n", "")
        if (feature_type in lowered_types or any(token in name for token in name_tokens)) and str(raw_quantity) != "0":
            return "yes"
    numeric = [quantity(feature) for feature in matched]
    if matched and any(value is not None for value in numeric) and all(value in (None, 0) for value in numeric):
        return "no"
    return "unknown"


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=root / "data" / "combined")
    parser.add_argument("--overrides", type=Path, default=root / "data" / "vendor-feature-overrides.csv")
    args = parser.parse_args()

    devices = read_csv(args.data_dir / "device-variants.csv")
    overrides = read_csv(args.overrides)
    capabilities: list[dict[str, Any]] = []
    scores: list[dict[str, Any]] = []

    for row in devices:
        processors = json.loads(row.get("processor_cores") or "[]")
        features = json.loads(row.get("features_json") or "[]")
        core_names = sorted({item.get("Dcore", "") for item in processors if item.get("Dcore")})
        fpu_present = fpu_from_processors(processors)
        max_clock_hz = number(row.get("max_clock_hz"))
        operating_voltage_min_v, operating_voltage_max_v = operating_voltage_range(features)

        timer_count = sum_features(
            features,
            types={"Timer"},
            exclude_name_tokens=("watchdog", "wdg", "rtc"),
        )
        pwm_count = sum_features(features, types={"PWM"}, name_tokens=("pwm",))
        gpio_count = sum_features(features, types={"IOs", "I/O"}, name_tokens=("input and output ports", "general purpose i/o"))
        spi_count = sum_features(features, types={"SPI"}, name_tokens=("spi interface",))
        i2c_count = sum_features(features, types={"I2C"}, name_tokens=("i2c interface",))
        usart_count = sum_features(features, types={"USART"}, name_tokens=("usart interface",))
        uart_count = sum_features(features, types={"UART"}, name_tokens=("uart interface",))
        can_count = sum_features(features, types={"CAN"}, name_tokens=("can interface",))
        dma_count = sum_features(features, types={"DMA"}, name_tokens=("direct memory access",))
        usb_count = sum_features(features, types={"USB"})
        usb_device_count = sum_features(features, types={"USBD", "USBOTG"})
        usb_host_count = sum_features(features, types={"USBH", "USBOTG"})
        ethernet_count = sum_features(features, types={"ETH"})
        rng_count = sum_features(features, types={"RNG"})
        i2s_count = sum_features(features, types={"I2S"}, name_tokens=("i2s interface",))
        lin_count = sum_features(features, types={"LIN"}, name_tokens=("lin interface",))
        configurable_serial_count = sum_features(
            features,
            types={"Com"},
            name_tokens=("sercom", "flexcom", "configurable serial"),
        )
        usb_otg_count = sum_features(features, types={"USBOTG"})
        sdio_count = sum_features(features, types={"SDIO"}, name_tokens=("sdio", "sdmmc"))
        watchdog_count = preferred_count(
            features,
            types={"WDT"},
            name_tokens=("watchdog", "iwdg", "wwdg"),
        )
        comparator_count = preferred_count(
            features,
            types={"COMP"},
            name_tokens=("comparator",),
        )
        opamp_count = sum_features(
            features,
            name_tokens=("operational amplifier", "op amp", "opamp"),
        )
        touch_count = preferred_count(features, types={"Touch"}, name_tokens=("touch",))
        camera_count = preferred_count(features, types={"Camera"}, name_tokens=("camera", "dcmi"))
        display_count = sum_features(features, types={"LCD", "GLCD"}, name_tokens=("lcd controller",))
        external_bus_count = preferred_count(
            features,
            types={"ExtBus"},
            name_tokens=("octo spi", "octospi", "quad spi", "qspi", "external bus", "fmc", "fsmc"),
        )
        external_interrupt_count = sum_features(features, types={"ExtInt"}, name_tokens=("external interrupt",))
        temperature_sensor_count = preferred_count(
            features,
            types={"TempSens", "TS"},
            name_tokens=("temperature sensor",),
        )
        rtc_present = feature_present(features, types={"RTC"}, name_tokens=("real time clock", "rtc"))
        crypto_present = feature_present(
            features,
            types={"Crypto"},
            name_tokens=("aes", "hash", "crypto", "pka", "encryption"),
        )
        source_verification = row.get("verification_status") or "cmsis_pack_metadata"
        inventory = peripheral_inventory(features, source_verification)
        named_external_buses = external_bus_types(features)
        power = power_measurements(features, row.get("source_id", ""), source_verification)
        engineering = engineering_metrics(features)

        adc_features = [
            item for item in features
            if item.get("type", "").lower() in {"adc", "a/d"}
            or "analog to digital" in item.get("name", "").lower()
        ]
        adc_unit_features = [
            item for item in features
            if item.get("type", "").lower() == "adcunits"
            or "adc converter unit" in item.get("name", "").lower()
        ]
        dac_features = [
            item for item in features
            if item.get("type", "").lower() in {"dac", "d/a"}
            or "digital to analog" in item.get("name", "").lower()
        ]
        adc_source_values = [
            value for value in (quantity(item) for item in adc_features)
            if value is not None
        ]
        dac_source_values = [
            value for value in (quantity(item) for item in dac_features)
            if value is not None
        ]
        adc_source_quantity = sum(adc_source_values) if adc_source_values else None
        dac_source_quantity = sum(dac_source_values) if dac_source_values else None
        adc_unit_values: list[float] = []
        adc_channel_values: list[float] = []
        exact_adc_units = [
            item for item in adc_unit_features
            if item.get("source_kind") in EXACT_DEVICE_SOURCE_KINDS
        ]
        unit_input = exact_adc_units or adc_unit_features
        explicit_channel_features = [
            item for item in adc_features
            if "channel" in item.get("name", "").lower()
        ]
        adc_semantic_inputs = (
            [*unit_input, *explicit_channel_features]
            if exact_adc_units else [*adc_features, *unit_input]
        )
        for item in adc_semantic_inputs:
            item_quantity = quantity(item)
            if item_quantity is None:
                continue
            item_name = item.get("name", "").lower()
            item_type = item.get("type", "").lower()
            if item_type == "adcunits" or "converter unit" in item_name:
                adc_unit_values.append(item_quantity)
            elif "channel" in item_name:
                adc_channel_values.append(item_quantity)
            elif not unit_input and ("converter" in item_name or item_type == "a/d"):
                adc_unit_values.append(item_quantity)
        adc_unit_count = sum(adc_unit_values) if adc_unit_values else None
        adc_channel_count = sum(adc_channel_values) if adc_channel_values else None
        adc_semantics = "unknown"
        if adc_features:
            names = " ".join(item.get("name", "").lower() for item in adc_features)
            if "channel" in names:
                adc_semantics = "channels"
            elif "converter" in names or any(item.get("type") == "A/D" for item in adc_features):
                adc_semantics = "converter_units"
            else:
                adc_semantics = "source_quantity_unspecified"

        accelerators, special_features = auto_features(features, source_verification)
        candidates: list[dict[str, Any]] = []
        for override in overrides:
            if not match_override(row, override):
                continue
            record = {
                "feature_id": override.get("feature_id", ""),
                "name": override.get("display_name", ""),
                "category": override.get("category", ""),
                "quantity": override.get("quantity", ""),
                "unit": override.get("unit", ""),
                "source_url": override.get("source_url", ""),
                "verification_status": override.get("verification_status", ""),
                "notes": override.get("notes", ""),
            }
            if record["verification_status"] in {"manufacturer_datasheet", "manufacturer_product_page", "cmsis_pack_metadata"}:
                accelerators.append(record)
            else:
                candidates.append(record)

        missing: list[str] = []
        for label, value in (
            ("core", core_names), ("clock", max_clock_hz), ("timer_count", timer_count),
            ("gpio_count", gpio_count), ("adc", adc_source_quantity),
        ):
            if label == "adc":
                value = next(
                    (
                        candidate for candidate in (
                            adc_source_quantity, adc_unit_count,
                            adc_channel_count,
                        )
                        if candidate is not None
                    ),
                    None,
                )
            if value in (None, "", []):
                missing.append(label)

        capability = {
            "device_id": row["device_id"],
            "manufacturer": row["manufacturer"],
            "device_name": row["device_name"],
            "core_names": ";".join(core_names),
            "primary_core": core_names[0] if len(core_names) == 1 else "+".join(core_names),
            "core_count": processor_core_count(processors),
            "max_clock_hz": compact_number(max_clock_hz),
            "operating_voltage_min_v": compact_number(operating_voltage_min_v),
            "operating_voltage_max_v": compact_number(operating_voltage_max_v),
            "fpu_present": fpu_present,
            "mpu_present": bool_from_processors(processors, "Dmpu"),
            "dsp_extension_present": bool_from_processors(processors, "Ddsp"),
            "trustzone_present": bool_from_processors(processors, "Dtz"),
            "timer_count": compact_number(timer_count),
            "timer_width_bits": ";".join(str(value) for value in sorted(set(
                bit_widths(features, {"Timer", "TimerArchitecture"})
                + [int(value) for value in engineering["timer_engineering_width_bits"].split(";") if value]
            ))),
            "pwm_source_quantity": compact_number(pwm_count),
            "adc_source_quantity": compact_number(adc_source_quantity),
            "adc_quantity_semantics": adc_semantics,
            "adc_unit_count": compact_number(adc_unit_count),
            "adc_channel_count": compact_number(adc_channel_count),
            "adc_resolution_bits": ";".join(str(value) for value in bit_widths(features, {"ADC", "ADCUnits", "A/D", "ADCPerformance"})),
            "dac_source_quantity": compact_number(dac_source_quantity),
            "dac_resolution_bits": ";".join(str(value) for value in bit_widths(features, {"DAC", "D/A", "DACPerformance"})),
            "adc_sample_rate_hz": compact_number(engineering["adc_sample_rate_hz"]),
            "dac_sample_rate_hz": compact_number(engineering["dac_sample_rate_hz"]),
            "io_speed_hz": compact_number(engineering["io_speed_hz"]),
            "flash_wait_states": compact_number(engineering["flash_wait_states"]),
            "flash_bank_count": compact_number(engineering["flash_bank_count"]),
            "flash_ecc_present": engineering["flash_ecc_present"],
            "cache_present": engineering["cache_present"],
            "ram_ecc_present": engineering["ram_ecc_present"],
            "ram_architecture": engineering["ram_architecture"],
            "ram_exclusive_present": engineering["ram_exclusive_present"],
            "gpio_count": compact_number(gpio_count),
            "spi_count": compact_number(spi_count),
            "i2c_count": compact_number(i2c_count),
            "usart_count": compact_number(usart_count),
            "uart_count": compact_number(uart_count),
            "can_count": compact_number(can_count),
            "can_fd_present": "yes" if any(item.get("type") == "CAN" and "fd" in item.get("name", "").lower() for item in features) else ("no" if can_count is not None else "unknown"),
            "usb_count": compact_number(usb_count),
            "usb_device_count": compact_number(usb_device_count),
            "usb_host_count": compact_number(usb_host_count),
            "ethernet_count": compact_number(ethernet_count),
            "dma_source_quantity": compact_number(dma_count),
            "rng_count": compact_number(rng_count),
            "i2s_count": compact_number(i2s_count),
            "lin_count": compact_number(lin_count),
            "configurable_serial_count": compact_number(configurable_serial_count),
            "usb_otg_count": compact_number(usb_otg_count),
            "sdio_count": compact_number(sdio_count),
            "watchdog_count": compact_number(watchdog_count),
            "rtc_present": rtc_present,
            "comparator_count": compact_number(comparator_count),
            "opamp_count": compact_number(opamp_count),
            "touch_source_quantity": compact_number(touch_count),
            "camera_interface_count": compact_number(camera_count),
            "display_controller_count": compact_number(display_count),
            "external_bus_count": compact_number(external_bus_count),
            "external_bus_types_json": json.dumps(named_external_buses, ensure_ascii=False),
            "external_interrupt_count": compact_number(external_interrupt_count),
            "temperature_sensor_count": compact_number(temperature_sensor_count),
            "crypto_accelerator_present": crypto_present,
            "power_measurements_json": json.dumps(power, ensure_ascii=False, sort_keys=True),
            "power_data_status": "official_source_measurement" if power else "not_found",
            "peripheral_inventory_json": json.dumps(inventory, ensure_ascii=False, sort_keys=True),
            "accelerators_json": json.dumps(accelerators, ensure_ascii=False, sort_keys=True),
            "special_features_json": json.dumps(special_features, ensure_ascii=False, sort_keys=True),
            "pending_feature_candidates_json": json.dumps(candidates, ensure_ascii=False, sort_keys=True),
            "missing_key_fields": ";".join(missing),
            "source_id": row["source_id"],
            "verification_status": "derived_from_source_metadata",
        }
        capabilities.append(capability)

        clock_mhz = max_clock_hz / 1_000_000 if max_clock_hz else None
        clock_component = log_score(clock_mhz, 8, 800)
        core_component = core_rank(core_names)
        compute_score, _ = weighted_score([(clock_component, 60), (core_component, 40)])

        flash_kb = number(row.get("flash_bytes"))
        ram_kb = number(row.get("ram_bytes"))
        flash_component = log_score(flash_kb / 1024 if flash_kb else None, 16, 4096)
        ram_component = log_score(ram_kb / 1024 if ram_kb else None, 4, 1024)
        memory_score, _ = weighted_score([(flash_component, 55), (ram_component, 45)])

        peripheral_pieces: list[float] = []
        if timer_count is not None:
            peripheral_pieces.append(min(25, timer_count / 12 * 25))
        if adc_channel_count is not None:
            peripheral_pieces.append(min(12, adc_channel_count / 16 * 12))
        if dac_source_quantity is not None:
            peripheral_pieces.append(min(5, dac_source_quantity / 2 * 5))
        serial_total = sum(value or 0 for value in (spi_count, i2c_count, usart_count, uart_count))
        if any(value is not None for value in (spi_count, i2c_count, usart_count, uart_count)):
            peripheral_pieces.append(min(20, serial_total / 12 * 20))
        if can_count is not None:
            peripheral_pieces.append(10 if can_count > 0 else 0)
        if usb_count is not None or usb_device_count is not None or usb_host_count is not None:
            peripheral_pieces.append(10 if (usb_count or 0) + (usb_device_count or 0) + (usb_host_count or 0) > 0 else 0)
        if ethernet_count is not None:
            peripheral_pieces.append(8 if ethernet_count > 0 else 0)
        if gpio_count is not None:
            peripheral_pieces.append(min(10, gpio_count / 100 * 10))
        peripheral_score = min(100.0, sum(peripheral_pieces)) if peripheral_pieces else None

        accelerator_score = None
        if accelerators:
            names = " ".join(item.get("name", "").lower() for item in accelerators)
            accelerator_score = 30.0
            if any(token in names for token in ("npu", "neural")):
                accelerator_score = 100.0
            elif any(token in names for token in ("cordic", "fmac", "tmu", "math", "trigonometric")):
                accelerator_score = 75.0
            elif any(token in names for token in ("chrom-art", "dma2d", "graphics", "jpeg")):
                accelerator_score = 65.0
            elif any(token in names for token in ("aes", "hash", "crypto")):
                accelerator_score = 50.0

        # FPU is a first-class part of this dimension. An explicit absence is
        # still complete data (zero capability), while an unknown stays out of
        # both the score and its coverage denominator.
        fpu_score = None if fpu_present == "unknown" else (60.0 if fpu_present == "yes" else 0.0)
        accelerator_score, _ = weighted_score(
            [(fpu_score, 35), (accelerator_score, 65)]
        )

        overall_score, score_coverage = weighted_score(
            [(compute_score, 35), (memory_score, 25), (peripheral_score, 25), (accelerator_score, 15)]
        )
        scores.append(
            {
                "device_id": row["device_id"],
                "manufacturer": row["manufacturer"],
                "device_name": row["device_name"],
                "selection_index": round(overall_score) if overall_score is not None else "",
                "compute_score": round(compute_score) if compute_score is not None else "",
                "memory_score": round(memory_score) if memory_score is not None else "",
                "peripheral_score": round(peripheral_score) if peripheral_score is not None else "",
                "accelerator_score": round(accelerator_score) if accelerator_score is not None else "",
                "score_coverage_percent": score_coverage,
                "benchmark_coremark": "",
                "benchmark_dmips": "",
                "benchmark_source_url": "",
                "benchmark_status": "not_imported",
                "scoring_model_version": MODEL_VERSION,
                "score_status": "derived_selection_index_not_benchmark",
            }
        )

    capability_fields = [
        "device_id", "manufacturer", "device_name", "core_names", "primary_core",
        "core_count", "max_clock_hz", "operating_voltage_min_v", "operating_voltage_max_v",
        "fpu_present", "mpu_present",
        "dsp_extension_present", "trustzone_present", "timer_count", "timer_width_bits",
        "pwm_source_quantity", "adc_source_quantity", "adc_quantity_semantics",
        "adc_unit_count", "adc_channel_count",
        "adc_resolution_bits", "dac_source_quantity", "dac_resolution_bits", "gpio_count",
        "adc_sample_rate_hz", "dac_sample_rate_hz", "io_speed_hz", "flash_wait_states",
        "flash_bank_count", "flash_ecc_present", "cache_present", "ram_ecc_present",
        "ram_architecture", "ram_exclusive_present",
        "spi_count", "i2c_count", "usart_count", "uart_count", "can_count",
        "can_fd_present", "usb_count", "usb_device_count", "usb_host_count", "ethernet_count",
        "dma_source_quantity", "rng_count", "i2s_count", "lin_count", "configurable_serial_count", "usb_otg_count",
        "sdio_count", "watchdog_count", "rtc_present", "comparator_count", "opamp_count",
        "touch_source_quantity", "camera_interface_count", "display_controller_count",
        "external_bus_count", "external_bus_types_json", "external_interrupt_count", "temperature_sensor_count",
        "crypto_accelerator_present", "power_measurements_json", "power_data_status",
        "peripheral_inventory_json", "accelerators_json",
        "special_features_json", "pending_feature_candidates_json", "missing_key_fields",
        "source_id", "verification_status",
    ]
    score_fields = [
        "device_id", "manufacturer", "device_name", "selection_index", "compute_score",
        "memory_score", "peripheral_score", "accelerator_score", "score_coverage_percent",
        "benchmark_coremark", "benchmark_dmips", "benchmark_source_url", "benchmark_status",
        "scoring_model_version", "score_status",
    ]
    write_csv(args.data_dir / "device-capabilities.csv", capability_fields, capabilities)
    write_csv(args.data_dir / "device-scores.csv", score_fields, scores)
    print(
        json.dumps(
            {
                "devices": len(devices),
                "capability_records": len(capabilities),
                "score_records": len(scores),
                "records_with_timer_count": sum(bool(row["timer_count"] != "") for row in capabilities),
                "records_with_accelerators": sum(json.loads(row["accelerators_json"]) != [] for row in capabilities),
                "data_dir": str(args.data_dir.resolve()),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

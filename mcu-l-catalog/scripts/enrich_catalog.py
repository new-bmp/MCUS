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
        if item.get("source_kind") in {"cubemx_device_db", "microchip_atdf", "puya_device_header", "infineon_device_db", "espressif_idf_soc_caps"}
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
        for value in re.findall(r"(?<!\d)(8|10|12|14|16|24|32)(?:-?[Bb]it)?", name):
            result.add(int(value))
        raw = feature.get("m", "")
        for value in re.findall(r"\d+(?:\.\d+)?", raw):
            numeric = float(value)
            if numeric <= 32:
                result.add(int(numeric))
    return sorted(result)


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
    return int(sum(explicit)) if explicit else len(processors)


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
    accelerator_types = {"npu", "crypto"}
    accelerator_phrases = (
        "accelerator", "neural", "chrom-art",
        "dma2d", "graphics", "trigonometric", "math unit", "jpeg codec",
    )
    accelerator_token_pattern = re.compile(
        r"(?<![a-z0-9])(?:npu|tmu|cordic|fmac|crc|aes|sha|hash|pka)\d*(?![a-z0-9])",
        re.I,
    )
    special_types = {"rng", "touch", "camera", "lcd", "glcd", "application", "security", "vendorcapability"}
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
    "usbotg": "connectivity", "eth": "connectivity", "sdio": "connectivity",
    "sdiohost": "connectivity", "sdioslave": "connectivity", "i3c": "connectivity",
    "wifi": "wireless", "wifi6": "wireless", "bluetooth": "wireless",
    "ieee802154": "wireless",
    "mipi": "connectivity", "mpserial": "connectivity", "com": "connectivity",
    "comother": "connectivity",
    "camera": "display_multimedia", "lcd": "display_multimedia", "glcd": "display_multimedia",
    "extbus": "memory_bus", "dma": "memory_bus",
    "crypto": "security", "security": "security", "rng": "security", "npu": "accelerator", "coreother": "accelerator",
    "rmt": "timing", "ledpwm": "timing", "mcpwm": "timing",
    "psram": "memory_bus", "rtc_ram": "memory_bus", "hall": "analog", "tof": "connectivity",
    "nvic": "system", "extint": "system", "pll": "clock", "xtal": "clock",
    "intrc": "clock", "clockother": "clock", "powermode": "power",
    "powerother": "power", "consumption": "power", "ios": "gpio", "i/o": "gpio",
    "gpiopadcount": "gpio", "spiperipheraltotal": "connectivity",
}


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
        if feature.get("source_kind") in {"cubemx_device_db", "microchip_atdf", "puya_device_header", "infineon_device_db", "espressif_idf_soc_caps"}
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
            if item.get("source_kind") in {"cubemx_device_db", "microchip_atdf", "puya_device_header", "infineon_device_db", "espressif_idf_soc_caps"}
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
        # External ADC-capable pin counts require a pinout table or datasheet.
        # Never derive them from the generic CMSIS feature n/count attribute.
        adc_external_pin_features = [
            item for item in features
            if item.get("type", "").lower() == "adcexternalpins"
            and item.get("source_kind") in {"cubemx_device_db", "microchip_atdf"}
        ]
        adc_external_pin_values = [
            quantity(item) for item in adc_external_pin_features
            if quantity(item) is not None
        ]
        adc_external_pin_count = (
            sum(adc_external_pin_values) if adc_external_pin_values else None
        )
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
                            adc_channel_count, adc_external_pin_count,
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
            "fpu_present": fpu_present,
            "mpu_present": bool_from_processors(processors, "Dmpu"),
            "dsp_extension_present": bool_from_processors(processors, "Ddsp"),
            "trustzone_present": bool_from_processors(processors, "Dtz"),
            "timer_count": compact_number(timer_count),
            "timer_width_bits": ";".join(str(value) for value in bit_widths(features, {"Timer"})),
            "pwm_source_quantity": compact_number(pwm_count),
            "adc_source_quantity": compact_number(adc_source_quantity),
            "adc_quantity_semantics": adc_semantics,
            "adc_unit_count": compact_number(adc_unit_count),
            "adc_channel_count": compact_number(adc_channel_count),
            "adc_external_pin_count": compact_number(adc_external_pin_count),
            "adc_resolution_bits": ";".join(str(value) for value in bit_widths(features, {"ADC", "ADCUnits", "A/D"})),
            "dac_source_quantity": compact_number(dac_source_quantity),
            "dac_resolution_bits": ";".join(str(value) for value in bit_widths(features, {"DAC", "D/A"})),
            "gpio_count": compact_number(gpio_count),
            "spi_count": compact_number(spi_count),
            "i2c_count": compact_number(i2c_count),
            "usart_count": compact_number(usart_count),
            "uart_count": compact_number(uart_count),
            "can_count": compact_number(can_count),
            "can_fd_present": "yes" if any(item.get("type") == "CAN" and "fd" in item.get("name", "").lower() for item in features) else ("no" if can_count is not None else "unknown"),
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
            "external_interrupt_count": compact_number(external_interrupt_count),
            "temperature_sensor_count": compact_number(temperature_sensor_count),
            "crypto_accelerator_present": crypto_present,
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
        if adc_source_quantity is not None:
            peripheral_pieces.append(min(12, adc_source_quantity / 16 * 12))
        if dac_source_quantity is not None:
            peripheral_pieces.append(min(5, dac_source_quantity / 2 * 5))
        serial_total = sum(value or 0 for value in (spi_count, i2c_count, usart_count, uart_count))
        if any(value is not None for value in (spi_count, i2c_count, usart_count, uart_count)):
            peripheral_pieces.append(min(20, serial_total / 12 * 20))
        if can_count is not None:
            peripheral_pieces.append(10 if can_count > 0 else 0)
        if usb_device_count is not None or usb_host_count is not None:
            peripheral_pieces.append(10 if (usb_device_count or 0) + (usb_host_count or 0) > 0 else 0)
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
        "core_count", "max_clock_hz", "fpu_present", "mpu_present",
        "dsp_extension_present", "trustzone_present", "timer_count", "timer_width_bits",
        "pwm_source_quantity", "adc_source_quantity", "adc_quantity_semantics",
        "adc_unit_count", "adc_channel_count", "adc_external_pin_count",
        "adc_resolution_bits", "dac_source_quantity", "dac_resolution_bits", "gpio_count",
        "spi_count", "i2c_count", "usart_count", "uart_count", "can_count",
        "can_fd_present", "usb_device_count", "usb_host_count", "ethernet_count",
        "dma_source_quantity", "rng_count", "i2s_count", "lin_count", "configurable_serial_count", "usb_otg_count",
        "sdio_count", "watchdog_count", "rtc_present", "comparator_count", "opamp_count",
        "touch_source_quantity", "camera_interface_count", "display_controller_count",
        "external_bus_count", "external_interrupt_count", "temperature_sensor_count",
        "crypto_accelerator_present", "peripheral_inventory_json", "accelerators_json",
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

#!/usr/bin/env python3
"""Import Renesas MCU options from the official product selector APIs.

The selector exposes exact base part numbers, orderable part numbers, package
data, and device-level resources.  No suffix combinations are generated.  RZ
MPUs are outside this adapter's scope and are explicitly excluded.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable

from import_cmsis_packs import parse_pdsc


MANUFACTURER = "Renesas"
ORIGIN = "https://www.renesas.com"
USER_AGENT = "MCUS-Catalog/1.0 (+https://github.com/new-bmp/MCUS)"
RA_PDSC_URL = "https://www2.renesas.eu/Keil_MDK_Packs/Renesas.RA_DFP.pdsc"

CATEGORIES = {
    "ra": {
        "url": f"{ORIGIN}/en/products/microcontrollers-microprocessors/ra-cortex-m-mcus/product-selector",
        "family": "RA",
        "series": "RA",
        "product_type": "general_purpose_mcu",
    },
    "rx": {
        "url": f"{ORIGIN}/en/products/microcontrollers-microprocessors/rx-32-bit-performance-efficiency-mcus/product-selector",
        "family": "RX",
        "series": "RX",
        "product_type": "general_purpose_mcu",
    },
    "rl78": {
        "url": f"{ORIGIN}/en/products/microcontrollers-microprocessors/rl78-low-power-8-16-bit-mcus/product-selector",
        "family": "RL78",
        "series": "RL78",
        "product_type": "low_power_mcu",
    },
    "rh850": {
        "url": f"{ORIGIN}/en/products/microcontrollers-microprocessors/rh850-automotive-mcus/product-selector",
        "family": "RH850",
        "series": "RH850",
        "product_type": "automotive_mcu",
    },
    "synergy": {
        "url": f"{ORIGIN}/en/products/microcontrollers-microprocessors/renesas-synergy-platform-mcus/product-selector",
        "family": "Synergy",
        "series": "Synergy",
        "product_type": "general_purpose_mcu",
    },
    "riscv": {
        "url": f"{ORIGIN}/en/products/microcontrollers-microprocessors/risc-v/product-selector",
        "family": "Renesas RISC-V",
        "series": "Renesas RISC-V",
        "product_type": "general_purpose_mcu",
    },
}


# Some selector option endpoints omit fields that are constant for every part
# in a product group.  A group value is inherited only when the official group
# row contains exactly one distinct value.  Aggregate resources are handled as
# a unit so a variable timer or ADC field cannot become a misleading partial
# total on the device row.
INHERITABLE_GROUP_FIELDS = {
    "field__cpu_architecture", "field__main_cpu", "field__cpu", "field__sub_cpu",
    "field__floating_point_unit", "field__operating_freq_max_",
    "field__operating_freq_max", "field__operating_freq_of_sub_cpu_max",
    "field__program_memory", "field__data_flash", "field__ram_size_kb",
    "field__i_o_ports", "field__dma_ch", "field__temperature_sensor_ch_",
    "field__watchdog_timer", "field__sci_ch", "field__spi_ch_", "field__i2c",
    "field__i3c_ch_", "field__can_ch_", "field__can_fd_ch", "field__lin",
    "field__sdhi_ch_", "field__32_bit_enhanced_timer_ch", "field__qspi",
    "field__qspi_interface", "field__ospi_ch_", "field__external_interrupt_pins",
    "field__32_bit_timer_ch_", "field__16_bit_timer_ch_", "field__8_bit_timer_ch_",
    "field__asynchronous_general_purpose_timer_interval_timer_ch",
    "field__16_bit_a_d_converter_ch_", "field__14_bit_a_d_converter_ch_",
    "field__12_bit_a_d_converter_ch_", "field__10_bit_a_d_converter_ch_",
    "field__24_bit_sigma_delta_a_d_converter_ch_", "field__16_bit_d_a_converter_ch",
    "field__12_bit_d_a_converter_ch_", "field__10_bit_d_a_converter_ch_",
    "field__8_bit_d_a_converter_ch_", "field__analog_comparator_ch",
    "field__high_speed_analog_comparator_ch_", "field__opamp_ch_", "field__pga_ch_",
    "field__capacitive_touch_sensing_unit_ch_", "field__ssi_ch",
    "field__serial_sound_interface_ch_", "field__ethernet_ch", "field__ethercat_ch",
    "field__rtc", "field__high_resolution_output_timer", "field__graphics_lcd_controller",
    "field__graphics_lcd_controller_y_n", "field__segment_lcd_controller",
    "field__camera_i_f", "field__image_codec", "field__2d_drawing_engine",
    "field__gpu", "field__accelerator", "field__external_memory_bus_bit_",
    "field__memory_interface", "field__usb_ports", "field__usbfs_host_ch_device_ch_",
    "field__usbhs_host_ch_device_ch_", "field__usb3_0_ch", "field__dma_dtc",
    "field__wireless_yes_no", "field__security_encryption", "field__pdm_interface",
}

AGGREGATE_GROUPS = (
    {
        "field__32_bit_timer_ch_", "field__16_bit_timer_ch_", "field__8_bit_timer_ch_",
        "field__asynchronous_general_purpose_timer_interval_timer_ch",
        "field__high_resolution_output_timer",
    },
    {
        "field__16_bit_a_d_converter_ch_", "field__14_bit_a_d_converter_ch_",
        "field__12_bit_a_d_converter_ch_", "field__10_bit_a_d_converter_ch_",
        "field__24_bit_sigma_delta_a_d_converter_ch_",
    },
    {
        "field__16_bit_d_a_converter_ch", "field__12_bit_d_a_converter_ch_",
        "field__10_bit_d_a_converter_ch_", "field__8_bit_d_a_converter_ch_",
    },
)


class DrupalSettingsParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.capture = False
        self.buffers: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        self.capture = (
            tag == "script"
            and values.get("type") == "application/json"
            and values.get("data-drupal-selector") == "drupal-settings-json"
        )

    def handle_data(self, data: str) -> None:
        if self.capture:
            self.buffers.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "script":
            self.capture = False


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "unknown"


def scalar(value: Any, default: Any = "") -> Any:
    if isinstance(value, list):
        return value[0] if value else default
    return default if value is None else value


def text(value: Any) -> str:
    value = scalar(value, "")
    return str(value).strip() if value is not None else ""


def number(value: Any) -> float | None:
    value = scalar(value, "")
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def whole(value: float | None) -> int | str:
    if value is None:
        return ""
    return int(value) if value.is_integer() else value


def list_values(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value).split(",") if item.strip()]


def populated(value: Any) -> bool:
    return value not in (None, "", [])


def distinct_source_values(value: Any) -> list[Any]:
    values = value if isinstance(value, list) else [value]
    result: list[Any] = []
    seen: set[str] = set()
    for item in values:
        if not populated(item):
            continue
        key = json.dumps(item, ensure_ascii=False, sort_keys=True)
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def inherit_group_singletons(option: dict[str, Any], group: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    row = dict(option)
    inherited: list[str] = []
    blocked: set[str] = set()
    for fields in AGGREGATE_GROUPS:
        if any(len(distinct_source_values(group.get(field))) > 1 for field in fields):
            blocked.update(fields)
    for field in sorted(INHERITABLE_GROUP_FIELDS - blocked):
        if populated(row.get(field)):
            continue
        values = distinct_source_values(group.get(field))
        if len(values) == 1:
            row[field] = values[0]
            inherited.append(field)
    return row, inherited


def fetch(
    url: str,
    cache_path: Path,
    *,
    refresh: bool,
    timeout: float,
    retries: int = 3,
) -> bytes:
    if cache_path.exists() and not refresh:
        return cache_path.read_bytes()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": USER_AGENT, "Accept": "application/json,text/html;q=0.9,*/*;q=0.8"},
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = response.read()
            cache_path.write_bytes(payload)
            return payload
        except (OSError, urllib.error.URLError, urllib.error.HTTPError) as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(0.75 * (2**attempt))
    assert last_error is not None
    raise last_error


def parse_settings(payload: bytes) -> dict[str, Any]:
    parser = DrupalSettingsParser()
    parser.feed(payload.decode("utf-8", errors="replace"))
    if not parser.buffers:
        raise ValueError("Drupal settings JSON was not present on the selector page")
    return json.loads("".join(parser.buffers))


def write_csv(path: Path, fields: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def exact_feature(
    feature_type: str,
    name: str,
    count: float | int | str | None = None,
    *,
    bits: int | None = None,
    category: str = "",
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "type": feature_type,
        "name": name,
        "source_kind": "renesas_product_selector",
    }
    if count not in (None, ""):
        record["count"] = str(whole(float(count)))
        record["n"] = str(whole(float(count)))
    if bits is not None:
        record["m"] = str(bits)
    if category:
        record["category"] = category
    return record


def boolean_count(value: Any) -> int | None:
    normalized = text(value).lower()
    if not normalized:
        return None
    if normalized in {"no", "none", "n/a", "not available", "0", "false"}:
        return 0
    return 1


def numeric_feature(
    result: list[dict[str, Any]],
    row: dict[str, Any],
    field: str,
    feature_type: str,
    label: str,
    *,
    bits: int | None = None,
    category: str = "",
) -> None:
    value = number(row.get(field))
    if value is not None:
        result.append(exact_feature(feature_type, label, value, bits=bits, category=category))


def boolean_feature(
    result: list[dict[str, Any]],
    row: dict[str, Any],
    field: str,
    feature_type: str,
    label: str,
    *,
    category: str = "",
) -> None:
    value = boolean_count(row.get(field))
    if value is not None:
        result.append(exact_feature(feature_type, label, value, category=category))


def parse_channel_descriptions(values: Any) -> list[tuple[str, str, int, int | None, str]]:
    parsed: list[tuple[str, str, int, int | None, str]] = []
    labels = {
        "SPI": ("SPI", "SPI channels"),
        "I2C": ("I2C", "I2C channels"),
        "IIC": ("I2C", "IIC channels"),
        "UART": ("UART", "UART channels"),
        "SCI": ("UART", "SCI / UART channels"),
    }
    for value in list_values(values):
        combined = re.search(
            r"SPI\s*/\s*I2C\s*/\s*UART\s*x\s*(\d+)\s*/\s*(\d+)\s*/\s*(\d+)\s*ch",
            value,
            re.I,
        )
        if combined:
            for name, count in zip(("SPI", "I2C", "UART"), combined.groups()):
                feature_type, label = labels[name]
                parsed.append((feature_type, label, int(count), None, "connectivity"))
            continue
        single = re.search(r"\b(SCI|SPI|I2C|IIC|UART)\s*x\s*(\d+)\s*ch", value, re.I)
        if single:
            name = single.group(1).upper()
            feature_type, label = labels[name]
            parsed.append((feature_type, label, int(single.group(2)), None, "connectivity"))
    return parsed


def parse_typed_channel_descriptions(
    values: Any,
    feature_type: str,
    noun: str,
    category: str,
) -> list[tuple[str, str, int, int | None, str]]:
    parsed: list[tuple[str, str, int, int | None, str]] = []
    for value in list_values(values):
        match = re.search(r"(\d+)\s*-?\s*Bit\s*x\s*(\d+)\s*ch", value, re.I)
        if match:
            bits, count = int(match.group(1)), int(match.group(2))
            parsed.append((feature_type, f"{bits}-bit {noun} channels", count, bits, category))
    return parsed


def consensus_descriptions(
    parsed: list[tuple[str, str, int, int | None, str]],
) -> list[tuple[str, str, int, int | None, str]]:
    grouped: dict[tuple[str, int | None], list[tuple[str, str, int, int | None, str]]] = {}
    for item in parsed:
        grouped.setdefault((item[0], item[3]), []).append(item)
    result: list[tuple[str, str, int, int | None, str]] = []
    for items in grouped.values():
        counts = {item[2] for item in items}
        if len(counts) == 1:
            result.append(items[0])
    return result


def add_description_features(
    result: list[dict[str, Any]],
    row: dict[str, Any],
    group: dict[str, Any],
    field: str,
    parser: Any,
    *,
    allow_exact_duplicates: bool = False,
) -> None:
    exact_values = row.get(field)
    parsed = parser(exact_values) if populated(exact_values) else consensus_descriptions(parser(group.get(field)))
    existing = {
        (item.get("type", "").upper(), int(item["m"]) if str(item.get("m", "")).isdigit() else None)
        for item in result
    }
    for feature_type, label, count, bits, category in parsed:
        key = (feature_type.upper(), bits)
        # Exact option descriptions may contain independent blocks of the same
        # protocol, so all of them are additive. Group fallbacks require one
        # unambiguous count per protocol/width and never duplicate exact data.
        if key in existing and (not populated(exact_values) or not allow_exact_duplicates):
            continue
        result.append(exact_feature(feature_type, label, count, bits=bits, category=category))
        existing.add(key)


def add_riscv_features(result: list[dict[str, Any]], row: dict[str, Any], group: dict[str, Any]) -> None:
    add_description_features(result, row, group, "field__timer", lambda value: parse_typed_channel_descriptions(value, "Timer", "timer", "timing"))
    add_description_features(result, row, group, "field__adc", lambda value: parse_typed_channel_descriptions(value, "ADC", "ADC", "analog"))
    add_description_features(result, row, group, "field__dac", lambda value: parse_typed_channel_descriptions(value, "DAC", "DAC", "analog"))
    add_description_features(
        result,
        row,
        group,
        "field__comm_interface",
        parse_channel_descriptions,
        allow_exact_duplicates=True,
    )

    dma_values = row.get("field__dma") if populated(row.get("field__dma")) else group.get("field__dma")
    dma_counts = {
        int(match.group(1))
        for value in list_values(dma_values)
        if (match := re.search(r"\bDMA\s*(\d+)\s*ch", value, re.I))
    }
    if len(dma_counts) == 1:
        result.append(exact_feature("DMA", "DMA channels", dma_counts.pop(), category="memory_bus"))
    if any(value.strip().upper() == "DTC" for value in list_values(dma_values)):
        result.append(exact_feature("VendorCapability", "Data Transfer Controller (DTC)", 1, category="memory_bus"))


def selector_features(row: dict[str, Any], group: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    group = group or {}
    mappings = (
        ("field__i_o_ports", "IOs", "General-purpose I/O ports", None, "gpio"),
        ("field__dma_ch", "DMA", "DMA channels", None, "memory_bus"),
        ("field__temperature_sensor_ch_", "TempSens", "Temperature sensor channels", None, "analog"),
        ("field__watchdog_timer", "WDT", "Watchdog timers", None, "timing"),
        ("field__sci_ch", "UART", "SCI / UART channels", None, "connectivity"),
        ("field__spi_ch_", "SPI", "SPI channels", None, "connectivity"),
        ("field__i2c", "I2C", "I2C channels", None, "connectivity"),
        ("field__i3c_ch_", "I3C", "I3C channels", None, "connectivity"),
        ("field__can_ch_", "CAN", "Classical CAN channels", None, "connectivity"),
        ("field__can_fd_ch", "CAN", "CAN FD channels", None, "connectivity"),
        ("field__lin", "LIN", "LIN channels", None, "connectivity"),
        ("field__sdhi_ch_", "SDIO", "SDHI / SD card host channels", None, "connectivity"),
        ("field__32_bit_enhanced_timer_ch", "PWM", "PWM output pins", None, "timing"),
        ("field__32_bit_timer_ch_", "Timer", "32-bit timer channels", 32, "timing"),
        ("field__16_bit_timer_ch_", "Timer", "16-bit timer channels", 16, "timing"),
        ("field__8_bit_timer_ch_", "Timer", "8-bit timer channels", 8, "timing"),
        ("field__asynchronous_general_purpose_timer_interval_timer_ch", "Timer", "Asynchronous / interval timer channels", None, "timing"),
        ("field__16_bit_a_d_converter_ch_", "ADC", "16-bit ADC channels", 16, "analog"),
        ("field__14_bit_a_d_converter_ch_", "ADC", "14-bit ADC channels", 14, "analog"),
        ("field__12_bit_a_d_converter_ch_", "ADC", "12-bit ADC channels", 12, "analog"),
        ("field__10_bit_a_d_converter_ch_", "ADC", "10-bit ADC channels", 10, "analog"),
        ("field__24_bit_sigma_delta_a_d_converter_ch_", "ADC", "24-bit sigma-delta ADC channels", 24, "analog"),
        ("field__16_bit_d_a_converter_ch", "DAC", "16-bit DAC channels", 16, "analog"),
        ("field__12_bit_d_a_converter_ch_", "DAC", "12-bit DAC channels", 12, "analog"),
        ("field__10_bit_d_a_converter_ch_", "DAC", "10-bit DAC channels", 10, "analog"),
        ("field__8_bit_d_a_converter_ch_", "DAC", "8-bit DAC channels", 8, "analog"),
        ("field__analog_comparator_ch", "COMP", "Analog comparator channels", None, "analog"),
        ("field__opamp_ch_", "AnalogOther", "Operational amplifier channels", None, "analog"),
        ("field__pga_ch_", "AnalogOther", "Programmable gain amplifier channels", None, "analog"),
        ("field__capacitive_touch_sensing_unit_ch_", "Touch", "Capacitive touch channels", None, "analog"),
        ("field__ssi_ch", "I2S", "SSI digital audio channels", None, "display_multimedia"),
        ("field__serial_sound_interface_ch_", "I2S", "Serial sound interface channels", None, "display_multimedia"),
        ("field__ethernet_ch", "ETH", "Ethernet channels", None, "connectivity"),
        ("field__ethercat_ch", "ETH", "EtherCAT channels", None, "connectivity"),
        ("field__qspi", "ExtBus", "QSPI memory interfaces", None, "memory_bus"),
        ("field__qspi_interface", "ExtBus", "QSPI memory interfaces", None, "memory_bus"),
        ("field__ospi_ch_", "ExtBus", "OSPI memory interfaces", None, "memory_bus"),
        ("field__high_speed_analog_comparator_ch_", "COMP", "High-speed analog comparator channels", None, "analog"),
        ("field__external_interrupt_pins", "ExtInt", "External interrupt inputs", None, "gpio"),
    )
    comm_text = " ".join(list_values(row.get("field__comm_interface"))).upper()
    for field, feature_type, label, bits, category in mappings:
        if field == "field__i2c" and ("I2C" in comm_text or "IIC" in comm_text):
            continue
        numeric_feature(result, row, field, feature_type, label, bits=bits, category=category)

    booleans = (
        ("field__rtc", "RTC", "Real-time clock", "timing"),
        ("field__high_resolution_output_timer", "Timer", "High-resolution output timer", "timing"),
        ("field__graphics_lcd_controller", "GLCD", "Graphics LCD controller", "display_multimedia"),
        ("field__segment_lcd_controller", "LCD", "Segment LCD controller", "display_multimedia"),
        ("field__camera_i_f", "Camera", "Parallel camera interface", "display_multimedia"),
        ("field__image_codec", "Accelerator", "Image codec", "accelerator"),
        ("field__2d_drawing_engine", "Accelerator", "2D drawing engine", "accelerator"),
        ("field__gpu", "Accelerator", "Graphics processing unit", "accelerator"),
        ("field__accelerator", "Accelerator", "Renesas hardware accelerator", "accelerator"),
        ("field__external_memory_bus_bit_", "ExtBus", "External memory bus", "memory_bus"),
        ("field__memory_interface", "ExtBus", "External memory interface", "memory_bus"),
    )
    for field, feature_type, label, category in booleans:
        boolean_feature(result, row, field, feature_type, label, category=category)

    role_counts_found = False
    for field, speed in (
        ("field__usbfs_host_ch_device_ch_", "USB FS"),
        ("field__usbhs_host_ch_device_ch_", "USB HS"),
        ("field__usb3_0_ch", "USB SuperSpeed"),
    ):
        raw = text(row.get(field))
        match = re.search(r"\(\s*(\d+)\s*/\s*(\d+)\s*\)", raw)
        if not match:
            continue
        role_counts_found = True
        result.append(exact_feature("USBH", f"{speed} host channels", int(match.group(1)), category="connectivity"))
        result.append(exact_feature("USBD", f"{speed} device channels", int(match.group(2)), category="connectivity"))
    if not role_counts_found:
        numeric_feature(result, row, "field__usb_ports", "USB", "USB ports", category="connectivity")

    dma_dtc = text(row.get("field__dma_dtc"))
    if dma_dtc and dma_dtc.lower() != "no":
        result.append(exact_feature("VendorCapability", f"DMA / DTC: {dma_dtc}", 1, category="memory_bus"))

    wireless = " ".join(list_values(row.get("field__wireless_yes_no")))
    if wireless:
        lowered = wireless.lower()
        absent = lowered in {"no", "none", "0"}
        if absent or "wi-fi" in lowered or "wifi" in lowered:
            result.append(exact_feature("WiFi", wireless if not absent else "Wi-Fi", 0 if absent else 1, category="wireless"))
        if absent or "bluetooth" in lowered or "ble" in lowered:
            result.append(exact_feature("Bluetooth", wireless if not absent else "Bluetooth", 0 if absent else 1, category="wireless"))

    security = ", ".join(list_values(row.get("field__security_encryption")))
    if security:
        result.append(exact_feature("Crypto", f"Security and encryption: {security}", 0 if security.lower() == "no" else 1, category="security"))

    for field, label in (
        ("field__image_codec", "Image codec"),
        ("field__2d_drawing_engine", "2D drawing engine"),
        ("field__accelerator", "Renesas accelerator"),
        ("field__gpu", "GPU"),
    ):
        raw = text(row.get(field))
        if raw and raw.lower() not in {"no", "none", "0", "yes"}:
            result.append(exact_feature("VendorCapability", f"{label}: {raw}", 1, category="accelerator"))
    add_riscv_features(result, row, group)
    return result


def processor_records(row: dict[str, Any], cmsis_row: dict[str, Any] | None) -> list[dict[str, str]]:
    if cmsis_row:
        try:
            parsed = json.loads(cmsis_row.get("processor_cores") or "[]")
            if parsed:
                return parsed
        except json.JSONDecodeError:
            pass
    core = text(row.get("field__main_cpu")) or text(row.get("field__cpu"))
    if not core:
        return []
    core = re.sub(r"^(?:ARM|Arm)\s+", "", core).strip()
    processor: dict[str, str] = {
        "Darchitecture": text(row.get("field__cpu_architecture")) or core,
        "Dcore": core,
    }
    clock_mhz = number(row.get("field__operating_freq_max_"))
    if clock_mhz is None:
        clock_mhz = number(row.get("field__operating_freq_max"))
    if clock_mhz is not None:
        processor["Dclock"] = str(int(clock_mhz * 1_000_000))
    fpu = text(row.get("field__floating_point_unit")).lower()
    if fpu:
        processor["Dfpu"] = {
            "no": "NO_FPU",
            "none": "NO_FPU",
            "single": "SP_FPU",
            "double": "DP_FPU",
            "yes": "FPU",
        }.get(fpu, text(row.get("field__floating_point_unit")))
    sub_cpu = text(row.get("field__sub_cpu")).lower()
    if sub_cpu in {"no", "none", "0", "false"}:
        processor["DcoreCount"] = "1"
    elif sub_cpu:
        processor["DsourceDescription"] = "exact core count not stated by official selector"
    return [processor]


def product_scope(row: dict[str, Any], category: str, config: dict[str, str]) -> tuple[str, str, str]:
    product_line = text(row.get("field__group_name")) or text(row.get("product_id"))
    if category == "rl78" and product_line and not product_line.upper().startswith("RL78/"):
        product_line = "RL78/" + product_line
    product_line = product_line or text(row.get("product_id")) or "Unknown"
    family = text(row.get("field__family_name")) or config["family"]
    series = text(row.get("field__series_name")) or config["series"]
    return family, series, product_line


def find_cmsis(base_part: str, cmsis_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    matches = [row for row in cmsis_rows if base_part.upper().startswith(row["device_name"].upper())]
    return max(matches, key=lambda row: len(row["device_name"])) if matches else None


def lifecycle(value: Any) -> str:
    normalized = text(value).lower().replace(" ", "_").replace("-", "_")
    aliases = {"not_recommended_for_new_designs": "nrnd", "last_time_buy": "last_time_buy"}
    return aliases.get(normalized, normalized or "unknown")


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=root / "data" / "vendor-packs" / "renesas")
    parser.add_argument("--cache-dir", type=Path, default=root / "cache" / "renesas")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    observed = utc_now()
    category_payloads: dict[str, dict[str, Any]] = {}
    group_tasks: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    source_rows: list[dict[str, Any]] = []

    for category, config in CATEGORIES.items():
        page = fetch(
            config["url"],
            args.cache_dir / "selectors" / f"{category}.html",
            refresh=args.refresh,
            timeout=args.timeout,
        )
        settings = parse_settings(page)
        products = settings.get("renesas_products") or {}
        node = str(products.get("nid") or "")
        revision = str(products.get("revision_id") or "")
        endpoint = str(products.get("products_table_endpoint") or "")
        if not node or not revision or "{node}" not in endpoint or "{revision}" not in endpoint:
            raise ValueError(f"selector metadata incomplete for {category}")
        selector_url = ORIGIN + endpoint.replace("{node}", node).replace("{revision}", revision)
        payload = json.loads(fetch(
            selector_url,
            args.cache_dir / "selectors" / f"{category}.json",
            refresh=args.refresh,
            timeout=args.timeout,
        ))
        category_payloads[category] = payload
        schema = str(payload.get("schemaVersion") or revision)
        source_rows.append({
            "source_id": f"renesas:selector:{category}@{schema}",
            "source_type": "manufacturer_product_selector_api",
            "publisher": "Renesas Electronics Corporation",
            "title": f"Renesas {config['family']} MCU product selector",
            "url": selector_url,
            "version": f"revision:{revision};schema:{schema}",
            "observed_at": observed,
            "verification_scope": "Official product groups and full orderable part numbers in the live selector snapshot.",
        })
        for group_row in payload.get("rows") or []:
            product_id = text(group_row.get("product_id"))
            title = text(group_row.get("title"))
            parent_path = text(group_row.get("parent_path"))
            if category == "riscv" and (
                product_id.upper().startswith("RZ/")
                or "microprocessor" in title.lower()
                or parent_path.startswith("/products/rz-")
            ):
                continue
            nid = text(group_row.get("nid"))
            vid = text(group_row.get("vid"))
            if not nid or not vid:
                errors.append({"scope": category, "source_url": selector_url, "item": product_id, "error": "missing product nid/vid"})
                continue
            options_url = f"{ORIGIN}/en/api/renesas-products/v1/options/{nid}/{vid}"
            group_tasks.append({
                "category": category,
                "config": config,
                "group": group_row,
                "product_id": product_id,
                "nid": nid,
                "vid": vid,
                "url": options_url,
                "cache": args.cache_dir / "options" / f"{category}-{slug(product_id)}-{vid}.json",
            })

    def get_group(task: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        payload = json.loads(fetch(
            task["url"],
            task["cache"],
            refresh=args.refresh,
            timeout=args.timeout,
        ))
        return task, payload

    group_payloads: list[tuple[dict[str, Any], dict[str, Any]]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(get_group, task): task for task in group_tasks}
        for future in as_completed(futures):
            task = futures[future]
            try:
                group_payloads.append(future.result())
            except Exception as exc:
                errors.append({
                    "scope": task["category"],
                    "source_url": task["url"],
                    "item": task["product_id"],
                    "error": f"{type(exc).__name__}: {exc}",
                })

    cmsis_rows: list[dict[str, Any]] = []
    cmsis_source: dict[str, Any] | None = None
    try:
        pdsc_payload = fetch(
            RA_PDSC_URL,
            args.cache_dir / "cmsis" / "Renesas.RA_DFP.pdsc",
            refresh=args.refresh,
            timeout=args.timeout,
        )
        release_match = re.search(rb'<release\s+version="([^"]+)"', pdsc_payload)
        version = release_match.group(1).decode() if release_match else "live"
        cmsis_rows, cmsis_source = parse_pdsc(pdsc_payload, {
            "pack_vendor": "Renesas",
            "pack_name": "RA_DFP",
            "pack_version": version,
            "pdsc_url": RA_PDSC_URL,
        }, observed)
        source_rows.append(cmsis_source)
    except Exception as exc:
        errors.append({"scope": "ra-cmsis", "source_url": RA_PDSC_URL, "item": "RA_DFP", "error": f"{type(exc).__name__}: {exc}"})

    devices: list[dict[str, Any]] = []
    parts: list[dict[str, Any]] = []
    inherited_group_fields = 0
    devices_with_group_inheritance = 0
    selector_part_paths: dict[str, str] = {}
    selector_parts_by_product: dict[tuple[str, str], list[str]] = {}
    for category, payload in category_payloads.items():
        for group in payload.get("rows") or []:
            product_id = text(group.get("product_id"))
            full_parts = list_values(group.get("partno"))
            paths = list_values(group.get("product_path"))
            selector_parts_by_product[(category, product_id)] = full_parts
            if len(full_parts) == len(paths):
                selector_part_paths.update({part: ORIGIN + path for part, path in zip(full_parts, paths)})

    for task, payload in sorted(group_payloads, key=lambda item: (item[0]["category"], item[0]["product_id"])):
        category = task["category"]
        config = task["config"]
        group = task["group"]
        product_id = task["product_id"]
        group_source_id = f"renesas:options:{category}:{slug(product_id)}@{task['vid']}"
        product_path = text(group.get("parent_path")) or f"/products/{slug(product_id)}"
        product_url = ORIGIN + product_path
        source_rows.append({
            "source_id": group_source_id,
            "source_type": "manufacturer_product_selector_api",
            "publisher": "Renesas Electronics Corporation",
            "title": f"Renesas {product_id} exact product options",
            "url": task["url"],
            "version": f"revision:{task['vid']};schema:{payload.get('schemaVersion', '')}",
            "observed_at": observed,
            "verification_scope": "Exact base part numbers, orderable codes, package data, lifecycle, memory, frequency, and peripheral fields.",
        })
        option_rows = payload.get("rows") or []
        group_base_parts = [text(item.get("partno")) for item in option_rows if text(item.get("partno"))]
        for option in option_rows:
            option_row, inherited_fields = inherit_group_singletons(option, group)
            inherited_group_fields += len(inherited_fields)
            devices_with_group_inheritance += int(bool(inherited_fields))
            base_part = text(option_row.get("partno"))
            if not base_part:
                errors.append({"scope": category, "source_url": task["url"], "item": product_id, "error": "option row missing base part number"})
                continue
            scope_row = dict(option_row)
            for field in ("field__family_name", "field__series_name", "field__group_name", "product_id"):
                if not text(scope_row.get(field)):
                    scope_row[field] = group.get(field)
            family, series, product_line = product_scope(scope_row, category, config)
            line_id = f"renesas::{slug(family)}::{slug(series)}::{slug(product_line)}"
            device_id = f"renesas::{slug(base_part)}"
            cmsis_match = find_cmsis(base_part, cmsis_rows) if category == "ra" else None
            processors = processor_records(option_row, cmsis_match)
            core_names = sorted({item.get("Dcore", "") for item in processors if item.get("Dcore")})
            clock_mhz = number(option_row.get("field__operating_freq_max_"))
            if clock_mhz is None:
                clock_mhz = number(option_row.get("field__operating_freq_max"))
            flash_kb = number(option_row.get("field__program_memory"))
            ram_kb = number(option_row.get("field__ram_size_kb"))
            data_flash_kb = number(option_row.get("field__data_flash"))
            package = text(option_row.get("pkg_type")) or text(option_row.get("code"))
            pins = number(option_row.get("lead_cnt"))
            features = selector_features(option_row, group)
            source_ids = [group_source_id]
            if cmsis_match and cmsis_source:
                source_ids.append(cmsis_source["source_id"])
            documents: list[dict[str, str]] = [{"title": f"Renesas {product_id} product page", "url": product_url}]
            document_path = text(option_row.get("document_path")) or text(group.get("document_path"))
            if document_path:
                documents.append({"title": text(group.get("featured_document_title")) or f"{product_id} datasheet", "url": ORIGIN + document_path})
            memory_regions = []
            if flash_kb is not None:
                memory_regions.append({"name": "Program memory", "type": "Flash", "bytes": int(flash_kb * 1024), "source_kind": "renesas_product_selector"})
            if data_flash_kb is not None:
                memory_regions.append({"name": "Data Flash", "type": "DataFlash", "bytes": int(data_flash_kb * 1024), "source_kind": "renesas_product_selector"})
            if ram_kb is not None:
                memory_regions.append({"name": "RAM", "type": "RAM", "bytes": int(ram_kb * 1024), "source_kind": "renesas_product_selector"})
            wireless = text(option_row.get("field__wireless_yes_no")).lower()
            product_type = "wireless_mcu" if wireless and wireless not in {"no", "none", "0"} else config["product_type"]
            if category == "ra" and series.upper().startswith("RA8"):
                product_type = "high_performance_mcu"
            device_lifecycle = lifecycle(option_row.get("part_status"))
            devices.append({
                "device_id": device_id,
                "product_line_id": line_id,
                "manufacturer": MANUFACTURER,
                "product_type": product_type,
                "architecture_class": "+".join(core_names) if core_names else (text(option_row.get("field__cpu_architecture")) or "unknown"),
                "family": family,
                "series": series,
                "product_line": product_line,
                "device_name": base_part,
                "generic_device_name": cmsis_match["device_name"] if cmsis_match else product_id,
                "manufacturer_variant_code": base_part[len(cmsis_match["device_name"]):] if cmsis_match and base_part.startswith(cmsis_match["device_name"]) else base_part,
                "processor_cores": json.dumps(processors, ensure_ascii=False, sort_keys=True),
                "max_clock_hz": int(clock_mhz * 1_000_000) if clock_mhz is not None else "",
                "flash_bytes": int(flash_kb * 1024) if flash_kb is not None else "",
                "ram_bytes": int(ram_kb * 1024) if ram_kb is not None else "",
                "package_types": package,
                "pin_counts": whole(pins),
                "memory_regions_json": json.dumps(memory_regions, ensure_ascii=False, sort_keys=True),
                "features_json": json.dumps(features, ensure_ascii=False, sort_keys=True),
                "documents_json": json.dumps(documents, ensure_ascii=False, sort_keys=True),
                "svd_files": cmsis_match.get("svd_files", "") if cmsis_match else "",
                "lifecycle": device_lifecycle,
                "source_id": ";".join(source_ids),
                "source_url": product_url,
                "source_version": f"revision:{task['vid']};schema:{payload.get('schemaVersion', '')}",
                "observed_at": observed,
                "verification_status": "manufacturer_product_selector_api",
            })
            full_parts = list_values(option_row.get("opns"))
            if not full_parts:
                full_parts = [
                    value for value in selector_parts_by_product.get((category, product_id), [])
                    if value.upper().startswith(base_part.upper())
                    and base_part == max(
                        (candidate for candidate in group_base_parts if value.upper().startswith(candidate.upper())),
                        key=len,
                    )
                ]
            for full_part in sorted(set(full_parts)):
                if not full_part.upper().startswith(base_part.upper()):
                    errors.append({"scope": category, "source_url": task["url"], "item": full_part, "error": f"orderable code does not match base part {base_part}"})
                    continue
                suffix = full_part[len(base_part):]
                part_url = selector_part_paths.get(full_part) or product_url
                parts.append({
                    "orderable_part_id": f"renesas::{slug(full_part)}",
                    "device_id": device_id,
                    "manufacturer": MANUFACTURER,
                    "family": family,
                    "series": series,
                    "product_line": product_line,
                    "device_name": base_part,
                    "part_number": full_part,
                    "manufacturer_suffix": suffix,
                    "package_code": text(option_row.get("code")),
                    "temperature_grade_code": "",
                    "packing_code": "",
                    "package_name": package,
                    "temperature_range": text(option_row.get("temperature_range")),
                    "packing_form": "; ".join(list_values(option_row.get("tube_tray"))),
                    "lifecycle": device_lifecycle,
                    "source_id": group_source_id,
                    "source_url": part_url,
                    "observed_at": observed,
                    "verification_status": "manufacturer_product_selector_api",
                    "decode_status": "exact_official_orderable_code; suffix_not_inferred",
                })

    # A duplicated selector row would otherwise silently change ownership.
    device_map: dict[str, dict[str, Any]] = {}
    part_map: dict[str, dict[str, Any]] = {}
    for row in devices:
        if row["device_id"] in device_map:
            errors.append({"scope": "deduplicate", "source_url": row["source_url"], "item": row["device_name"], "error": "duplicate base part"})
        else:
            device_map[row["device_id"]] = row
    for row in parts:
        key = row["part_number"].upper()
        if key in part_map:
            errors.append({"scope": "deduplicate", "source_url": row["source_url"], "item": row["part_number"], "error": "duplicate orderable part"})
        else:
            part_map[key] = row
    devices = sorted(device_map.values(), key=lambda row: (row["family"], row["series"], row["product_line"], row["device_name"]))
    parts = sorted(part_map.values(), key=lambda row: row["part_number"])
    source_rows = sorted({row["source_id"]: row for row in source_rows}.values(), key=lambda row: row["source_id"])

    device_fields = [
        "device_id", "product_line_id", "manufacturer", "product_type", "architecture_class",
        "family", "series", "product_line", "device_name", "generic_device_name",
        "manufacturer_variant_code", "processor_cores", "max_clock_hz", "flash_bytes",
        "ram_bytes", "package_types", "pin_counts", "memory_regions_json", "features_json",
        "documents_json", "svd_files", "lifecycle", "source_id", "source_url",
        "source_version", "observed_at", "verification_status",
    ]
    part_fields = [
        "orderable_part_id", "device_id", "manufacturer", "family", "series", "product_line",
        "device_name", "part_number", "manufacturer_suffix", "package_code",
        "temperature_grade_code", "packing_code", "package_name", "temperature_range",
        "packing_form", "lifecycle", "source_id", "source_url", "observed_at",
        "verification_status", "decode_status",
    ]
    source_fields = [
        "source_id", "source_type", "publisher", "title", "url", "version",
        "observed_at", "verification_scope",
    ]
    error_fields = ["scope", "source_url", "item", "error"]
    write_csv(args.output_dir / "device-variants.csv", device_fields, devices)
    write_csv(args.output_dir / "orderable-parts.csv", part_fields, parts)
    write_csv(args.output_dir / "sources.csv", source_fields, source_rows)
    write_csv(args.output_dir / "import-errors.csv", error_fields, errors)

    category_counts = {
        category: {
            "product_lines": len({row["product_line"] for row in devices if row["family"] == config["family"] or (category == "riscv" and row["family"] == "Renesas RISC-V")}),
            "device_variants": len([row for row in devices if row["family"] == config["family"] or (category == "riscv" and row["family"] == "Renesas RISC-V")]),
        }
        for category, config in CATEGORIES.items()
    }
    report = {
        "status": "ok" if not errors else "partial",
        "adapter": "import_renesas_official.py",
        "manufacturer": MANUFACTURER,
        "categories": category_counts,
        "product_lines": len({row["product_line_id"] for row in devices}),
        "device_variants": len(devices),
        "orderable_parts": len(parts),
        "source_records": len(source_rows),
        "import_errors": len(errors),
        "excluded_scope": ["RZ MPU/SoC families", "RZ/Five"],
        "cmsis_ra_supplement_records": len(cmsis_rows),
        "devices_with_group_singleton_inheritance": devices_with_group_inheritance,
        "group_singleton_fields_inherited": inherited_group_fields,
        "cache_sha256": {
            str(path.relative_to(args.cache_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(args.cache_dir.rglob("*")) if path.is_file() and path.suffix.lower() in {".json", ".pdsc"}
        },
        "source_scope": "Renesas official product selector APIs for RA, RX, RL78, RH850, Synergy, and Renesas RISC-V MCUs, supplemented by the official RA CMSIS pack.",
        "completeness_claim": "All exact base parts and orderable codes returned by the selected official MCU selector scopes at observation time. RZ MPUs are excluded; unpublished parts and historical parts omitted by Renesas are not claimed.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "official-adapter-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "cache_sha256"}, ensure_ascii=False, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())

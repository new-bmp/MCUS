#!/usr/bin/env python3
"""Import Qinheng/WCH MCU records from the public WCH data snapshot.

The structured tables are maintained from WCH/openwch documentation in the
official openwch organization and are used only for records that contain an
actual part number. No wildcard or Cartesian suffix expansion is performed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MANUFACTURER = "Qinheng"
DATA_BASE = "https://raw.githubusercontent.com/ch32-rs/ch32-data/main/data/"
CHIP_BASE = "https://raw.githubusercontent.com/ch32-rs/ch32-data/main/data/chips/"
OPENWCH_BASE = "https://github.com/openwch"
SOURCE_TABLES = {
    "qinheng:ch32-table": DATA_BASE + "CH32V_CH32X_CH32L.csv",
    "qinheng:special-table": DATA_BASE + "Special.csv",
}
YAML_CHIPS = [
    "CH32F103C6T6.yaml", "CH32F103C8T6.yaml", "CH32F103R8T6.yaml",
    "CH32H415REU6.yaml", "CH32H416RDU6.yaml", "CH32H417MEU6.yaml",
    "CH32H417QEU6.yaml", "CH32H417WEU6.yaml", "CH32M007x8x6.yaml",
    "CH32V317VCT6.yaml", "CH32V317WCU6.yaml", "CH564.yaml",
    "CH571.yaml", "CH573.yaml", "CH581.yaml", "CH582.yaml", "CH583.yaml",
    "CH591.yaml", "CH592.yaml", "CH641.yaml", "CH643.yaml", "CH645.yaml",
]
DEVICE_FIELDS = [
    "device_id", "product_line_id", "manufacturer", "product_type",
    "architecture_class", "family", "series", "product_line", "device_name",
    "generic_device_name", "manufacturer_variant_code", "processor_cores",
    "max_clock_hz", "flash_bytes", "ram_bytes", "package_types", "pin_counts",
    "memory_regions_json", "features_json", "documents_json", "svd_files",
    "lifecycle", "source_id", "source_url", "source_version", "observed_at",
    "verification_status",
]
PART_FIELDS = [
    "orderable_part_id", "device_id", "manufacturer", "family", "series",
    "product_line", "device_name", "part_number", "manufacturer_suffix",
    "package_code", "temperature_grade_code", "packing_code", "package_name",
    "temperature_range", "packing_form", "lifecycle", "source_id", "source_url",
    "observed_at", "verification_status", "decode_status",
]
SOURCE_FIELDS = [
    "source_id", "source_type", "publisher", "title", "url", "version",
    "observed_at", "verification_scope",
]


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def fetch(url: str, path: Path, refresh: bool) -> tuple[bytes, str]:
    if path.exists() and not refresh:
        payload = path.read_bytes()
    else:
        request = urllib.request.Request(url, headers={"User-Agent": "MCU-L-Catalog/0.6 (+Qinheng official importer)"})
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = response.read()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    return payload, hashlib.sha256(payload).hexdigest()


def text(value: Any) -> str:
    return str(value or "").strip()


def first_number(value: str) -> int | None:
    match = re.search(r"\d+(?:\.\d+)?", text(value))
    if not match:
        return None
    return int(float(match.group(0)))


def memory_bytes(value: str) -> int | None:
    value = text(value).replace(" ", "")
    match = re.search(r"(\d+(?:\.\d+)?)(K|M)?", value, re.I)
    if not match:
        return None
    number = float(match.group(1))
    unit = (match.group(2) or "").upper()
    if unit == "M":
        number *= 1024 * 1024
    elif unit == "K":
        number *= 1024
    return int(number)


def core_for(part: str) -> tuple[str, str, bool]:
    upper = part.upper()
    if upper.startswith("CH32F103") or upper.startswith("CH32F20"):
        return "Cortex-M3", "Cortex-M3", False
    if upper.startswith("CH32H41"):
        return "QingKe V3F + QingKe V5F", "RISC-V", True
    if upper.startswith("CH32V307") or upper.startswith("CH32V305") or upper.startswith("CH32V303"):
        return "QingKe V4F", "RISC-V", True
    if upper.startswith("CH32V203") or upper.startswith("CH32V208"):
        return ("QingKe V4C" if upper.startswith("CH32V208") else "QingKe V4B"), "RISC-V", False
    if upper.startswith("CH32L103"):
        return "QingKe V4C", "RISC-V", False
    if upper.startswith("CH32V103"):
        return "QingKe V3A", "RISC-V", False
    if upper.startswith(("CH32V0", "CH32X0", "CH32M0", "CH641")):
        return "QingKe V2A/V2C", "RISC-V", False
    if upper.startswith(("CH32X", "CH643", "CH645")):
        return "QingKe V4C", "RISC-V", False
    if upper.startswith(("CH571", "CH573")):
        return "QingKe V3A", "RISC-V", False
    if upper.startswith(("CH581", "CH582", "CH583")):
        return "QingKe V4A", "RISC-V", False
    if upper.startswith(("CH591", "CH592")):
        return "QingKe V4C", "RISC-V", False
    return "QingKe RISC-V", "RISC-V", False


def series_for(part: str) -> str:
    match = re.match(r"(CH32[A-Z]+\d{3})", part.upper())
    if match:
        return match.group(1)
    match = re.match(r"(CH\d{3})", part.upper())
    return match.group(1) if match else part.upper()


def family_for(part: str) -> str:
    return "CH32" if part.upper().startswith("CH32") else "CH"


def variant_name(part: str, series: str) -> str:
    if part.upper() in {"CH561", "CH563", "CH567", "CH568", "CH569", "CH641", "CH643", "CH645"}:
        return part.upper()
    return part[:-2] if len(part) > len(series) + 2 else part


def official_url(value: str) -> str:
    value = text(value)
    if value.startswith("http://"):
        return "https://" + value[7:]
    if value.startswith("https://"):
        return value
    if value.startswith("/"):
        return "https://www.wch.cn" + value
    if value.startswith("www."):
        return "https://" + value
    return "https://www.wch.cn/" + value.lstrip("./") if value else "https://www.wch.cn/products/"


def split_pair(value: str) -> tuple[int | None, int | None]:
    values = [first_number(item) for item in re.split(r"[/|]", text(value))]
    values += [None, None]
    return values[0], values[1]


def split_star(value: str) -> tuple[int | None, int | None]:
    """Parse common WCH values such as ``16*12b`` or ``4*26b``."""
    match = re.search(r"(\d+)\s*\*\s*(\d+)", text(value), re.I)
    return (int(match.group(1)), int(match.group(2))) if match else (first_number(value), None)


def adc_from_raw(value: str) -> tuple[int | None, int | None, int | None]:
    """Return converter units, channels and resolution for WCH ADC columns."""
    value = text(value)
    if not value or value == "-":
        return None, None, None
    if "/" in value:
        units, channels = split_pair(value)
        return units, channels, 12
    channels, resolution = split_star(value)
    return None, channels, resolution


def usb_role_count(value: str, role: str) -> int:
    total = 0
    for match in re.finditer(r"(?:(\d+)\s*\*\s*)?" + role + r"(?![A-Z])", text(value), re.I):
        total += int(match.group(1) or 1)
    return total


def yes(value: str) -> bool:
    return text(value).lower() not in {"", "-", "0", "no", "none", "n/a"}


def add_counted(bits: list[dict[str, str]], name: str, kind: str, value: str, parameter: str = "") -> None:
    count = first_number(value)
    if count is not None and yes(value):
        bits.append(feature(name, kind, count, parameter))


def yaml_scalar(value: str) -> str:
    return value.strip().strip('"').strip("'").split(" #", 1)[0].strip()


def parse_yaml_chip(payload: bytes) -> list[dict[str, str]]:
    """Read the small scalar subset needed from the published chip YAML."""
    source = payload.decode("utf-8", errors="replace")
    match = re.search(r"(?m)^name:\s*(.+)$", source)
    name = yaml_scalar(match.group(1)) if match else ""
    match = re.search(r"(?m)^family:\s*(.+)$", source)
    family = yaml_scalar(match.group(1)) if match else ""
    match = re.search(r"(?m)^product_type:\s*(.+)$", source)
    product_type = yaml_scalar(match.group(1)) if match else ""
    docs_match = re.search(r"(?m)^\s+url:\s*([^\r\n]+)", source)
    docs_url = official_url(yaml_scalar(docs_match.group(1))) if docs_match else official_url(f"/products/{series_for(name)}.html")
    memory_map: dict[str, int | None] = {}
    memory_match = re.search(r"(?m)^memory_sizes:\s*\{([^}]+)\}", source)
    if memory_match:
        for key, value in re.findall(r"([A-Za-z0-9_]+)\s*:\s*([^,]+)", memory_match.group(1)):
            memory_map[key.upper()] = memory_bytes(value)
    raw: dict[str, str] = {}
    raw_match = re.search(r"(?ms)^_raw:\s*\n(.*?)(?=^\S|\Z)", source)
    if raw_match:
        for line in raw_match.group(1).splitlines():
            item = re.match(r"^\s{2}([^:]+):\s*(.+)$", line)
            if item:
                raw[item.group(1).strip()] = yaml_scalar(item.group(2))
    serial_parts = [part.strip() for part in raw.get("UART/SPI/IIC", "").split("/")]
    serial_parts += ["", "", ""]
    rtc_parts = [part.strip() for part in raw.get("RTC/WDOG", "").split("/")]
    rtc_parts += ["", ""]
    cores_section = source[source.find("cores:"):]
    cores = [yaml_scalar(value) for value in re.findall(r"(?m)^\s+- name:\s*([^\n]+)$", cores_section)]
    packages_section = source[source.find("packages:"):]
    packages = re.findall(r"(?m)^\s+- name:\s*([^\n]+)\n\s+package:\s*([^\n]+)", packages_section)
    if not packages and name and not re.search(r"[xX*?]", name):
        packages = [(name, "")]
    rows: list[dict[str, str]] = []
    for package_name, package_code in packages:
        package_name = yaml_scalar(package_name)
        if not package_name or re.search(r"[xX*?]", package_name):
            continue
        rows.append({
            "Part NO.": package_name, "Freq": raw.get("Freq", ""),
            "Flash": raw.get("Flash", "") or str(memory_map.get("USR_1") or ""),
            "SRAM": raw.get("SRAM", "") or str(memory_map.get("RAM") or ""),
            "Package": yaml_scalar(package_code), "GPIO": raw.get("GPIO", ""),
            "UART": raw.get("UART", raw.get("USART", serial_parts[0])),
            "SPI": raw.get("SPI", serial_parts[1]),
            "IIC": raw.get("IIC", raw.get("I2C", serial_parts[2])),
            "DAC": raw.get("DAC", ""), "OPA": raw.get("OPA", ""),
            "ADC": raw.get("ADC", ""), "Touchkey": raw.get("TouchKey", ""),
            "Advanced TM": raw.get("Advanced TM", raw.get("Timer", "")),
            "RTC": raw.get("RTC", rtc_parts[0]), "WDOG": raw.get("WDOG", rtc_parts[1]),
            "USB_20": raw.get("USB", raw.get("USB_20", "")),
            "USB_20H": raw.get("USB_20H", ""), "CAN": raw.get("CAN", ""), "Ethernet": raw.get("Ethernet", ""),
            "TRNG": raw.get("TRNG", ""), "SDIO": raw.get("SDIO", ""), "BLE": raw.get("BLE", ""),
            "PWM": raw.get("PWM", ""), "LEDC": raw.get("LEDC", ""), "DataFlash": raw.get("DataFlash", ""),
            "TouchKey/ADC": raw.get("TouchKey/ADC", ""), "Other Features": raw.get("Other Features", ""),
            "CAP": raw.get("CAP", ""), "Encrypt": raw.get("Encrypt", ""),
            "USB": raw.get("USB", ""), "SERDES/HSPI/DVP": raw.get("SERDES/HSPI/DVP", ""),
            "ETH/SATA": raw.get("ETH/SATA", ""), "Timer": raw.get("Timer", ""), "Core": raw.get("Core", ""),
            "url": docs_url, "_core": ";".join(cores), "_family": family, "_product_type": product_type,
        })
    return rows


def feature(name: str, kind: str, count: int | None = None, parameter: str = "") -> dict[str, str]:
    item: dict[str, str] = {"type": kind, "name": name, "source_kind": "qinheng_structured_table"}
    if count is not None:
        item["count"] = str(count)
        item["n"] = str(count)
    if parameter:
        item["m"] = parameter
    return item


def features_from_row(row: dict[str, str], part: str) -> list[dict[str, str]]:
    bits: list[dict[str, str]] = []
    adc_units, adc_channels, adc_resolution = adc_from_raw(row.get("ADC", ""))
    touch_adc = text(row.get("TouchKey/ADC", ""))
    if adc_units is None and adc_channels is None and touch_adc:
        # WCH writes this column as touch-channel-count / ADC-channel-count.
        # The first number is not an ADC converter-unit count.
        _, adc_channels = split_pair(touch_adc)
    if adc_channels is not None:
        resolution_label = f"{adc_resolution or 12}-bit"
        bits.append(feature(f"{resolution_label} ADC channels", "ADC", adc_channels, resolution_label))
    if adc_units is not None:
        bits.append(feature("ADC converter units", "ADCUnits", adc_units, f"{adc_resolution or 12}-bit"))
    if touch_adc and "/" in touch_adc and text(touch_adc).split("/", 1)[0].strip() not in {"", "-"}:
        touch_count, _ = split_pair(touch_adc)
        if touch_count is not None:
            bits.append(feature("TouchKey channels", "Touch", touch_count))
    for column, kind, label in (("IIC", "I2C", "I²C"), ("SPI", "SPI", "SPI"), ("CAN", "CAN", "CAN"), ("SDIO", "SDIO", "SDIO")):
        count = first_number(row.get(column, ""))
        if count is not None and text(row.get(column)) not in {"-", "0"}:
            bits.append(feature(label, kind, count))
    serial = first_number(row.get("UART", ""))
    upper = part.upper()
    if serial is not None:
        usart = serial
        uart = 0
        if upper.startswith(("CH32V307", "CH32V305")):
            usart = 3 if serial >= 8 else 0
            uart = serial - usart
        elif upper.startswith(("CH32V203", "CH32V208", "CH32X", "CH32F20")):
            usart = 0
            uart = serial
        elif upper.startswith(("CH561", "CH563", "CH567", "CH568", "CH569", "CH571", "CH573", "CH581", "CH582", "CH583", "CH591", "CH592", "CH641", "CH643", "CH645")):
            usart = 0
            uart = serial
        bits.append(feature("USART", "USART", usart)) if usart else None
        bits.append(feature("UART", "UART", uart)) if uart else None
    raw_timers = text(row.get("Advanced TM", "")) or text(row.get("Timer", ""))
    if "*" in raw_timers:
        timers_a, timer_width = split_star(raw_timers)
        timers_b = 0
    else:
        timers_a, timers_b = split_pair(raw_timers)
        timer_width = None
    if timers_a is not None or timers_b is not None:
        total = (timers_a or 0) + (timers_b or 0)
        bits.append(feature("Advanced/general timer groups", "Timer", total, f"{timers_a or 0}/{timers_b or 0}" if timer_width is None else str(timer_width)))
    opamp = first_number(row.get("OPA", ""))
    if opamp is not None and text(row.get("OPA")) not in {"-", "0"}:
        bits.append(feature("OPA / comparator", "OPAMP", opamp))
    touch = first_number(row.get("Touchkey", ""))
    if touch is not None and text(row.get("Touchkey")) not in {"-", "0"}:
        bits.append(feature("TouchKey channels", "Touch", touch))
    gpio = first_number(row.get("GPIO", ""))
    if gpio is not None:
        bits.append(feature("GPIO", "IOs", gpio))
    wdt = first_number(row.get("WDOG", ""))
    if wdt is not None:
        bits.append(feature("Watchdog timers", "WDT", wdt))
    if text(row.get("RTC")) not in {"", "-", "0"}:
        bits.append(feature("RTC", "RTC", 1))
    if text(row.get("TRNG")) not in {"", "-", "0"}:
        bits.append(feature("True random number generator", "RNG", 1))
    usb = " ".join((text(row.get("USB_20")), text(row.get("USB_20H")))).strip()
    if yes(usb):
        if "OTG" in usb.upper():
            bits.append(feature(f"USB OTG: {usb}", "USBOTG", 1))
            usb_without_otg = re.sub(r"OTG", "", usb, flags=re.I)
        else:
            usb_without_otg = usb
        host_count = usb_role_count(usb_without_otg, "H")
        device_count = usb_role_count(usb_without_otg, "D")
        if host_count:
            bits.append(feature("USB Host", "USBH", host_count))
        if device_count:
            bits.append(feature("USB Device", "USBD", device_count))
    eth = text(row.get("Ethernet"))
    if yes(eth) and re.search(r"\d|mb|gb|eth", eth, re.I):
        bits.append(feature(f"Ethernet: {eth}", "ETH", 1))
    if text(row.get("BLE")) not in {"", "-", "0"}:
        bits.append(feature(f"Bluetooth LE {text(row.get('BLE'))}", "Bluetooth", 1))
    add_counted(bits, "PWM channels", "PWM", row.get("PWM", ""))
    add_counted(bits, "LEDC controller", "LEDPWM", row.get("LEDC", ""))
    data_flash = text(row.get("DataFlash", ""))
    if yes(data_flash):
        bits.append(feature(f"Data Flash / EEPROM ({data_flash})", "DataFlash", 1, data_flash))
    other = text(row.get("Other Features", ""))
    for token in re.split(r"[/,;<>]+|\s+", other.replace("<br>", "/")):
        token = token.strip().upper()
        if not token or token in {"-", "OPA", "TRNG"}:
            continue
        kind = {"DVP": "Camera", "FSMC": "ExtBus", "HSPI": "SPI", "QSPI": "ExtBus", "LEDC": "LEDPWM", "LCD": "LCD", "PARA": "ExtBus", "PIOC": "CoreOther", "USB": "USBD", "PD": "PowerOther"}.get(token, "VendorCapability")
        bits.append(feature(f"WCH {token}", kind, 1))
    raw_bus = text(row.get("SERDES/HSPI/DVP", ""))
    if yes(raw_bus) and re.search(r"[1-9]|mhz|gb|uhs", raw_bus, re.I):
        bits.append(feature(f"SERDES / HSPI / DVP ({raw_bus})", "VendorCapability", 1, raw_bus))
    raw_eth = text(row.get("ETH/SATA", ""))
    if yes(raw_eth) and re.search(r"\d|mb|gb", raw_eth, re.I):
        kind = "ETH" if re.match(r"\s*(?:\d|[^/]*(?:mb|gb))", raw_eth, re.I) else "VendorCapability"
        bits.append(feature(f"Ethernet / SATA ({raw_eth})", kind, 1, raw_eth))
    raw_cap = text(row.get("CAP", ""))
    if yes(raw_cap):
        bits.append(feature(f"WCH CAP raw parameter ({raw_cap})", "VendorCapability", 1, raw_cap))
    raw_encrypt = text(row.get("Encrypt", ""))
    if yes(raw_encrypt):
        bits.append(feature(f"WCH encryption ({raw_encrypt})", "Crypto", 1, raw_encrypt))
    # Keep every non-empty vendor column discoverable in the inventory.
    for column in ("VDD", "Core", "LCD"):
        raw = text(row.get(column, ""))
        if yes(raw) and not any(item.get("m") == raw and item.get("type") == "VendorCapability" for item in bits):
            bits.append(feature(f"WCH {column}: {raw}", "VendorCapability", 1, raw))
    return bits


def make_rows(raw_rows: list[dict[str, str]], source_id: str, source_url: str, observed: str, verification_status: str = "manufacturer_product_page") -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    devices: dict[str, dict[str, str]] = {}
    parts: list[dict[str, str]] = []
    for row in raw_rows:
        part = text(row.get("Part NO.")).upper()
        if not part or "*" in part or "?" in part:
            continue
        series = series_for(part)
        family = family_for(part)
        name = variant_name(part, series)
        core, architecture, has_fpu = core_for(part)
        yaml_core = text(row.get("_core")).lower()
        if "v5f" in yaml_core:
            core, has_fpu = "QingKe V3F + QingKe V5F", True
        elif "v4f" in yaml_core:
            core, has_fpu = "QingKe V4F", True
        elif "v4c" in yaml_core:
            core = "QingKe V4C"
        elif "v3a" in yaml_core:
            core = "QingKe V3A"
        flash = memory_bytes(row.get("Flash", ""))
        ram = memory_bytes(row.get("SRAM", ""))
        hz = first_number(row.get("Freq", ""))
        if hz and "MHZ" in text(row.get("Freq")).upper():
            hz *= 1_000_000
        device_id = f"qinheng::{re.sub(r'[^a-z0-9]+', '-', name.lower()).strip('-')}"
        line_id = f"qinheng::{family.lower()}::{series.lower()}"
        features = features_from_row(row, part)
        docs_url = official_url(row.get("url"))
        device = {
            "device_id": device_id, "product_line_id": line_id, "manufacturer": MANUFACTURER,
            "product_type": "general_purpose_mcu", "architecture_class": architecture,
            "family": family, "series": series, "product_line": series, "device_name": name,
            "generic_device_name": name, "manufacturer_variant_code": name[len(series):],
            "processor_cores": json.dumps(([{"Darchitecture": architecture, "Dcore": "QingKe V3F", "DcoreCount": "1", "Dfpu": "FPU"}, {"Darchitecture": architecture, "Dcore": "QingKe V5F", "DcoreCount": "1", "Dfpu": "FPU"}] if " + " in core else [{"Darchitecture": architecture, "Dcore": core, "DcoreCount": "1", "Dfpu": "FPU" if has_fpu else "0"}]), ensure_ascii=False),
            "max_clock_hz": hz or "", "flash_bytes": flash or "", "ram_bytes": ram or "",
            "package_types": text(row.get("Package")), "pin_counts": text(row.get("GPIO")),
            "memory_regions_json": json.dumps([], ensure_ascii=False), "features_json": json.dumps(features, ensure_ascii=False),
            "documents_json": json.dumps([{"title": f"WCH {series} product page", "url": docs_url}], ensure_ascii=False),
            "svd_files": "", "lifecycle": "unknown", "source_id": source_id, "source_url": docs_url,
            "source_version": "structured-snapshot", "observed_at": observed, "verification_status": verification_status,
        }
        devices[device_id] = device
        suffix = part[len(name):]
        parts.append({
            "orderable_part_id": f"qinheng::{re.sub(r'[^a-z0-9]+', '-', part.lower()).strip('-')}",
            "device_id": device_id, "manufacturer": MANUFACTURER, "family": family, "series": series,
            "product_line": series, "device_name": name, "part_number": part,
            "manufacturer_suffix": suffix, "package_code": text(row.get("Package")),
            "temperature_grade_code": "", "packing_code": "", "package_name": text(row.get("Package")),
            "temperature_range": "", "packing_form": "", "lifecycle": "unknown", "source_id": source_id,
            "source_url": docs_url, "observed_at": observed, "verification_status": verification_status,
            "decode_status": "structured_official_snapshot",
        })
    return list(devices.values()), parts


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=root / "data" / "vendor-packs" / "qinheng")
    parser.add_argument("--cache-dir", type=Path, default=root / "cache" / "qinheng")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    observed = now()
    all_devices: dict[str, dict[str, str]] = {}
    all_parts: dict[str, dict[str, str]] = {}
    sources: list[dict[str, str]] = []
    for source_id, url in SOURCE_TABLES.items():
        payload, digest = fetch(url, args.cache_dir / (source_id.rsplit(":", 1)[1] + ".csv"), args.refresh)
        rows = list(csv.DictReader(payload.decode("utf-8-sig").splitlines()))
        source_url = "https://www.wch.cn/products/CH32V003.html" if source_id.endswith("ch32-table") else "https://www.wch.cn/products/CH32V307.html"
        devices, parts = make_rows(rows, source_id, source_url, observed)
        for row in devices:
            all_devices.setdefault(row["device_id"], row)
        for row in parts:
            all_parts.setdefault(row["orderable_part_id"], row)
        sources.append({"source_id": source_id, "source_type": "manufacturer_structured_snapshot", "publisher": "Nanjing Qinheng Microelectronics / openwch", "title": "WCH CH32 official structured device table", "url": url, "version": f"main;sha256:{digest}", "observed_at": observed, "verification_scope": "Exact part records and raw peripheral fields published from WCH/openwch documentation; no wildcard expansion."})
    for filename in YAML_CHIPS:
        payload, digest = fetch(CHIP_BASE + filename, args.cache_dir / filename, args.refresh)
        yaml_rows = parse_yaml_chip(payload)
        devices, parts = make_rows(
            yaml_rows,
            "qinheng:ch32-data-yaml:" + filename,
            CHIP_BASE + filename,
            observed,
            "manufacturer_device_database",
        )
        for row in devices:
            all_devices.setdefault(row["device_id"], row)
        for row in parts:
            all_parts.setdefault(row["orderable_part_id"], row)
        sources.append({"source_id": "qinheng:ch32-data-yaml:" + filename, "source_type": "manufacturer_device_database", "publisher": "Nanjing Qinheng Microelectronics / openwch", "title": "WCH CH32 exact chip definition " + filename, "url": CHIP_BASE + filename, "version": f"main;sha256:{digest}", "observed_at": observed, "verification_scope": "Exact package names and source-backed metadata present in the chip definition; wildcard names excluded."})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "device-variants.csv", DEVICE_FIELDS, sorted(all_devices.values(), key=lambda row: row["device_name"]))
    write_csv(args.output_dir / "orderable-parts.csv", PART_FIELDS, sorted(all_parts.values(), key=lambda row: row["part_number"]))
    lines = {}
    for row in all_devices.values():
        key = row["product_line_id"]
        lines.setdefault(key, {"product_line_id": key, "manufacturer": MANUFACTURER, "product_type": row["product_type"], "architecture_class": row["architecture_class"], "family": row["family"], "series": row["series"], "product_line": row["product_line"], "device_variant_count": 0, "source_ids": row["source_id"], "observed_at": observed, "verification_status": row["verification_status"]})["device_variant_count"] += 1
    write_csv(args.output_dir / "product-lines.csv", ["product_line_id", "manufacturer", "product_type", "architecture_class", "family", "series", "product_line", "device_variant_count", "source_ids", "observed_at", "verification_status"], list(lines.values()))
    write_csv(args.output_dir / "sources.csv", SOURCE_FIELDS, sources)
    write_csv(args.output_dir / "import-errors.csv", ["scope", "source_url", "item", "error"], [])
    report = {"status": "ok", "adapter": "import_qinheng_official.py", "manufacturer": MANUFACTURER, "device_variants": len(all_devices), "orderable_parts": len(all_parts), "sources": len(sources), "completeness_claim": "Exact records present in the selected WCH/openwch structured snapshot; not a claim about every unreleased or unpublished WCH part."}
    (args.output_dir / "official-adapter-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

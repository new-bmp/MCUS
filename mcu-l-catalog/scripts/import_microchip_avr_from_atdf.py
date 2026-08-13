#!/usr/bin/env python3
"""Import exact AVR devices and variants from official Microchip ATDF packs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


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
MANIFEST_FIELDS = [
    "source_id", "pack_url", "cache_path", "sha256", "bytes", "atdf_files",
    "device_records", "orderable_parts",
]

PACKS = (
    ("ATmega_DFP", "3.6.299", "Microchip.ATmega_DFP.3.6.299.atpack"),
    ("ATtiny_DFP", "3.4.278", "Microchip.ATtiny_DFP.3.4.278.atpack"),
    ("AVR-Dx_DFP", "2.8.343", "Microchip.AVR-Dx_DFP.2.8.343.atpack"),
    ("AVR-Ex_DFP", "2.12.239", "Microchip.AVR-Ex_DFP.2.12.239.atpack"),
    ("AVR-Lx_DFP", "1.3.46", "Microchip.AVR-Lx_DFP.1.3.46.atpack"),
    ("AVR-Sx_DFP", "1.2.26", "Microchip.AVR-Sx_DFP.1.2.26.atpack"),
    ("ATautomotive_DFP", "3.1.73", "Microchip.ATautomotive_DFP.3.1.73.atpack"),
)
BASE_URL = "https://packs.download.microchip.com/"
SOURCE_PREFIX = "microchip-avr-atdf:"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def parse_int(value: str) -> int | None:
    try:
        return int(value, 0)
    except (TypeError, ValueError):
        return None


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fields: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def download(url: str, path: Path, timeout: float, refresh: bool) -> bytes:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != "packs.download.microchip.com":
        raise ValueError(f"unapproved AVR pack URL: {url}")
    if refresh or not path.exists():
        request = urllib.request.Request(
            url, headers={"User-Agent": "MCU-L-Catalog/0.5 (+official AVR ATDF importer)"}
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    payload = path.read_bytes()
    with zipfile.ZipFile(path) as archive:
        archive.testzip()
    return payload


def module_type(module_name: str) -> str | None:
    module = module_name.upper().replace("-", "_")
    if module in {"TC", "TC8", "TC8_ASYNC", "TC16", "TCA", "TCB", "TCD", "TCE", "TIMER"}:
        return "Timer"
    if module == "ADC":
        return "ADCUnits"
    if module == "DAC":
        return "DAC"
    if module in {"AC", "ACC", "CMP", "COMP"}:
        return "COMP"
    if module == "OPAMP":
        return "OPAMP"
    if module == "SPI":
        return "SPI"
    if module in {"TWI", "I2C"}:
        return "I2C"
    if module == "USART":
        return "USART"
    if module == "UART":
        return "UART"
    if module in {"CAN", "CANCTRL"}:
        return "CAN"
    if module in {"USB", "USB_DEVICE"}:
        return "USBD"
    if module in {"WDT", "WDTCSR"}:
        return "WDT"
    if module == "RTC":
        return "RTC"
    if module in {"PORT", "VPORT", "GPIO"}:
        return None
    return module_name or None


def product_line(device_name: str, atdf_family: str, architecture: str) -> str:
    upper = device_name.upper()
    family_upper = atdf_family.upper()
    if upper.startswith("AT90"):
        return "AT90 specialty AVR"
    if upper.startswith("ATMEGA"):
        if architecture.upper() == "AVR8X":
            return "megaAVR 0-series"
        return "Classic ATmega"
    if upper.startswith("ATTINY"):
        if "TINYAVR 2" in family_upper:
            return "tinyAVR 2-series"
        if architecture.upper() == "AVR8X":
            return "tinyAVR 0/1-series"
        return "Classic ATtiny"
    for code, label in (
        ("DA", "AVR DA"), ("DB", "AVR DB"), ("DD", "AVR DD"),
        ("DU", "AVR DU"), ("EA", "AVR EA"), ("EB", "AVR EB"),
        ("LA", "AVR LA"), ("SD", "AVR SD"),
    ):
        if re.match(rf"AVR\d+{code}\d+", upper):
            return label
    if upper.startswith("ATA"):
        return "Automotive AVR"
    return "Other AVR"


def variant_code(device_name: str, line: str) -> str:
    if line in {"Classic ATmega", "megaAVR 0-series", "AT90 specialty AVR"}:
        return re.sub(r"^(?:ATmega|AT90)", "", device_name, flags=re.I)
    if "tinyAVR" in line or line == "Classic ATtiny":
        return re.sub(r"^ATtiny", "", device_name, flags=re.I)
    if device_name.upper().startswith("AVR"):
        return device_name[3:]
    return device_name


def feature(type_name: str, name: str, count: int) -> dict[str, str]:
    return {
        "type": type_name, "name": name, "count": str(count),
        "source_kind": "microchip_atdf",
    }


def parse_atdf(payload: bytes, *, source: dict[str, str], archive_path: str, observed_at: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root = ET.fromstring(payload)
    device = root.find("./devices/device")
    if device is None or not device.get("name"):
        raise ValueError("ATDF has no devices/device name")
    name = device.get("name", "")
    architecture = device.get("architecture", "AVR8")
    variants = list(root.findall("./variants/variant"))
    pinouts = {item.get("name", ""): item for item in root.findall("./pinouts/pinout")}
    selected_pinouts = {item.get("pinout", "") for item in variants if item.get("pinout")}
    package_pads: set[str] = set()
    pin_counts: set[int] = set()
    for pinout_name in selected_pinouts:
        pinout = pinouts.get(pinout_name)
        if pinout is None:
            continue
        pins = list(pinout)
        pin_counts.add(len(pins))
        package_pads.update(pin.get("pad", "") for pin in pins if pin.get("pad"))
    gpio_pads = {pad for pad in package_pads if re.fullmatch(r"P[A-Z]+\d+", pad, re.I)}

    grouped: dict[str, set[str]] = defaultdict(set)
    adc_channels: set[str] = set()
    adc_pads: set[str] = set()
    vendor_features: dict[str, str] = {
        "CCL": "CCL configurable custom logic",
        "EVSYS": "Event System",
        "MVIO": "Multi-Voltage I/O (MVIO)",
        "ZCD": "Zero-Cross Detector (ZCD)",
        "CRCSCAN": "CRC memory scanner",
        "PTC": "Peripheral Touch Controller (PTC)",
    }
    peripherals = device.find("peripherals")
    if peripherals is not None:
        for module in peripherals:
            module_name = module.get("name", "")
            normalized = module_type(module_name)
            instances = list(module)
            if normalized:
                grouped[normalized].update(
                    item.get("name", "") or module_name for item in instances
                )
            if module_name.upper() in vendor_features and instances:
                grouped["VendorCapability"].add(vendor_features[module_name.upper()])
            if module_name.upper() != "ADC":
                continue
            for instance in instances:
                for signal in instance.findall("./signals/signal"):
                    group = signal.get("group", "").upper()
                    function = signal.get("function", "").upper()
                    index = signal.get("index", "")
                    pad = signal.get("pad", "")
                    if not (group.startswith(("ADC", "AIN")) or function.startswith(("ADC", "AIN"))):
                        continue
                    # Newer AVR ATDFs may repeat function="AIN0" and carry
                    # the real channel number in index. Older files use the
                    # function name itself. Prefer index whenever available.
                    channel = index if index != "" else function
                    if channel:
                        adc_channels.add(channel)
                    if pad in package_pads:
                        adc_pads.add(pad)

    features: list[dict[str, str]] = []
    exact_types = (
        "Timer", "ADCUnits", "DAC", "SPI", "I2C", "USART", "UART", "CAN",
        "USBD", "WDT", "RTC", "COMP", "OPAMP",
    )
    for type_name in exact_types:
        instances = sorted(grouped.pop(type_name, set()))
        features.append(feature(type_name, ", ".join(instances) or "ATDF exact instance count", len(instances)))
    features.extend((
        feature("IOs", "GPIO pads present in official package pinouts", len(gpio_pads)),
        feature("ADC", "ADC external signal channels", len(adc_channels)),
        feature("ADCExternalPins", "External package pads with ADC input signals", len(adc_pads)),
    ))
    for type_name, instances in sorted(grouped.items()):
        for instance in sorted(instances):
            features.append(feature(type_name, instance, 1))

    flash_candidates: list[tuple[int, dict[str, str]]] = []
    ram_total = 0
    regions: list[dict[str, str]] = []
    for segment in device.findall("./address-spaces/address-space/memory-segment"):
        kind = segment.get("type", "").lower()
        size = parse_int(segment.get("size", ""))
        if size is None:
            continue
        region = {key: value for key, value in {
            "name": segment.get("name", ""), "start": segment.get("start", ""),
            "size": segment.get("size", ""), "type": kind,
            "external": segment.get("external", ""),
        }.items() if value}
        if kind in {"flash", "ram", "eeprom"}:
            regions.append(region)
        if kind == "flash" and not any(token in segment.get("name", "").upper() for token in ("BOOT_SECTION", "FUSE", "USER", "SIGNATURE")):
            flash_candidates.append((size, region))
        if kind == "ram" and segment.get("external", "").lower() != "true":
            ram_total += size
    flash_bytes = max((size for size, _ in flash_candidates), default=0)
    speeds = [parse_int(item.get("speedmax", "")) for item in variants]
    speeds = [speed for speed in speeds if speed is not None]
    packages = sorted({item.get("package", "") for item in variants if item.get("package")})
    line = product_line(name, device.get("family", ""), architecture)
    device_id = f"microchip::{slug(name)}"
    source_url = source["url"]
    processors = [{
        "Dcore": architecture,
        "Dclock": str(max(speeds)) if speeds else "",
        "Dfpu": "NO_FPU",
        "Dmpu": "NO_MPU",
        "Ddsp": "NO_DSP",
        "Dtz": "NO_TZ",
        "source_kind": "microchip_atdf_architecture",
    }]
    device_row = {
        "device_id": device_id,
        "product_line_id": f"microchip::avr::avr-8-bit::{slug(line)}",
        "manufacturer": "Microchip",
        "product_type": "general_purpose_mcu",
        "architecture_class": "8-bit AVR",
        "family": "AVR",
        "series": "AVR 8-bit",
        "product_line": line,
        "device_name": name,
        "generic_device_name": name,
        "manufacturer_variant_code": variant_code(name, line),
        "processor_cores": json.dumps(processors, ensure_ascii=False, sort_keys=True),
        "max_clock_hz": str(max(speeds)) if speeds else "",
        "flash_bytes": str(flash_bytes) if flash_bytes else "",
        "ram_bytes": str(ram_total) if ram_total else "",
        "package_types": ";".join(packages),
        "pin_counts": ";".join(str(value) for value in sorted(pin_counts)),
        "memory_regions_json": json.dumps(regions, ensure_ascii=False, sort_keys=True),
        "features_json": json.dumps(features, ensure_ascii=False, sort_keys=True),
        "documents_json": json.dumps([{
            "title": f"Official Microchip product page for {name}",
            "name": f"https://www.microchip.com/en-us/product/{name}",
        }], ensure_ascii=False, sort_keys=True),
        "svd_files": "",
        "lifecycle": "unknown",
        "source_id": source["source_id"],
        "source_url": source_url,
        "source_version": source["version"],
        "observed_at": observed_at,
        "verification_status": "manufacturer_device_database",
    }

    part_rows: list[dict[str, Any]] = []
    seen_parts: set[str] = set()
    for item in variants:
        part_number = item.get("ordercode", "").strip()
        # Slash-separated package labels are family selectors, not purchasable SKUs.
        if not part_number.upper().startswith(name.upper()) or "/" in part_number or part_number.lower() == "standard":
            continue
        if part_number.upper() in seen_parts:
            continue
        seen_parts.add(part_number.upper())
        temp_min, temp_max = item.get("tempmin", ""), item.get("tempmax", "")
        part_rows.append({
            "orderable_part_id": f"microchip::{slug(part_number)}",
            "device_id": device_id,
            "manufacturer": "Microchip", "family": "AVR", "series": "AVR 8-bit",
            "product_line": line, "device_name": name, "part_number": part_number,
            "manufacturer_suffix": part_number[len(name):], "package_code": "",
            "temperature_grade_code": "", "packing_code": "",
            "package_name": item.get("package", ""),
            "temperature_range": f"{temp_min}..{temp_max} °C" if temp_min and temp_max else "",
            "packing_form": "", "lifecycle": "unknown",
            "source_id": source["source_id"], "source_url": source_url,
            "observed_at": observed_at,
            "verification_status": "manufacturer_device_database",
            "decode_status": "atdf_variant_record",
        })
    return device_row, part_rows


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--vendor-pack", type=Path, default=root / "data/vendor-packs/microchip")
    parser.add_argument("--cache-dir", type=Path, default=root / "cache/microchip/packs")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()
    observed_at = utc_now()
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    prior_devices = [
        row for row in read_csv(args.vendor_pack / "device-variants.csv")
        if SOURCE_PREFIX not in row.get("source_id", "")
    ]
    prior_parts = [
        row for row in read_csv(args.vendor_pack / "orderable-parts.csv")
        if SOURCE_PREFIX not in row.get("source_id", "")
    ]
    prior_sources = [
        row for row in read_csv(args.vendor_pack / "sources.csv")
        if not row.get("source_id", "").startswith(SOURCE_PREFIX)
    ]
    devices: list[dict[str, Any]] = []
    parts: list[dict[str, Any]] = []
    sources: list[dict[str, str]] = []
    manifests: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    line_counts: Counter[str] = Counter()

    for pack_name, version, filename in PACKS:
        url = BASE_URL + filename
        path = args.cache_dir / filename
        try:
            payload = download(url, path, args.timeout, args.refresh)
            source = {
                "source_id": f"{SOURCE_PREFIX}{pack_name}@{version}",
                "source_type": "manufacturer_atdf_device_database",
                "publisher": "Microchip",
                "title": f"Microchip {pack_name} exact-device ATDF files",
                "url": url, "version": version, "observed_at": observed_at,
                "verification_scope": (
                    "Exact AVR device names, architecture, memory segments, peripheral instances, "
                    "package pinouts, ADC signal pads, and ATDF manufacturer order codes."
                ),
            }
            sources.append(source)
            pack_devices = 0
            pack_parts = 0
            with zipfile.ZipFile(path) as archive:
                names = sorted(name for name in archive.namelist() if name.lower().endswith(".atdf"))
                for archive_path in names:
                    try:
                        device, device_parts = parse_atdf(
                            archive.read(archive_path), source=source,
                            archive_path=archive_path, observed_at=observed_at,
                        )
                        devices.append(device)
                        parts.extend(device_parts)
                        line_counts[device["product_line"]] += 1
                        pack_devices += 1
                        pack_parts += len(device_parts)
                    except Exception as exc:
                        errors.append({"pack": filename, "path": archive_path, "error": str(exc)})
            manifests.append({
                "source_id": source["source_id"], "pack_url": url,
                "cache_path": str(path.resolve()), "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload), "atdf_files": pack_devices,
                "device_records": pack_devices, "orderable_parts": pack_parts,
            })
        except Exception as exc:
            errors.append({"pack": filename, "path": "", "error": str(exc)})

    duplicate_devices = [name for name, count in Counter(row["device_id"] for row in devices).items() if count > 1]
    if duplicate_devices:
        errors.extend({"pack": "", "path": device_id, "error": "duplicate AVR device across latest packs"} for device_id in duplicate_devices)
    unique_devices = {row["device_id"]: row for row in devices}
    unique_parts = {row["orderable_part_id"]: row for row in parts}
    all_devices = sorted([*prior_devices, *unique_devices.values()], key=lambda row: (row["family"], row["series"], row["product_line"], row["device_name"]))
    all_parts = sorted([*prior_parts, *unique_parts.values()], key=lambda row: row["part_number"])
    all_sources = sorted([*prior_sources, *sources], key=lambda row: row["source_id"])
    write_csv(args.vendor_pack / "device-variants.csv", DEVICE_FIELDS, all_devices)
    write_csv(args.vendor_pack / "orderable-parts.csv", PART_FIELDS, all_parts)
    write_csv(args.vendor_pack / "sources.csv", SOURCE_FIELDS, all_sources)
    write_csv(args.vendor_pack / "avr-atdf-source-manifest.csv", MANIFEST_FIELDS, manifests)
    report = {
        "status": "ok" if not errors else "partial",
        "official_packs": len(PACKS), "device_records": len(unique_devices),
        "orderable_parts": len(unique_parts), "product_lines": dict(sorted(line_counts.items())),
        "parse_errors": errors, "observed_at": observed_at,
        "accuracy_policy": [
            "Every MCU row comes from one exact official Microchip ATDF file.",
            "Overlapping AVR boot regions are excluded from total Flash; the primary program region is used.",
            "ADC converter units, signal channels, and package-exposed ADC pads are stored separately.",
            "Slash-separated ATDF package selectors are not presented as purchasable order codes.",
            "AVR8/AVR8X architecture establishes that no hardware FPU is present.",
        ],
    }
    (args.vendor_pack / "avr-atdf-import-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())

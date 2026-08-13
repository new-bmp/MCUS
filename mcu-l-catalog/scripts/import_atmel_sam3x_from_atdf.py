#!/usr/bin/env python3
"""Import legacy Atmel SAM3X devices from the official exact-device ATDF pack."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from augment_microchip_from_atdf import parse_atdf, variant_ordercode


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

VERSION = "1.0.50"
FILENAME = f"Atmel.SAM3X_DFP.{VERSION}.atpack"
PACK_URL = f"http://packs.download.atmel.com/{FILENAME}"
PDSC_URL = "https://www.keil.com/pack/Atmel.SAM3X_DFP.pdsc"
EXPECTED_SHA256 = "17d719adc9af102faf55a5b625b55151c663ac4a026dde80ab7db4a1a3286168"
SOURCE_ID = f"atmel-atdf:SAM3X_DFP@{VERSION}"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def slug(value: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


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


def ensure_pack(path: Path, refresh: bool, timeout: float) -> bytes:
    if refresh or not path.exists():
        request = urllib.request.Request(
            PACK_URL, headers={"User-Agent": "MCU-L-Catalog/0.5 (+official SAM3X importer)"}
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError(f"SAM3X pack SHA-256 mismatch: {digest}")
    with zipfile.ZipFile(path) as archive:
        bad = archive.testzip()
        if bad:
            raise ValueError(f"corrupt SAM3X pack member: {bad}")
    return payload


def exact_sam3x_metadata(payload: bytes, parsed: dict[str, Any]) -> tuple[list[dict[str, str]], str, str]:
    root = ET.fromstring(payload)
    device = root.find("./devices/device")
    if device is None:
        raise ValueError("SAM3X ATDF has no device")
    adc_channels: set[str] = set()
    adc_pads: set[str] = set()
    timer_channels: set[str] = set()
    features = [
        item for item in parsed["features"]
        if not (
            item.get("type") in {"Timer", "IOs", "ADCExternalPins", "DAC", "ComOther"}
            or item.get("type") == "DACC"
        )
    ]
    peripherals = device.find("peripherals")
    if peripherals is not None:
        for module in peripherals:
            module_name = module.get("name", "")
            for instance in module:
                for signal in instance.findall("./signals/signal"):
                    group = signal.get("group", "").upper()
                    index = signal.get("index", "")
                    pad = signal.get("pad", "")
                    if module_name == "ADC" and group == "AD":
                        if index:
                            adc_channels.add(index)
                        if pad:
                            adc_pads.add(pad)
                    if module_name == "TC" and index:
                        timer_channels.add(index)
    features.extend((
        {"type": "Timer", "name": "TC channel indices", "count": str(len(timer_channels)), "source_kind": "microchip_atdf"},
        {"type": "PWM", "name": "PWM controller", "count": "1", "source_kind": "microchip_atdf"},
        {"type": "ADC", "name": "ADC external signal channels", "count": str(len(adc_channels)), "source_kind": "microchip_atdf"},
        {"type": "ADCExternalPins", "name": "External pads with ADC input signals", "count": str(len(adc_pads)), "source_kind": "microchip_atdf"},
        {"type": "DAC", "name": "DACC controller", "count": "1", "source_kind": "microchip_atdf"},
        {"type": "USBOTG", "name": "UOTGHS USB OTG controller", "count": "1", "source_kind": "microchip_atdf"},
    ))
    variants = list(root.findall("./variants/variant"))
    packages = sorted({item.get("package", "") for item in variants if item.get("package")})
    pin_counts = sorted({
        int(match.group(1)) for package in packages
        if (match := re.search(r"(\d+)$", package))
    })
    return features, ";".join(packages), ";".join(str(value) for value in pin_counts)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--vendor-pack", type=Path, default=root / "data/vendor-packs/microchip")
    parser.add_argument("--cache-dir", type=Path, default=root / "cache/microchip/packs")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()
    observed_at = utc_now()
    pack_path = args.cache_dir / FILENAME
    payload = ensure_pack(pack_path, args.refresh, args.timeout)

    source = {
        "source_id": SOURCE_ID, "source_type": "manufacturer_atdf_device_database",
        "publisher": "Atmel (now Microchip)", "title": "Atmel SAM3X exact-device ATDF files",
        "url": PDSC_URL, "version": VERSION, "observed_at": observed_at,
        "verification_scope": (
            "Exact SAM3X device names, Cortex-M3 processor metadata, memory banks, peripheral "
            "instances, package pinouts, ADC signal pads, and manufacturer order codes."
        ),
    }
    new_devices: list[dict[str, Any]] = []
    new_parts: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    with zipfile.ZipFile(pack_path) as archive:
        atdf_names = sorted(name for name in archive.namelist() if name.lower().endswith(".atdf"))
        for archive_path in atdf_names:
            try:
                atdf_payload = archive.read(archive_path)
                record = parse_atdf(atdf_payload, archive_path)
                exact_features, package_types, pin_counts = exact_sam3x_metadata(atdf_payload, record)
                name = record["device_name"]
                line = name[2:-1]  # ATSAM3X4C -> SAM3X4; ATSAM3X8E -> SAM3X8
                device_id = f"microchip::{slug(name)}"
                processors = [{
                    "Dcore": "Cortex-M3", "Dclock": record["max_clock_hz"],
                    "Dfpu": "NO_FPU", "Dmpu": "MPU", "Ddsp": "NO_DSP",
                    "Dtz": "NO_TZ", "source_kind": "atmel_sam3x_pdsc_and_atdf",
                }]
                new_devices.append({
                    "device_id": device_id,
                    "product_line_id": f"microchip::sam3::sam3x::{slug(line)}",
                    "manufacturer": "Microchip", "product_type": "general_purpose_mcu",
                    "architecture_class": "Cortex-M3", "family": "SAM3",
                    "series": "SAM3X", "product_line": line, "device_name": name,
                    "generic_device_name": name,
                    "manufacturer_variant_code": name[len(line):],
                    "processor_cores": json.dumps(processors, ensure_ascii=False, sort_keys=True),
                    "max_clock_hz": record["max_clock_hz"],
                    "flash_bytes": record["flash_bytes"], "ram_bytes": record["ram_bytes"],
                    "package_types": package_types, "pin_counts": pin_counts,
                    "memory_regions_json": record["memory_regions_json"],
                    "features_json": json.dumps(exact_features, ensure_ascii=False, sort_keys=True),
                    "documents_json": json.dumps([{
                        "title": f"Official Atmel/Microchip SAM3X pack for {name}", "name": PDSC_URL,
                    }], ensure_ascii=False, sort_keys=True),
                    "svd_files": f"svd/{name}.svd", "lifecycle": "unknown",
                    "source_id": SOURCE_ID, "source_url": PDSC_URL,
                    "source_version": VERSION, "observed_at": observed_at,
                    "verification_status": "manufacturer_device_database",
                })
                for variant in record["variants"]:
                    part_number = variant_ordercode(variant, name)
                    if not part_number or not part_number.upper().startswith(name.upper()):
                        continue
                    temp_min, temp_max = variant.get("tempmin", ""), variant.get("tempmax", "")
                    new_parts.append({
                        "orderable_part_id": f"microchip::{slug(part_number)}", "device_id": device_id,
                        "manufacturer": "Microchip", "family": "SAM3", "series": "SAM3X",
                        "product_line": line, "device_name": name, "part_number": part_number,
                        "manufacturer_suffix": part_number[len(name):], "package_code": "",
                        "temperature_grade_code": "", "packing_code": "",
                        "package_name": variant.get("package", ""),
                        "temperature_range": f"{temp_min}..{temp_max} °C" if temp_min and temp_max else "",
                        "packing_form": "", "lifecycle": "unknown", "source_id": SOURCE_ID,
                        "source_url": PDSC_URL, "observed_at": observed_at,
                        "verification_status": "manufacturer_device_database",
                        "decode_status": "atdf_variant_record",
                    })
            except Exception as exc:
                errors.append({"path": archive_path, "error": str(exc)})

    old_devices = [
        row for row in read_csv(args.vendor_pack / "device-variants.csv")
        if SOURCE_ID not in row.get("source_id", "")
    ]
    old_parts = [
        row for row in read_csv(args.vendor_pack / "orderable-parts.csv")
        if SOURCE_ID not in row.get("source_id", "")
    ]
    old_sources = [
        row for row in read_csv(args.vendor_pack / "sources.csv")
        if row.get("source_id") != SOURCE_ID
    ]
    write_csv(args.vendor_pack / "device-variants.csv", DEVICE_FIELDS, sorted(
        [*old_devices, *new_devices], key=lambda row: (row["family"], row["series"], row["product_line"], row["device_name"])
    ))
    write_csv(args.vendor_pack / "orderable-parts.csv", PART_FIELDS, sorted(
        [*old_parts, *new_parts], key=lambda row: row["part_number"]
    ))
    write_csv(args.vendor_pack / "sources.csv", SOURCE_FIELDS, sorted(
        [*old_sources, source], key=lambda row: row["source_id"]
    ))
    report = {
        "status": "ok" if not errors else "partial", "devices": len(new_devices),
        "orderable_parts": len(new_parts), "atdf_files": len(new_devices),
        "pack_url": PACK_URL, "pdsc_url": PDSC_URL,
        "pack_sha256": hashlib.sha256(payload).hexdigest(), "pack_bytes": len(payload),
        "errors": errors, "observed_at": observed_at,
    }
    (args.vendor_pack / "sam3x-atdf-import-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())

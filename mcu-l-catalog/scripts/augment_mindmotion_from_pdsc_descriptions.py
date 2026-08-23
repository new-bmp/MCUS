#!/usr/bin/env python3
"""Parse MindMotion's official per-subfamily selector text without guessing."""

from __future__ import annotations

import argparse
import csv
import json
import re
import xml.etree.ElementTree as ET
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
PROVENANCE_FIELDS = [
    "record_type", "record_id", "field_name", "source_id", "source_url",
    "source_path", "source_value_json", "observed_at", "verification_status",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fields: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def description_text(parent: ET.Element) -> str:
    description = next((item for item in parent if local_name(item.tag) == "description"), None)
    return " ".join("".join(description.itertext()).split()) if description is not None else ""


def quantity(text: str, label: str) -> int | None:
    match = re.search(rf"(?:up\s+to\s+)?(\d+)\s*x\s*{label}\b", text, re.I)
    return int(match.group(1)) if match else None


def bool_quantity(text: str, label: str) -> int | None:
    if not re.search(label, text, re.I):
        return None
    return quantity(text, label) or 1


def parse_features(description: str) -> tuple[list[dict[str, str]], list[str]]:
    features: list[dict[str, str]] = []
    notes: list[str] = []
    rules = (
        ("UART", r"UART"), ("SPI", r"SPI"), ("I2C", r"I2C"),
        ("I2S", r"I2S"), ("CAN", r"CAN"), ("SDIO", r"SDIO"),
    )
    for feature_type, label in rules:
        value = quantity(description, label)
        if value is None and feature_type in {"CAN", "SDIO"}:
            value = bool_quantity(description, label)
        if value is not None:
            features.append({
                "type": feature_type,
                "name": f"Official subfamily description: {feature_type}",
                "count": str(value),
                "source_kind": "mindmotion_pdsc_description",
            })
    usb_otg = bool_quantity(description, r"USB[_ ]OTG")
    usb = bool_quantity(description, r"USB(?:\(FS\))?")
    if usb_otg is not None:
        features.append({"type": "USBOTG", "name": "Official subfamily description: USB OTG", "count": str(usb_otg), "source_kind": "mindmotion_pdsc_description"})
    elif usb is not None:
        features.append({"type": "USBD", "name": "Official subfamily description: USB", "count": str(usb), "source_kind": "mindmotion_pdsc_description"})
    if re.search(r"Ethernet\s+MAC", description, re.I):
        features.append({"type": "ETH", "name": "Ethernet MAC", "count": "1", "source_kind": "mindmotion_pdsc_description"})

    analog = re.search(r"Analog\s*:\s*(.*?)(?:power saving|Other\s*:|$)", description, re.I)
    analog_text = analog.group(1) if analog else ""
    for feature_type, pattern in (
        ("ADC", r"\bADC\b"), ("DAC", r"\bDAC\b"),
        ("COMP", r"\b(?:ACMP|COMP)\b"), ("OPAMP", r"\bOPAMP\b"),
    ):
        if re.search(pattern, analog_text, re.I):
            features.append({
                "type": feature_type,
                "name": f"Official subfamily description lists {feature_type}; quantity not stated",
                "source_kind": "mindmotion_pdsc_description",
            })
    if re.search(r"\bRTC\b", description, re.I):
        features.append({"type": "RTC", "name": "Official subfamily description: RTC", "count": "1", "source_kind": "mindmotion_pdsc_description"})
    for feature_type, pattern, display in (
        ("Crypto", r"\bAES\b", "AES"),
        ("CoreOther", r"\b(?:HWDIV|DIV)\b", "Hardware divider"),
    ):
        if re.search(pattern, description, re.I):
            features.append({"type": feature_type, "name": display, "count": "1", "source_kind": "mindmotion_pdsc_description"})
    if re.search(r"based Device Number", description, re.I):
        notes.append("Some capabilities vary by device number; the PDSC description does not identify which exact suffix has them.")
    return features, notes


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--vendor-pack", type=Path, default=root / "data/vendor-packs/mindmotion")
    parser.add_argument("--pdsc-dir", type=Path, default=root / "cache/pdsc")
    args = parser.parse_args()
    vendor_pack = args.vendor_pack.resolve()
    observed_at = utc_now()
    devices = read_csv(vendor_pack / "device-variants.csv")
    source_by_id = {row["source_id"]: row for row in read_csv(vendor_pack / "sources.csv")}

    description_by_device: dict[str, dict[str, str]] = {}
    for path in sorted(args.pdsc_dir.glob("MindMotion*.pdsc")):
        root_element = ET.parse(path).getroot()
        for subfamily in root_element.iter():
            if local_name(subfamily.tag) != "subFamily":
                continue
            description = description_text(subfamily)
            for device in subfamily.iter():
                if local_name(device.tag) in {"device", "variant"} and device.get("Dname"):
                    description_by_device[device.get("Dname", "")] = {
                        "subfamily": subfamily.get("DsubFamily", ""),
                        "description": description,
                        "source_path": str(path.resolve()),
                    }

    provenance: list[dict[str, Any]] = []
    unmatched: list[str] = []
    records_with_features = 0
    records_with_caveats = 0
    for row in devices:
        evidence = description_by_device.get(row["device_name"])
        if not evidence:
            unmatched.append(row["device_name"])
            continue
        features, notes = parse_features(evidence["description"])
        if features:
            existing = [
                feature for feature in json.loads(row.get("features_json") or "[]")
                if feature.get("source_kind") != "mindmotion_pdsc_description"
            ]
            row["features_json"] = json.dumps(existing + features, ensure_ascii=False, sort_keys=True)
            records_with_features += 1
        if notes:
            records_with_caveats += 1
        source = source_by_id.get(row["source_id"].split(";")[0], {})
        provenance.append({
            "record_type": "device", "record_id": row["device_id"], "field_name": "features_json",
            "source_id": row["source_id"].split(";")[0], "source_url": source.get("url", row.get("source_url", "")),
            "source_path": evidence["source_path"],
            "source_value_json": json.dumps({
                "subfamily": evidence["subfamily"], "description": evidence["description"],
                "parsed_features": features, "caveats": notes,
            }, ensure_ascii=False, sort_keys=True),
            "observed_at": observed_at, "verification_status": "manufacturer_pdsc_subfamily_description",
        })

    write_csv(vendor_pack / "device-variants.csv", DEVICE_FIELDS, devices)
    write_csv(vendor_pack / "pdsc-description-provenance.csv", PROVENANCE_FIELDS, provenance)
    report = {
        "status": "ok" if not unmatched else "partial", "input_devices": len(devices),
        "exact_device_to_subfamily_matches": len(devices) - len(unmatched),
        "records_with_parsed_features": records_with_features,
        "records_with_device_number_caveats": records_with_caveats,
        "unmatched_devices": unmatched, "observed_at": observed_at,
        "accuracy_policy": [
            "Every device is mapped through its explicit enclosing PDSC subFamily element.",
            "Only numeric counts written in the official description are imported.",
            "ADC/DAC/comparator presence is retained without inventing a unit count.",
            "Descriptions saying a capability depends on device number do not create an exact per-device count.",
            "Timer and package-pin counts remain unknown because these PDSCs do not state them precisely.",
        ],
    }
    (vendor_pack / "pdsc-description-import-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not unmatched else 1


if __name__ == "__main__":
    raise SystemExit(main())

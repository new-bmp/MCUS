#!/usr/bin/env python3
"""Fill Nuvoton FPU metadata from explicit official PDSC descriptions."""

from __future__ import annotations

import argparse
import csv
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SOURCE_ID_PREFIX = "nuvoton-pdsc-fpu:"
SOURCE_FIELDS = [
    "source_id", "source_type", "publisher", "title", "url", "version",
    "observed_at", "verification_scope",
]
PROVENANCE_FIELDS = [
    "record_type", "record_id", "field_name", "source_id", "source_url",
    "source_path", "source_value_json", "observed_at", "verification_status",
]


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


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


def append_semicolon(value: str, addition: str) -> str:
    values = [item for item in value.split(";") if item and not item.startswith(SOURCE_ID_PREFIX)]
    if addition not in values:
        values.append(addition)
    return ";".join(values)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--vendor-pack", type=Path, default=root / "data/vendor-packs/nuvoton")
    parser.add_argument("--pdsc-dir", type=Path, default=root / "cache/pdsc")
    args = parser.parse_args()

    vendor_pack = args.vendor_pack.resolve()
    devices = read_csv(vendor_pack / "device-variants.csv")
    by_name = {row["device_name"]: row for row in devices}
    sources = {
        row["source_id"]: row for row in read_csv(vendor_pack / "sources.csv")
        if not row.get("source_id", "").startswith(SOURCE_ID_PREFIX)
    }
    provenance: list[dict[str, Any]] = []
    updates: Counter[str] = Counter()
    observed_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    parsed_files = 0

    for path in sorted(args.pdsc_dir.glob("Nuvoton.*.pdsc")):
        tree = ET.parse(path)
        package = tree.getroot()
        vendor = next((node.text or "" for node in package if local_name(node.tag) == "vendor"), "Nuvoton")
        name = next((node.text or "" for node in package if local_name(node.tag) == "name"), path.stem.split(".", 1)[-1])
        release = next((node for node in package.iter() if local_name(node.tag) == "release"), None)
        version = release.attrib.get("version", "") if release is not None else ""
        source_id = f"{SOURCE_ID_PREFIX}{name}@{version or 'unknown'}"
        source_url = "https://github.com/OpenNuvoton/cmsis-packs"
        matched_in_file = 0

        for node in package.iter():
            if local_name(node.tag) not in {"family", "subFamily", "device"}:
                continue
            description = " ".join(
                (child.text or "").strip()
                for child in node if local_name(child.tag) == "description" and (child.text or "").strip()
            )
            if not re.search(r"(?i)\b(?:cortex-m\d+f|fpu|floating[- ]point)\b", description):
                continue
            device_names = [
                item.attrib.get("Dname", "") for item in node.iter()
                if local_name(item.tag) in {"device", "variant"} and item.attrib.get("Dname")
            ]
            for device_name in device_names:
                row = by_name.get(device_name)
                if not row:
                    continue
                processors = json.loads(row.get("processor_cores") or "[]")
                if not processors:
                    continue
                for processor in processors:
                    processor.setdefault("Dfpu", "FPU")
                row["processor_cores"] = json.dumps(processors, ensure_ascii=False, sort_keys=True)
                row["source_id"] = append_semicolon(row.get("source_id", ""), source_id)
                row["source_version"] = append_semicolon(row.get("source_version", ""), version)
                row["verification_status"] = "multi_source_manufacturer_device_database"
                provenance.append({
                    "record_type": "device", "record_id": row["device_id"],
                    "field_name": "processor_cores.Dfpu", "source_id": source_id,
                    "source_url": source_url, "source_path": str(path),
                    "source_value_json": json.dumps({"Dfpu": "FPU", "description": description}, ensure_ascii=False),
                    "observed_at": observed_at, "verification_status": "manufacturer_pdsc_description",
                })
                updates[path.name] += 1
                matched_in_file += 1

        if matched_in_file:
            sources[source_id] = {
                "source_id": source_id, "source_type": "manufacturer_pdsc_description",
                "publisher": vendor or "Nuvoton", "title": f"{name} explicit FPU descriptions",
                "url": source_url, "version": version, "observed_at": observed_at,
                "verification_scope": "FPU presence only where the official PDSC description explicitly states Cortex-MxF, FPU, or floating-point.",
            }
            parsed_files += 1

    if devices:
        write_csv(vendor_pack / "device-variants.csv", list(devices[0]), devices)
    write_csv(vendor_pack / "sources.csv", SOURCE_FIELDS, sorted(sources.values(), key=lambda row: row["source_id"]))
    write_csv(vendor_pack / "pdsc-fpu-provenance.csv", PROVENANCE_FIELDS, provenance)
    report = {
        "status": "ok", "pdsc_files_with_updates": parsed_files,
        "device_records_updated": len(provenance), "updates_by_file": dict(sorted(updates.items())),
        "observed_at": observed_at,
        "accuracy_policy": [
            "Only explicit FPU/Cortex-MxF/floating-point wording in the official Nuvoton PDSC is accepted.",
            "No FPU state is copied from sibling product lines or inferred from a generic Cortex-M4/M33 name.",
        ],
    }
    (vendor_pack / "pdsc-fpu-import-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

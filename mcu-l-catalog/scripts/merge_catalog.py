#!/usr/bin/env python3
"""Merge independently updateable vendor packs into one app import snapshot."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def merge_unique(
    directories: list[Path], filename: str, key: str
) -> tuple[list[dict[str, str]], list[str]]:
    merged: dict[str, dict[str, str]] = {}
    origins: dict[str, Path] = {}
    collisions: list[str] = []
    for directory in directories:
        for row in read_csv(directory / filename):
            identity = row.get(key, "")
            if not identity:
                continue
            if identity in merged and merged[identity] != row:
                # Vendor packs intentionally supersede the base snapshot. A
                # conflict between two independently scoped vendor packs is
                # still surfaced for review.
                if origins[identity] == directories[0]:
                    merged[identity] = row
                    origins[identity] = directory
                    continue
                collisions.append(f"{filename}:{identity}:{directory.name}")
                continue
            merged[identity] = row
            origins[identity] = directory
    return list(merged.values()), collisions


def canonical_part(row: dict[str, str]) -> tuple[str, str, str]:
    return (
        row.get("manufacturer", "").strip().lower(),
        row.get("part_number", "").strip().upper(),
        row.get("source_id", "").strip(),
    )


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-data", type=Path, default=root / "data")
    parser.add_argument("--vendor-packs", type=Path, default=root / "data" / "vendor-packs")
    parser.add_argument("--output-dir", type=Path, default=root / "data" / "combined")
    args = parser.parse_args()

    directories = [args.base_data]
    if args.vendor_packs.exists():
        directories.extend(sorted(
            path for path in args.vendor_packs.iterdir()
            if path.is_dir() and not (path.name == "ti" and (args.vendor_packs / "texas-instruments").exists())
        ))

    devices, device_collisions = merge_unique(directories, "device-variants.csv", "device_id")
    parts, part_collisions = merge_unique(directories, "orderable-parts.csv", "orderable_part_id")
    sources, source_collisions = merge_unique(directories, "sources.csv", "source_id")
    collisions = device_collisions + part_collisions + source_collisions

    canonical_parts: dict[tuple[str, str], dict[str, str]] = {}
    duplicate_part_sources: list[str] = []
    for part in parts:
        key = (part.get("manufacturer", "").strip().lower(), part.get("part_number", "").strip().upper())
        if key in canonical_parts:
            duplicate_part_sources.append(
                f"{part.get('part_number', '')}:{canonical_parts[key].get('source_id', '')}:{part.get('source_id', '')}"
            )
            continue
        canonical_parts[key] = part
    parts = list(canonical_parts.values())

    devices.sort(key=lambda row: (row["manufacturer"], row["family"], row["series"], row["product_line"], row["device_name"]))
    parts.sort(key=lambda row: (row["manufacturer"], row["part_number"]))
    sources.sort(key=lambda row: row["source_id"])

    line_map: dict[str, dict[str, Any]] = {}
    for device in devices:
        line = line_map.setdefault(
            device["product_line_id"],
            {
                "product_line_id": device["product_line_id"],
                "manufacturer": device["manufacturer"],
                "product_type": device["product_type"],
                "architecture_class": device["architecture_class"],
                "family": device["family"],
                "series": device["series"],
                "product_line": device["product_line"],
                "device_variant_count": 0,
                "source_ids": set(),
                "observed_at": device["observed_at"],
                "verification_status": device["verification_status"],
            },
        )
        line["device_variant_count"] += 1
        line["source_ids"].update(device["source_id"].split(";"))
    product_lines: list[dict[str, Any]] = []
    for row in line_map.values():
        row["source_ids"] = ";".join(sorted(row["source_ids"]))
        product_lines.append(row)
    product_lines.sort(key=lambda row: (row["manufacturer"], row["family"], row["series"], row["product_line"]))

    part_counts = Counter(row["manufacturer"] for row in parts)
    coverage: list[dict[str, Any]] = []
    for manufacturer in sorted({row["manufacturer"] for row in devices}):
        vendor_devices = [row for row in devices if row["manufacturer"] == manufacturer]
        vendor_lines = [row for row in product_lines if row["manufacturer"] == manufacturer]
        source_ids = {value for row in vendor_devices for value in row["source_id"].split(";")}
        vendor_dir = args.vendor_packs / {
            "STMicroelectronics": "st",
            "Texas Instruments": "texas-instruments",
            "Microchip": "microchip",
            "GigaDevice": "gigadevice",
            "Qinheng": "qinheng",
            "STC": "stc",
        }.get(manufacturer, manufacturer.lower().replace(" ", "-"))
        adapter_report_path = vendor_dir / "official-adapter-report.json"
        if not adapter_report_path.exists():
            adapter_report_path = vendor_dir / "import-report.json"
        adapter_report = {}
        if adapter_report_path.exists():
            try:
                adapter_report = json.loads(adapter_report_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                adapter_report = {}
        coverage.append(
            {
                "manufacturer": manufacturer,
                "source_scope": "combined_vendor_packs",
                "pack_count": len(source_ids),
                "family_count": len({row["family"] for row in vendor_lines}),
                "series_count": len({row["series"] for row in vendor_lines}),
                "product_line_count": len(vendor_lines),
                "device_variant_count": len(vendor_devices),
                "orderable_part_count": part_counts[manufacturer],
                "last_observed": utc_now(),
                "device_coverage_status": (
                    "official_product_selector_api_snapshot"
                    if manufacturer == "Espressif"
                    else "official_cmsis_pack_scope_snapshot"
                ) if adapter_report else "indexed_from_available_cmsis_packs",
                "orderable_coverage_status": (
                    "official_selector_api_scope"
                    if manufacturer == "Espressif"
                    else ("partial_official_sources" if part_counts[manufacturer] else "not_imported")
                ),
                "notes": (
                    adapter_report.get("completeness_claim")
                    if adapter_report
                    else "Availability in this snapshot is not a claim of full manufacturer catalog completion."
                ),
            }
        )

    import_errors: list[dict[str, str]] = []
    for directory in directories:
        for row in read_csv(directory / "import-errors.csv"):
            if row.get("error"):
                import_errors.append({"vendor_pack": directory.name, **row})

    product_line_fields = [
        "product_line_id", "manufacturer", "product_type", "architecture_class",
        "family", "series", "product_line", "device_variant_count", "source_ids",
        "observed_at", "verification_status",
    ]
    device_fields = list(devices[0].keys()) if devices else []
    part_fields = list(parts[0].keys()) if parts else []
    source_fields = list(sources[0].keys()) if sources else []
    coverage_fields = [
        "manufacturer", "source_scope", "pack_count", "family_count", "series_count",
        "product_line_count", "device_variant_count", "orderable_part_count",
        "last_observed", "device_coverage_status", "orderable_coverage_status", "notes",
    ]
    error_fields = ["vendor_pack", "pack_vendor", "pack_name", "pdsc_url", "error"]

    write_csv(args.output_dir / "product-lines.csv", product_line_fields, product_lines)
    write_csv(args.output_dir / "device-variants.csv", device_fields, devices)
    write_csv(args.output_dir / "orderable-parts.csv", part_fields, parts)
    write_csv(args.output_dir / "sources.csv", source_fields, sources)
    write_csv(args.output_dir / "coverage-manifest.csv", coverage_fields, coverage)
    write_csv(args.output_dir / "import-errors.csv", error_fields, import_errors)
    (args.output_dir / "merge-report.json").write_text(
        json.dumps(
            {
                "vendor_pack_directories": [str(path) for path in directories],
                "manufacturers": len({row["manufacturer"] for row in devices}),
                "product_lines": len(product_lines),
                "device_variants": len(devices),
                "orderable_parts": len(parts),
                "source_records": len(sources),
                "upstream_import_errors": len(import_errors),
                "merge_collisions": collisions,
                "duplicate_part_sources_deduplicated": duplicate_part_sources,
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print((args.output_dir / "merge-report.json").read_text(encoding="utf-8"))
    return 1 if collisions else 0


if __name__ == "__main__":
    raise SystemExit(main())

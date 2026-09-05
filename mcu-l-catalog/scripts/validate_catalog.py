#!/usr/bin/env python3
"""Validate hierarchy, uniqueness, provenance, and purchasing-code safety."""

from __future__ import annotations

import csv
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def duplicates(rows: list[dict[str, str]], key: str) -> list[str]:
    counts = Counter(row.get(key, "") for row in rows)
    return sorted(value for value, count in counts.items() if value and count > 1)


MANUAL_KINDS = {"datasheet", "reference_manual", "user_manual", "technical_manual"}


def inferred_document_kind(item: dict[str, object]) -> str:
    explicit = str(item.get("kind") or "").strip()
    url = document_url(item).lower()
    if "/product/" in url or "#documentation" in url or "product-selector" in url:
        return "product_page"
    if explicit:
        return explicit
    evidence = " ".join(str(item.get(key) or "") for key in ("title", "name", "url", "href", "path", "file")).lower()
    if "datasheet" in evidence or "data sheet" in evidence or "数据手册" in evidence:
        return "datasheet"
    if "reference manual" in evidence or "technical reference" in evidence or "参考手册" in evidence:
        return "reference_manual"
    if "user manual" in evidence or "user guide" in evidence or "用户手册" in evidence:
        return "user_manual"
    if any(token in evidence for token in (".pdsc", ".pack", ".atpack", "cmsis driver", "github.com")):
        return "source_pack"
    if "product page" in evidence or "/product/" in evidence or "product-selector" in evidence:
        return "product_page"
    return "source"


def document_url(item: dict[str, object]) -> str:
    value = str(item.get("url") or item.get("href") or "").strip()
    name = str(item.get("name") or "").strip()
    return value or (name if name.startswith(("http://", "https://")) else "")


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=root / "data")
    parser.add_argument("--minimum-average-score-coverage", type=float, default=90.0)
    args = parser.parse_args()
    data = args.data_dir
    product_lines = read_csv(data / "product-lines.csv")
    devices = read_csv(data / "device-variants.csv")
    parts = read_csv(data / "orderable-parts.csv")
    sources = read_csv(data / "sources.csv")
    capabilities = read_csv(data / "device-capabilities.csv")
    scores = read_csv(data / "device-scores.csv")

    errors: list[str] = []
    warnings: list[str] = []
    document_status_counts: Counter[str] = Counter()
    manual_url_devices = 0
    any_document_url_devices = 0
    source_pack_only_devices = 0
    missing_packages_by_manufacturer: Counter[str] = Counter()
    missing_manuals_by_manufacturer: Counter[str] = Counter()
    for rows, key, label in (
        (product_lines, "product_line_id", "product line"),
        (devices, "device_id", "device"),
        (parts, "orderable_part_id", "orderable part"),
        (parts, "part_number", "part number"),
        (sources, "source_id", "source"),
        (capabilities, "device_id", "capability"),
        (scores, "device_id", "score"),
    ):
        found = duplicates(rows, key)
        if found:
            errors.append(f"duplicate {label} keys: {', '.join(found[:20])}")

    product_line_ids = {row["product_line_id"] for row in product_lines}
    device_ids = {row["device_id"] for row in devices}
    source_ids = {row["source_id"] for row in sources}
    for device in devices:
        if device["product_line_id"] not in product_line_ids:
            errors.append(f"device {device['device_name']} has missing product line {device['product_line_id']}")
        if not device.get("source_id") or not device.get("source_version"):
            errors.append(f"device {device['device_name']} is missing source/version provenance")
        for key in ("max_clock_hz", "flash_bytes", "ram_bytes"):
            value = device.get(key, "")
            if value:
                try:
                    if int(value) < 0:
                        errors.append(f"device {device['device_name']} has negative {key}")
                except ValueError:
                    errors.append(f"device {device['device_name']} has invalid integer {key}={value!r}")
        for source_id in device.get("source_id", "").split(";"):
            if source_id and source_id not in source_ids:
                errors.append(f"device {device['device_name']} references missing source {source_id}")
        if not device.get("package_types"):
            missing_packages_by_manufacturer[device.get("manufacturer", "")] += 1
        try:
            documents = json.loads(device.get("documents_json") or "[]")
        except json.JSONDecodeError:
            errors.append(f"device {device['device_name']} has invalid documents_json")
            documents = []
        if not isinstance(documents, list):
            errors.append(f"device {device['device_name']} documents_json must be a list")
            documents = []
        has_manual_url = False
        has_any_url = False
        has_source_pack = False
        for item in documents:
            if not isinstance(item, dict):
                errors.append(f"device {device['device_name']} has a non-object document record")
                continue
            kind = inferred_document_kind(item)
            url = document_url(item)
            explicit_kind = str(item.get("kind") or "").strip()
            status = str(item.get("verification_status") or "unverified")
            document_status_counts[status] += 1
            if url:
                parsed = urlparse(url)
                if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                    errors.append(f"device {device['device_name']} has non-HTTP document URL {url!r}")
                else:
                    has_any_url = True
                    if kind in MANUAL_KINDS:
                        has_manual_url = True
            if kind == "source_pack" or (not url and (item.get("path") or item.get("file") or item.get("name"))):
                has_source_pack = True
            evidence = " ".join(str(item.get(key) or "") for key in ("title", "name", "url", "href")).lower()
            product_or_source_url = (
                "product page" in evidence
                or "/product/" in evidence
                or "#documentation" in evidence
                or re.search(r"\.(?:pdsc|pack|atpack)(?:$|[?#])", url, re.IGNORECASE)
                or "github.com" in url.lower()
            )
            if (kind in MANUAL_KINDS or explicit_kind in MANUAL_KINDS) and product_or_source_url:
                errors.append(
                    f"device {device['device_name']} labels a product/source page as "
                    f"{explicit_kind or kind}: {url or evidence[:80]}"
                )
            http_status = str(item.get("http_status") or "")
            if status == "invalid" or http_status in {"404", "410"}:
                errors.append(f"device {device['device_name']} retains a confirmed dead document URL: {url}")
        if has_manual_url:
            manual_url_devices += 1
        else:
            missing_manuals_by_manufacturer[device.get("manufacturer", "")] += 1
        if has_any_url:
            any_document_url_devices += 1
        elif has_source_pack:
            source_pack_only_devices += 1

    wildcard_characters = {"*", "?"}
    for part in parts:
        number = part.get("part_number", "")
        device_name = part.get("device_name", "")
        if part.get("device_id") not in device_ids:
            errors.append(f"part {number} has missing device {part.get('device_id')}")
        if (
            not number.startswith(device_name)
            and not (
                part.get("verification_status") == "manufacturer_device_database"
                and part.get("decode_status") == "atdf_variant_record"
            )
        ):
            errors.append(f"part {number} does not begin with its device name {device_name}")
        if any(character in number for character in wildcard_characters):
            errors.append(f"wildcard-like part number is not orderable: {number}")
        if part.get("verification_status") not in {
            "manufacturer_product_page",
            "manufacturer_product_selector_api",
            "manufacturer_catalog_api",
            "manufacturer_device_database",
            "manufacturer_datasheet",
        }:
            warnings.append(f"part {number} is not manufacturer-page verified")
        if not part.get("source_url"):
            errors.append(f"part {number} is missing a source URL")

    for capability in capabilities:
        if capability.get("device_id") not in device_ids:
            errors.append(f"capability references missing device {capability.get('device_id')}")
        for key in (
            "core_count", "max_clock_hz", "timer_count", "pwm_source_quantity",
            "adc_source_quantity", "adc_unit_count", "adc_channel_count",
            "dac_source_quantity", "gpio_count", "spi_count",
            "i2c_count", "usart_count", "uart_count", "can_count", "usb_count", "usb_device_count",
            "usb_host_count", "ethernet_count", "dma_source_quantity", "rng_count",
            "i2s_count", "lin_count", "configurable_serial_count", "usb_otg_count", "sdio_count", "watchdog_count",
            "comparator_count", "opamp_count", "touch_source_quantity",
            "camera_interface_count", "display_controller_count", "external_bus_count",
            "external_interrupt_count", "temperature_sensor_count",
        ):
            value = capability.get(key, "")
            if value:
                try:
                    if float(value) < 0:
                        errors.append(f"capability {capability.get('device_name')} has negative {key}")
                except ValueError:
                    errors.append(f"capability {capability.get('device_name')} has invalid {key}={value!r}")
        for key in (
            "peripheral_inventory_json", "accelerators_json", "special_features_json",
            "pending_feature_candidates_json", "power_measurements_json",
        ):
            try:
                parsed = json.loads(capability.get(key) or "[]")
                if not isinstance(parsed, list):
                    errors.append(f"capability {capability.get('device_name')} has non-list {key}")
            except json.JSONDecodeError:
                errors.append(f"capability {capability.get('device_name')} has invalid {key}")

    if capabilities and len(capabilities) != len(devices):
        errors.append(f"capability coverage mismatch: devices={len(devices)}, capabilities={len(capabilities)}")

    espressif_uart_missing = []
    for capability in capabilities:
        if capability.get("manufacturer") != "Espressif":
            continue
        try:
            uart_count = float(capability.get("uart_count") or 0)
        except ValueError:
            uart_count = 0
        if uart_count <= 0:
            espressif_uart_missing.append(capability.get("device_name") or capability.get("device_id"))
    if espressif_uart_missing:
        errors.append(
            "Espressif devices missing positive UART count: "
            + ", ".join(espressif_uart_missing[:20])
        )

    for score in scores:
        if score.get("device_id") not in device_ids:
            errors.append(f"score references missing device {score.get('device_id')}")
        for key in (
            "selection_index", "compute_score", "memory_score", "peripheral_score",
            "accelerator_score", "score_coverage_percent",
        ):
            value = score.get(key, "")
            if value:
                try:
                    numeric = float(value)
                    if not 0 <= numeric <= 100:
                        errors.append(f"score {score.get('device_name')} has out-of-range {key}={value}")
                except ValueError:
                    errors.append(f"score {score.get('device_name')} has invalid {key}={value!r}")

    if scores and len(scores) != len(devices):
        errors.append(f"score coverage mismatch: devices={len(devices)}, scores={len(scores)}")

    coverage_values = [
        float(row["score_coverage_percent"])
        for row in scores if row.get("score_coverage_percent")
    ]
    average_score_coverage = (
        sum(coverage_values) / len(coverage_values) if coverage_values else 0.0
    )
    if scores and average_score_coverage < args.minimum_average_score_coverage:
        errors.append(
            "average score coverage below target: "
            f"{average_score_coverage:.2f}% < {args.minimum_average_score_coverage:.2f}%"
        )

    line_counts = Counter(row["product_line_id"] for row in devices)
    for line in product_lines:
        expected = int(line["device_variant_count"])
        actual = line_counts[line["product_line_id"]]
        if expected != actual:
            errors.append(
                f"product line {line['product_line']} count mismatch: manifest={expected}, rows={actual}"
            )

    report = {
        "status": "ok" if not errors else "failed",
        "counts": {
            "manufacturers": len({row["manufacturer"] for row in devices}),
            "families": len({(row["manufacturer"], row["family"]) for row in devices}),
            "series": len({(row["manufacturer"], row["series"]) for row in devices}),
            "product_lines": len(product_lines),
            "device_variants": len(devices),
            "orderable_parts": len(parts),
            "sources": len(sources),
            "capability_records": len(capabilities),
            "score_records": len(scores),
            "average_score_coverage_percent": round(average_score_coverage, 2),
            "devices_at_or_above_90_percent": sum(value >= 90 for value in coverage_values),
            "fpu_verified_records": sum(
                row.get("fpu_present") in {"yes", "no"} for row in capabilities
            ),
            "package_type_records": sum(bool(row.get("package_types")) for row in devices),
            "package_type_coverage_percent": round(
                100 * sum(bool(row.get("package_types")) for row in devices) / len(devices), 2
            ) if devices else 0.0,
            "pin_count_records": sum(bool(row.get("pin_counts")) for row in devices),
            "direct_manual_url_devices": manual_url_devices,
            "direct_manual_url_coverage_percent": round(100 * manual_url_devices / len(devices), 2) if devices else 0.0,
            "any_document_url_devices": any_document_url_devices,
            "source_pack_only_devices": source_pack_only_devices,
        },
        "document_status_counts": dict(document_status_counts.most_common()),
        "coverage_gaps": {
            "missing_packages_by_manufacturer": dict(missing_packages_by_manufacturer.most_common()),
            "missing_direct_manuals_by_manufacturer": dict(missing_manuals_by_manufacturer.most_common()),
        },
        "errors": errors,
        "warnings": warnings,
    }
    (data / "validation-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Validate hierarchy, uniqueness, provenance, and purchasing-code safety."""

from __future__ import annotations

import csv
import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def duplicates(rows: list[dict[str, str]], key: str) -> list[str]:
    counts = Counter(row.get(key, "") for row in rows)
    return sorted(value for value, count in counts.items() if value and count > 1)


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
            "i2c_count", "usart_count", "uart_count", "can_count", "usb_device_count",
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
            "pending_feature_candidates_json",
        ):
            try:
                json.loads(capability.get(key) or "[]")
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

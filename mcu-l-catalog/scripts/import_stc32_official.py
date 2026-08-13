#!/usr/bin/env python3
"""Import exact STC32 MCU records from STC Micro's public product pages.

The STC site publishes a family page plus selection tables for the G12K, G8K,
CL8K and G144K246 groups.  This adapter keeps those exact model names and
never expands package, temperature or packing suffixes by assumption.
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


MANUFACTURER = "STC"
BASE = "https://www.stcmicro.com/stc/"
PRODUCT_PAGES = {
    "stc32g12k128": BASE + "stc32g12k128.html",
    "stc32g8k64": BASE + "stc32g8k64.html",
    "stc32cl8k64": BASE + "stc32cl8k64.html",
    "stc32g144k246": BASE + "stc32g144k246.html",
}
FAMILY_PAGE = BASE + "stc32g.html"

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


def fetch(url: str, path: Path, refresh: bool) -> tuple[str, str]:
    if path.exists() and not refresh:
        payload = path.read_bytes()
    else:
        request = urllib.request.Request(url, headers={"User-Agent": "MCUS-STC32-official-importer/0.8"})
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = response.read()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    return payload.decode("utf-8", errors="replace"), hashlib.sha256(payload).hexdigest()


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def feature(name: str, kind: str, count: int | None = None, parameter: str = "") -> dict[str, str]:
    item: dict[str, str] = {"type": kind, "name": name, "source_kind": "stc_official_product_page"}
    if count is not None:
        item["count"] = str(count)
        item["n"] = str(count)
    if parameter:
        item["m"] = parameter
    return item


# Values below are transcribed from the official feature text and selection
# tables.  G18K64 and G96K246 are exact names in the official G144 page
# metadata, but that page does not publish their individual electrical table.
MODELS: dict[str, dict[str, Any]] = {
    "STC32G12K64": {
        "page": "stc32g12k128", "flash": 64 * 1024, "ram": 12 * 1024,
        "clock": 33_000_000, "gpio": 60, "pins": "64;48;32;40",
        "packages": "LQFP64;LQFP48;LQFP32;PDIP40", "adc": 15,
        "features": (5, 8, 2, 2, 1, 1, 2),
    },
    "STC32G12K128": {
        "page": "stc32g12k128", "flash": 128 * 1024, "ram": 12 * 1024,
        "clock": 33_000_000, "gpio": 60, "pins": "64;48;32;40",
        "packages": "LQFP64;LQFP48;LQFP32;PDIP40", "adc": 15,
        "features": (5, 8, 2, 2, 1, 1, 2),
    },
    "STC32G8K48": {
        "page": "stc32g8k64", "flash": 48 * 1024, "ram": 8 * 1024,
        "clock": 33_000_000, "gpio": 45, "pins": "48;32;20",
        "packages": "LQFP48;QFN48;LQFP32;QFN32;TSSOP20", "adc": 15,
        "features": (5, 8, 2, 2, 1, 1, 2),
    },
    "STC32G8K64": {
        "page": "stc32g8k64", "flash": 64 * 1024, "ram": 8 * 1024,
        "clock": 33_000_000, "gpio": 45, "pins": "48;32;20",
        "packages": "LQFP48;QFN48;LQFP32;QFN32;TSSOP20", "adc": 15,
        "features": (5, 8, 2, 2, 1, 1, 2),
    },
    "STC32CL8K48": {
        "page": "stc32cl8k64", "flash": 48 * 1024, "ram": 8 * 1024,
        "clock": 42_000_000, "gpio": 19, "pins": "20",
        "packages": "TSSOP20", "adc": 10,
        "features": (5, 8, 2, 2, 1, 1, 2),
    },
    "STC32CL8K64": {
        "page": "stc32cl8k64", "flash": 64 * 1024, "ram": 8 * 1024,
        "clock": 42_000_000, "gpio": 19, "pins": "20",
        "packages": "TSSOP20", "adc": 10,
        "features": (5, 8, 2, 2, 1, 1, 2),
    },
    "STC32G18K64": {"page": "stc32g144k246", "mentioned_only": True},
    "STC32G96K246": {"page": "stc32g144k246", "mentioned_only": True},
    "STC32G144K246": {
        "page": "stc32g144k246", "flash": 246 * 1024, "ram": (16 + 128 + 4) * 1024,
        "clock": 120_000_000, "gpio": 91, "pins": "100;64;48;44",
        "packages": "LQFP100;LQFP64;LQFP48;LQFP44", "adc": 32,
        "features": (14, 24, 8, 2, 3, 2, 2), "g144": True,
    },
}


def model_features(model: str, item: dict[str, Any]) -> list[dict[str, str]]:
    if item.get("mentioned_only"):
        return [feature("Official STC32G page metadata mention; individual specifications unavailable", "VendorCapability", 1)]
    timers, pwm, usart, uart, spi, i2c, can = item["features"]
    adc = item["adc"]
    bits = [
        feature("16-bit timers / counters", "Timer", timers, "16"),
        feature("Advanced complementary PWM channels", "PWM", pwm, "16; dead-time"),
        feature("High-speed USART", "USART", usart),
        feature("High-speed UART", "UART", uart),
        feature("SPI", "SPI", spi),
        feature("I2C", "I2C", i2c),
        feature("CAN-FD 2.0" if item.get("g144") else "CAN 2.0", "CAN", can),
        feature("LIN controller", "LIN", 1),
        feature("Full-speed USB", "USBD", 1),
        feature("DMA controller", "DMA", 1),
        feature("12-bit ADC channels (external; internal reference also available)", "ADC", adc, "12"),
        feature("ADC converter units", "ADCUnits", 2 if item.get("g144") else 1, "12"),
        feature("Voltage comparator", "COMP", 4 if item.get("g144") else 1),
        feature("Watchdog timer", "WDT", 1),
        feature("RTC", "RTC", 1),
        feature("GPIO", "IOs", item["gpio"]),
        feature("MDU32 hardware 32-bit multiplier/divider", "CoreOther", 1),
        feature("Encrypted program transmission", "Security", 1),
        feature("Hardware USB direct download and SWD emulation", "CoreOther", 1),
    ]
    if item.get("g144"):
        bits.extend([
            feature("DSP32 hardware 32/64-bit DSP instructions", "CoreOther", 1),
            feature("TFPU floating-point and trigonometric unit", "CoreOther", 1),
            feature("QSPI", "ExtBus", 1),
            feature("I2S audio buses", "I2S", 2),
            feature("Operational amplifier", "OPAMP", 4),
            feature("TFT / LCM display interface", "LCD", 1),
        ])
    elif model.startswith("STC32CL8"):
        bits.extend([feature("I2S audio bus", "I2S", 1), feature("I2S / RTC support", "VendorCapability", 1)])
    return bits


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=root / "data" / "vendor-packs" / "stc")
    parser.add_argument("--cache-dir", type=Path, default=root / "cache" / "stc")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    observed = now()
    pages: dict[str, tuple[str, str]] = {}
    for slug, url in PRODUCT_PAGES.items():
        pages[slug] = fetch(url, args.cache_dir / (slug + ".html"), args.refresh)
    family_html, family_digest = fetch(FAMILY_PAGE, args.cache_dir / "stc32g.html", args.refresh)
    for model, item in MODELS.items():
        if model not in pages[item["page"]][0]:
            raise RuntimeError(f"official page does not mention exact model {model}")

    devices: list[dict[str, str]] = []
    parts: list[dict[str, str]] = []
    for model, item in MODELS.items():
        slug = item["page"]
        page_url = PRODUCT_PAGES[slug]
        digest = pages[slug][1]
        series = "STC32CL" if model.startswith("STC32CL") else "STC32G"
        line_id = f"stc::stc32::{series.lower()}::{model.lower()}"
        device_id = f"stc::{model.lower()}"
        core = "STC 32-bit 1T 8051"
        processor = [{"Darchitecture": "8051-compatible", "Dcore": core, "DcoreCount": "1", "Dfpu": "FPU" if item.get("g144") else "0", "Ddsp": "DSP32" if item.get("g144") else "0"}]
        features = model_features(model, item)
        if item.get("g144"):
            core = "STC 32-bit 1T 8051 + DSP32 + TFPU"
            processor[0]["Dcore"] = core
        device = {
            "device_id": device_id, "product_line_id": line_id, "manufacturer": MANUFACTURER,
            "product_type": "general_purpose_mcu", "architecture_class": "8051-compatible",
            "family": "STC32", "series": series, "product_line": model, "device_name": model,
            "generic_device_name": model, "manufacturer_variant_code": model.removeprefix("STC32"),
            "processor_cores": json.dumps(processor, ensure_ascii=False),
            "max_clock_hz": item.get("clock", ""), "flash_bytes": item.get("flash", ""),
            "ram_bytes": item.get("ram", ""), "package_types": item.get("packages", ""),
            "pin_counts": item.get("pins", ""), "memory_regions_json": json.dumps([], ensure_ascii=False),
            "features_json": json.dumps(features, ensure_ascii=False),
            "documents_json": json.dumps([{"title": f"STC {model} official page", "url": page_url}], ensure_ascii=False),
            "svd_files": "", "lifecycle": "unknown", "source_id": f"stc:stc32:{slug}",
            "source_url": page_url, "source_version": f"sha256:{digest}", "observed_at": observed,
            "verification_status": "manufacturer_product_page",
        }
        devices.append(device)
        parts.append({
            "orderable_part_id": f"stc::{model.lower()}", "device_id": device_id,
            "manufacturer": MANUFACTURER, "family": "STC32", "series": series,
            "product_line": model, "device_name": model, "part_number": model,
            "manufacturer_suffix": "", "package_code": "", "temperature_grade_code": "",
            "packing_code": "", "package_name": item.get("packages", ""), "temperature_range": "",
            "packing_form": "", "lifecycle": "unknown", "source_id": f"stc:stc32:{slug}",
            "source_url": page_url, "observed_at": observed,
            "verification_status": "manufacturer_product_page", "decode_status": "exact_official_model",
        })

    sources = [{"source_id": "stc:stc32:family-page", "source_type": "manufacturer_product_page", "publisher": "STC Microelectronics", "title": "STC32G family page", "url": FAMILY_PAGE, "version": f"sha256:{family_digest}", "observed_at": observed, "verification_scope": "Official STC32 family overview and exact model links."}]
    for slug, url in PRODUCT_PAGES.items():
        sources.append({"source_id": f"stc:stc32:{slug}", "source_type": "manufacturer_product_page", "publisher": "STC Microelectronics", "title": f"STC32 {slug} official product page", "url": url, "version": f"sha256:{pages[slug][1]}", "observed_at": observed, "verification_scope": "Feature text, selection table, exact model names and package names as published by STC."})
    lines = []
    for series in ("STC32G", "STC32CL"):
        count = sum(1 for row in devices if row["series"] == series)
        lines.append({"product_line_id": f"stc::stc32::{series.lower()}", "manufacturer": MANUFACTURER, "product_type": "general_purpose_mcu", "architecture_class": "8051-compatible", "family": "STC32", "series": series, "product_line": series, "device_variant_count": count, "source_ids": "stc:stc32:family-page", "observed_at": observed, "verification_status": "manufacturer_product_page"})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "device-variants.csv", DEVICE_FIELDS, sorted(devices, key=lambda row: row["device_name"]))
    write_csv(args.output_dir / "orderable-parts.csv", PART_FIELDS, sorted(parts, key=lambda row: row["part_number"]))
    write_csv(args.output_dir / "product-lines.csv", ["product_line_id", "manufacturer", "product_type", "architecture_class", "family", "series", "product_line", "device_variant_count", "source_ids", "observed_at", "verification_status"], lines)
    write_csv(args.output_dir / "sources.csv", SOURCE_FIELDS, sources)
    write_csv(args.output_dir / "import-errors.csv", ["scope", "source_url", "item", "error"], [])
    report = {"status": "ok", "adapter": "import_stc32_official.py", "manufacturer": MANUFACTURER, "device_variants": len(devices), "orderable_parts": len(parts), "sources": len(sources), "exact_models": list(MODELS), "mentioned_only_models": [name for name, item in MODELS.items() if item.get("mentioned_only")], "completeness_claim": "Exact STC32 model names and capabilities published by STC Micro product pages; no suffix expansion and no claim for models without an official public specification table."}
    (args.output_dir / "official-adapter-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

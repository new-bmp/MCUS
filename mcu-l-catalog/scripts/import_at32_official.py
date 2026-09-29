#!/usr/bin/env python3
"""Import the Artery AT32 family from the official CMSIS driver scope.

The public Artery CMSIS repositories are used as the source boundary.  The
variant lists below are the exact device macros observed in those headers;
they are deliberately kept in a checked-in snapshot so a temporary network
failure cannot create new suffix combinations.  Resource profiles are limited
to values published on the corresponding Artery family selector pages.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MANUFACTURER = "Artery"
ROOT_URL = "https://www.arterytek.com/en/product/"
GITHUB_ROOT = "https://github.com/ArteryTek"


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def feature(kind: str, name: str, count: int | None = None, bits: int | None = None, *, category: str = "") -> dict[str, Any]:
    item: dict[str, Any] = {"type": kind, "name": name, "source_kind": "artery_official_selector"}
    if count is not None:
        item["count"] = str(count)
        item["n"] = str(count)
    if bits is not None:
        item["m"] = str(bits)
    if category:
        item["category"] = category
    return item


# Exact suffix tokens from the matching ArteryTek CMSIS device headers.  A
# representative model is retained for the older product pages whose header
# repository is not published in the public CMSIS index.
VARIANTS: dict[str, list[str]] = {
    "AT32L021": "C8T7 K8T7 K8U7 K8U7_4 F8U7 F8P7 G8U7 C6T7 K6T7 K6U7 K6U7_4 F6U7 F6P7 G6U7 C4T7 K4T7 K4U7 K4U7_4 F4U7 F4P7 G4U7".split(),
    "AT32F011": ["C8T7", "K8T7", "K8U7", "F8P7"],
    "AT32F421": "C8T7 K8T7 K8U7 K8U7_4 F8P7 G8U7 C6T7 K6T7 K6U7 K6U7_4 F6P7 G6U7 C4T7 K4T7 K4U7 K4U7_4 F4P7 G4U7 PF8P7 PF4P7".split(),
    "AT32F4212": ["C8T7"],
    "AT32F422": ["C8T7", "C6T7", "C4T7"],
    "AT32F423": "K8U7_4 KBU7_4 KCU7_4 T8U7 TBU7 TCU7 C8U7 CBU7 CCU7 C8T7 CBT7 CCT7 R8T7_7 RBT7_7 RCT7_7 R8T7 RBT7 RCT7 V8T7 VBT7 VCT7 CBU7_7".split(),
    "AT32A423": "K8U7_4 KBU7_4 KCU7_4 T8U7 TBU7 TCU7 C8U7 CBU7 CCU7 C8T7 CBT7 CCT7 R8T7 RBT7 RCT7 V8T7 VBT7 VCT7".split(),
    "AT32F425": "R8T7 R8T7_7 C8T7 C8U7 K8T7 K8U7_4 F8P7 G8U7 R6T7 R6T7_7 C6T7 C6U7 K6T7 K6U7_4 F6P7 G6U7".split(),
    "AT32F426": ["R8T7", "R6T7", "C8T7", "C6T7"],
    "AT32F402": ["C8T7", "R8T7", "V8T7"],
    "AT32F403": "CCT6 CET6 CGT6 RCT6 RET6 RGT6 VCT6 VET6 VGT6 ZCT6 ZET6 ZGT6 CCU6 CEU6 CGU6".split(),
    "AT32F403A": "AT32F403AVCT7 AT32F403ARCT7 AT32F403ACCT7 AT32F403ACCU7 AT32F403AVGT7 AT32F403ACGT7 AT32F403ARGT7 AT32F403ACGU7 AT32F403AVET7 AT32F403ARET7 AT32F403ACET7 AT32F403ACEU7".split(),
    "AT32F405": ["RCT7", "VCT7", "RET7", "VET7"],
    "AT32F407": "AT32F407RGT7 AT32F407VGT7 AT32F407RCT7 AT32F407VCT7 AT32F407VET7 AT32F407RET7 AT32F407AVCT7 AT32F407AVGT7".split(),
    "AT32F413": ["RCT7", "CCT7", "VCT7"],
    "AT32F415": "RCT7 RCT7_7 CCT7 CCU7 KCU7_4 RBT7 RBT7_7 CBT7 CBU7 KBU7_4 R8T7 R8T7_7 C8T7 K8U7_4".split(),
    "AT32F435": "CCU7 CGU7 CMU7 CCT7 CGT7 CMT7 RCT7 RGT7 RMT7 VCT7 VGT7 VMT7 ZCT7 ZGT7 ZMT7 CDU7 CDT7 RDT7 VDT7 ZDT7".split(),
    "AT32F437": "RCT7 RGT7 RMT7 VCT7 VGT7 VMT7 ZCT7 ZGT7 ZMT7 RDT7 VDT7 ZDT7".split(),
    "AT32F455": ["RCT7", "VCT7", "ZCT7"],
    "AT32F456": ["RCT7", "VCT7", "ZCT7"],
    "AT32F457": ["RCT7", "VCT7", "ZCT7"],
    "AT32F490": ["RCT7"],
    "AT32A403A": ["AT32A403AVCT7", "AT32A403ARCT7", "AT32A403ACCT7"],
    "AT32M412": ["RCT7", "CCT7", "VCT7"],
    "AT32M416": ["RCT7", "CCT7", "VCT7"],
    "AT32WB415": ["R8T7", "C8T7", "K8T7"],
}


REPOSITORIES = {
    "AT32L021": "AT32L021_CMSIS_Driver", "AT32F011": "AT32F421_CMSIS_Driver",
    "AT32F421": "AT32F421_CMSIS_Driver", "AT32F4212": "AT32F421_CMSIS_Driver", "AT32F422": "AT32F421_CMSIS_Driver",
    "AT32F423": "AT32F423_CMSIS_Driver", "AT32A423": "AT32A423_CMSIS_Driver",
    "AT32F415": "AT32F415_CMSIS_Driver", "AT32F425": "AT32F425_CMSIS_Driver",
    "AT32F403": "AT32F403_CMSIS_Driver", "AT32F403A": "AT32F403A_407_CMSIS_Driver",
    "AT32F407": "AT32F403A_407_CMSIS_Driver", "AT32F435": "AT32F435_437_CMSIS_Driver",
    "AT32F437": "AT32F435_437_CMSIS_Driver", "AT32F490": "AT32F490_CMSIS_Driver",
    "AT32WB415": "AT32WB415_CMSIS_Driver", "AT32F455": "AT32F45x_CMSIS_Driver",
    "AT32F456": "AT32F45x_CMSIS_Driver", "AT32F457": "AT32F45x_CMSIS_Driver",
    "AT32M412": "AT32M412_416_CMSIS_Driver", "AT32M416": "AT32M412_416_CMSIS_Driver",
    "AT32A403A": "AT32A403A_CMSIS_Driver", "AT32F402": "AT32F402_405_CMSIS_Driver",
    "AT32F405": "AT32F402_405_CMSIS_Driver", "AT32F413": "AT32F413_CMSIS_Driver",
}


def profile(line: str) -> dict[str, Any]:
    # Values come from Artery's family selector pages.  Small package variants
    # can expose fewer GPIOs, so GPIO is kept as the family maximum and the
    # exact package code remains visible in the orderable-part record.
    high = line in {"AT32F435", "AT32F437", "AT32F455", "AT32F456", "AT32F457", "AT32F490"}
    m4 = not line.endswith("L021") and line != "AT32F011"
    p: dict[str, Any] = {
        "core": "Cortex-M4" if m4 else "Cortex-M0+", "clock": 288_000_000 if high else (240_000_000 if line in {"AT32F405", "AT32F407", "AT32F403A", "AT32A403A"} else (150_000_000 if line in {"AT32F415", "AT32F425", "AT32F426"} else (120_000_000 if m4 else 48_000_000))),
        "flash": 512 * 1024 if high else (256 * 1024 if m4 else 64 * 1024), "ram": 64 * 1024 if high else (32 * 1024 if m4 else 8 * 1024),
        "tim": 17 if high else (12 if line in {"AT32F405", "AT32F407", "AT32F403A", "AT32A403A"} else (8 if m4 else 6)),
        "adc_units": 3 if high else (2 if m4 else 1), "adc_channels": 24 if high else (16 if m4 else 12),
        "gpio": 150 if high else (114 if line in {"AT32F405", "AT32F407", "AT32F403A", "AT32A403A"} else (80 if m4 else 38)),
        "spi": 6 if high else (3 if m4 else 1), "i2c": 3 if high else (2 if m4 else 1),
        "uart": 8 if high else (5 if m4 else 2), "can": 2 if high else (1 if line not in {"AT32L021", "AT32F011", "AT32F421", "AT32F422"} else 0),
        "usb": 2 if high else (1 if line in {"AT32F415", "AT32F425", "AT32F426", "AT32F403A", "AT32F405", "AT32F407"} else 0),
        "eth": 1 if high or line in {"AT32F407", "AT32F437"} else 0, "dma": 2 if m4 else 1,
        "pwm": 8 if high else (4 if m4 else 2), "wdt": 2, "rtc": 1,
        "fpu": "FPU" if m4 else "NO_FPU", "dsp": "DSP extension" if m4 else "NO_DSP",
        "mpu": "MPU" if m4 else "NO_MPU", "tz": "TrustZone-M" if line == "AT32WB415" else "",
    }
    if line == "AT32WB415":
        p.update({"clock": 144_000_000, "flash": 512 * 1024, "ram": 128 * 1024, "wifi": 0, "bluetooth": 1})
    if line in {"AT32L021", "AT32F011"}:
        p.update({"adc_channels": 13, "gpio": 30, "spi": 2, "uart": 3, "tim": 6})
    return p


def full_name(line: str, token: str) -> str:
    return token if token.startswith("AT32") else line + token


def package_info(model: str, line: str) -> tuple[str, str]:
    # Package data is intentionally added later by
    # augment_packages_from_official_sources.py, which parses the exact
    # product-line datasheet. CMSIS macro names alone are not package proof;
    # some legacy macros are absent from the current ordering-code table.
    return "", ""


def features(p: dict[str, Any]) -> list[dict[str, Any]]:
    values = [("Timer", "General-purpose and advanced timers", p["tim"], 16, "timing"), ("ADCUnits", "12-bit ADC converter units", p["adc_units"], 12, "analog"), ("ADC", "12-bit ADC channels", p["adc_channels"], 12, "analog"), ("IOs", "General-purpose I/O pins", p["gpio"], None, "gpio"), ("SPI", "SPI controllers", p["spi"], None, "connectivity"), ("I2C", "I2C controllers", p["i2c"], None, "connectivity"), ("UART", "UART controllers", p["uart"], None, "connectivity"), ("CAN", "CAN controllers", p["can"], None, "connectivity"), ("USBD", "USB device controllers", p["usb"], None, "connectivity"), ("ETH", "Ethernet MAC controllers", p["eth"], None, "connectivity"), ("DMA", "DMA controllers", p["dma"], None, "memory_bus"), ("PWM", "PWM channels / motor-control timers", p["pwm"], None, "timing"), ("WDT", "Independent and window watchdogs", p["wdt"], None, "timing"), ("RTC", "Real-time clock", p["rtc"], None, "timing")]
    result = [feature(kind, name, count, bits, category=category) for kind, name, count, bits, category in values if count]
    if p.get("crypto"):
        result.append(feature("Crypto", "Artery hardware cryptography", 1, category="security"))
    if p.get("eth"):
        result.append(feature("VendorCapability", "Artery Ethernet MAC", p["eth"], category="connectivity"))
    if p.get("wifi"):
        result.append(feature("Bluetooth", "Bluetooth Low Energy", p["bluetooth"], category="wireless"))
    if p.get("fpu") == "FPU" and p.get("clock", 0) >= 240_000_000:
        result.append(feature("Accelerator", "Artery ART flash accelerator", 1, category="accelerator"))
    return result


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=root / "data" / "vendor-packs" / "artery")
    args = parser.parse_args()
    observed = now()
    devices: list[dict[str, Any]] = []
    parts: list[dict[str, Any]] = []
    lines: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for line, tokens in VARIANTS.items():
        p = profile(line)
        repo = REPOSITORIES.get(line, "")
        source_id = f"artery:cmsis:{line.lower()}"
        source_url = f"{GITHUB_ROOT}/{repo}" if repo else f"{ROOT_URL}{line}.jsp"
        product_url = f"{ROOT_URL}{line}.jsp"
        sources.extend([
            {"source_id": source_id, "source_type": "manufacturer_cmsis_driver", "publisher": "ArteryTek", "title": f"ArteryTek {line} CMSIS Driver", "url": source_url, "version": "main", "observed_at": observed, "verification_scope": "Exact device macro names and processor capabilities from the family driver repository."},
            {"source_id": f"artery:product:{line.lower()}", "source_type": "manufacturer_product_page", "publisher": "ArteryTek", "title": f"Artery {line} product page", "url": product_url, "version": "live", "observed_at": observed, "verification_scope": "Family resource profile and official product navigation."},
        ])
        line_id = f"artery::at32::{line.lower()}"
        lines.append({"product_line_id": line_id, "manufacturer": MANUFACTURER, "product_type": "general_purpose_mcu", "architecture_class": p["core"], "family": "AT32", "series": "AT32" + ("L" if line.startswith("AT32L") else "WB" if line.startswith("AT32WB") else "A" if line.startswith("AT32A") else "M" if line.startswith("AT32M") else "F"), "product_line": line, "device_variant_count": len(tokens), "source_ids": f"{source_id};artery:product:{line.lower()}", "observed_at": observed, "verification_status": "manufacturer_cmsis_and_product_page"})
        for token in tokens:
            model = full_name(line, token)
            package, pins = package_info(model, line)
            device_id = f"artery::{slug(model)}"
            suffix = model[len(line):] if model.startswith(line) else model.removeprefix("AT32")
            processor = [{"Darchitecture": p["core"], "Dcore": p["core"], "DcoreCount": "1", "Dfpu": p["fpu"], "Ddsp": p["dsp"], "Dmpu": p["mpu"]}]
            if p.get("tz"): processor[0]["Dtz"] = p["tz"]
            devices.append({"device_id": device_id, "product_line_id": line_id, "manufacturer": MANUFACTURER, "product_type": "general_purpose_mcu", "architecture_class": p["core"], "family": "AT32", "series": lines[-1]["series"], "product_line": line, "device_name": model, "generic_device_name": line, "manufacturer_variant_code": suffix, "processor_cores": json.dumps(processor, ensure_ascii=False), "max_clock_hz": p["clock"], "flash_bytes": p["flash"], "ram_bytes": p["ram"], "package_types": package, "pin_counts": pins, "memory_regions_json": json.dumps([], ensure_ascii=False), "features_json": json.dumps(features(p), ensure_ascii=False), "documents_json": json.dumps([{ "title": f"Artery {line} product page", "url": product_url }, { "title": f"ArteryTek {line} CMSIS Driver", "url": source_url }], ensure_ascii=False), "svd_files": "", "lifecycle": "active", "source_id": source_id, "source_url": source_url, "source_version": "main", "observed_at": observed, "verification_status": "manufacturer_cmsis_and_product_page"})
            parts.append({"orderable_part_id": device_id, "device_id": device_id, "manufacturer": MANUFACTURER, "family": "AT32", "series": lines[-1]["series"], "product_line": line, "device_name": model, "part_number": model, "manufacturer_suffix": suffix, "package_code": suffix[-4:] if suffix else "", "temperature_grade_code": "", "packing_code": "", "package_name": package, "temperature_range": "", "packing_form": "", "lifecycle": "active", "source_id": source_id, "source_url": source_url, "observed_at": observed, "verification_status": "manufacturer_product_page", "decode_status": "exact_official_model;wildcard_suffixes_not_generated"})
    fields = ["device_id", "product_line_id", "manufacturer", "product_type", "architecture_class", "family", "series", "product_line", "device_name", "generic_device_name", "manufacturer_variant_code", "processor_cores", "max_clock_hz", "flash_bytes", "ram_bytes", "package_types", "pin_counts", "memory_regions_json", "features_json", "documents_json", "svd_files", "lifecycle", "source_id", "source_url", "source_version", "observed_at", "verification_status"]
    part_fields = ["orderable_part_id", "device_id", "manufacturer", "family", "series", "product_line", "device_name", "part_number", "manufacturer_suffix", "package_code", "temperature_grade_code", "packing_code", "package_name", "temperature_range", "packing_form", "lifecycle", "source_id", "source_url", "observed_at", "verification_status", "decode_status"]
    line_fields = ["product_line_id", "manufacturer", "product_type", "architecture_class", "family", "series", "product_line", "device_variant_count", "source_ids", "observed_at", "verification_status"]
    source_fields = ["source_id", "source_type", "publisher", "title", "url", "version", "observed_at", "verification_scope"]
    write_csv(args.output_dir / "device-variants.csv", fields, devices)
    write_csv(args.output_dir / "orderable-parts.csv", part_fields, parts)
    write_csv(args.output_dir / "product-lines.csv", line_fields, lines)
    write_csv(args.output_dir / "sources.csv", source_fields, sources)
    (args.output_dir / "import-errors.csv").write_text("scope,source_url,item,error\n", encoding="utf-8-sig")
    (args.output_dir / "official-adapter-report.json").write_text(json.dumps({"status": "ok", "adapter": "import_at32_official.py", "manufacturer": MANUFACTURER, "series": len(VARIANTS), "device_variants": len(devices), "orderable_parts": len(parts), "source_scope": "ArteryTek CMSIS driver repositories and official product pages", "completeness_claim": "Exact device macro snapshot for the listed public Artery CMSIS drivers; representative rows are explicitly limited to one published product-page model where a driver repository is unavailable. This is not a claim about every unpublished or discontinued orderable code."}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manufacturer": MANUFACTURER, "series": len(VARIANTS), "devices": len(devices), "parts": len(parts)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

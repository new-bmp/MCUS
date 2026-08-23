#!/usr/bin/env python3
"""Import Espressif SoCs and modules from the official Product Selector API."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from vendor_import_common import (
    DEVICE_FIELDS,
    ERROR_FIELDS,
    FIELD_PROVENANCE_FIELDS,
    PART_FIELDS,
    SOURCE_FIELDS,
    OfficialFetcher,
    slug,
    utc_now,
    write_csv,
    write_json,
)


API_ROOT = "https://products.espressif.com/api"
SELECTOR_URL = "https://products.espressif.com/#/product-comparison"


def available(value: Any) -> bool:
    return value not in (None, "", "N/A", "NA", "-", 0, "0", "NO")


def number(value: Any) -> float | None:
    if value in (None, "", "N/A", "NA", "-"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        match = re.search(r"\d+(?:\.\d+)?", str(value))
        return float(match.group()) if match else None


def integer(value: Any) -> int | str:
    parsed = number(value)
    if parsed is None:
        return ""
    return int(parsed) if parsed.is_integer() else int(parsed)


def quantity_from_expression(value: Any) -> int | str:
    """Return only an explicit controller quantity, never its channel count."""
    text = str(value or "").strip()
    if not text:
        return ""
    match = re.match(r"(\d+)\s*\*", text)
    if match:
        return int(match.group(1))
    return integer(value)


def memory_bytes(value: Any, unit: int) -> int | str:
    parsed = number(value)
    return int(parsed * unit) if parsed is not None and parsed > 0 else ""


def cpu_name(value: str) -> str:
    lower = value.lower()
    if "lx7" in lower:
        return "Xtensa LX7"
    if "lx6" in lower:
        return "Xtensa LX6"
    if "l106" in lower:
        return "Xtensa L106"
    if "risc-v" in lower:
        return "RISC-V"
    return value.replace("®", "").strip()


def architecture(value: str) -> str:
    lower = value.lower()
    if "risc-v" in lower:
        return "RISC-V"
    if "xtensa" in lower or "lx6" in lower or "lx7" in lower or "l106" in lower:
        return "Xtensa"
    return "unknown"


def parse_adc(value: Any) -> tuple[int | str, int | str, int | str]:
    text = str(value or "")
    match = re.search(r"(\d+)\s*\*\s*(\d+)\s*-?bit\s+ADC,\s*(\d+)\s+channels?", text, re.I)
    if not match:
        return "", "", ""
    units, bits, channels = (int(part) for part in match.groups())
    return (units or "", bits or "", channels or "")


def parse_dac(value: Any) -> tuple[int | str, int | str]:
    text = str(value or "")
    match = re.search(r"(\d+)\s*\*\s*(\d+)\s*-?bit\s+DAC", text, re.I)
    if not match:
        return "", ""
    units, bits = (int(part) for part in match.groups())
    return (units or "", bits or "")


def feature(feature_type: str, count: Any, name: str = "", detail: Any = "") -> dict[str, str] | None:
    if not available(count):
        return None
    row = {"type": feature_type, "n": str(count)}
    if name:
        row["name"] = name
    if available(detail):
        row["m"] = str(detail)
    return row


def product_line(series: str, name: str, product_type: str) -> str:
    if product_type == "wireless_soc":
        return series
    return name


def lifecycle(status: Any) -> str:
    mapping = {
        "mass production": "active",
        "sample": "sampling",
        "nrnd": "nrnd",
        "eol": "eol",
        "replaced": "replaced",
    }
    return mapping.get(str(status or "").lower(), "unknown")


def row_to_records(row: dict[str, Any], product_type: str, source_id: str, observed_at: str) -> tuple[dict[str, Any], dict[str, Any]]:
    mpn = str(row.get("mpn") or "").strip().upper()
    name = str(row.get("name") or mpn).strip().upper()
    series = str(row.get("series") or name).strip().upper()
    line = product_line(series, name, product_type)
    cpu = str(row.get("cpu") or "").strip()
    core = cpu_name(cpu)
    adc_units, adc_bits, adc_channels = parse_adc(row.get("adc"))
    dac_units, dac_bits = parse_dac(row.get("dac"))
    features: list[dict[str, str]] = []
    for item in (
        feature("ADC", adc_channels, "ADC channels", adc_bits),
        feature("ADCUnits", adc_units, "ADC converter units", adc_bits),
        feature("DAC", dac_units, "DAC converter units", dac_bits),
        feature("Touch", row.get("touch_sensor"), "Capacitive touch channels"),
        feature("TempSens", row.get("temperature_sensor"), "Temperature sensor"),
        feature("IOs", row.get("gpio"), "GPIO"),
        feature("UART", row.get("uart")),
        feature("SPI", row.get("spi")),
        feature("SDIOHost", row.get("sdio_host")),
        feature("SDIOSlave", row.get("sdio_slave")),
        feature("I2S", row.get("i2s")),
        feature("I2C", row.get("i2c")),
        feature("I3C", row.get("i3c")),
        feature("RMT", quantity_from_expression(row.get("rmt")), "Remote control transceiver", row.get("rmt")),
        feature("LEDPWM", quantity_from_expression(row.get("ledpwm")), "LED PWM controller", row.get("ledpwm")),
        feature("MCPWM", quantity_from_expression(row.get("mcpwm")), "Motor-control PWM", row.get("mcpwm")),
        feature("USBOTG", row.get("usb_otg")),
        feature("ETH", row.get("ethernet_mac"), "Ethernet MAC"),
        feature("CAN", row.get("can"), "TWAI controller"),
        feature("Camera", 1 if available(row.get("camera_interface")) else 0, str(row.get("camera_interface") or "")),
        feature("Hall", row.get("hall_sensor"), "Hall sensor"),
        feature("TOF", 1 if available(row.get("tof")) else 0, "Time-of-flight interface"),
        feature("JTAG", 1 if available(row.get("jtag")) else 0),
        feature("WiFi", 1 if available(row.get("wifi")) else 0, str(row.get("wifi") or "")),
        feature("WiFi6", 1 if available(row.get("wifi6")) else 0, str(row.get("wifi6") or "")),
        feature("Bluetooth", 1 if available(row.get("bluetooth")) else 0, str(row.get("bluetooth") or "")),
        feature("IEEE802154", 1 if available(row.get("thread_802_15_4")) else 0, "IEEE 802.15.4 / Thread / Zigbee"),
        feature("PSRAM", row.get("psram"), "Integrated/in-package PSRAM", "MB"),
        feature("RTC_RAM", row.get("rtc_sram"), "RTC SRAM", "KB"),
    ):
        if item:
            features.append(item)

    processor = {
        "Dcore": core,
        "Dclock": str(integer(row.get("freq")) * 1_000_000) if integer(row.get("freq")) != "" else "",
        "DcoreCount": str(integer(row.get("core"))) if integer(row.get("core")) != "" else "",
        "Darchitecture": architecture(cpu),
        "DsourceDescription": cpu,
    }
    documents = [{"title": "ESP Product Selector", "url": SELECTOR_URL}]
    device_name = mpn
    device_id = f"espressif::{slug(device_name)}"
    line_id = f"espressif::esp::{slug(series)}::{slug(line)}"
    device = {
        "device_id": device_id,
        "product_line_id": line_id,
        "manufacturer": "Espressif",
        "product_type": product_type,
        "architecture_class": architecture(cpu),
        "family": "ESP",
        "series": series,
        "product_line": line,
        "device_name": device_name,
        "generic_device_name": name,
        "manufacturer_variant_code": mpn[len(name):].lstrip("-") if mpn.startswith(name) else mpn,
        "processor_cores": json.dumps([processor], ensure_ascii=False, sort_keys=True),
        "max_clock_hz": integer(row.get("freq")) * 1_000_000 if integer(row.get("freq")) != "" else "",
        "flash_bytes": memory_bytes(row.get("flash"), 1024 * 1024),
        "ram_bytes": memory_bytes(row.get("sram"), 1024),
        "package_types": row.get("package", ""),
        "pin_counts": integer(row.get("pins")) if row.get("pins") is not None else "",
        "memory_regions_json": json.dumps(
            [
                {"id": "SRAM", "size": memory_bytes(row.get("sram"), 1024)},
                {"id": "ROM", "size": memory_bytes(row.get("rom"), 1024)},
                {"id": "FLASH", "size": memory_bytes(row.get("flash"), 1024 * 1024)},
                {"id": "PSRAM", "size": memory_bytes(row.get("psram"), 1024 * 1024)},
            ],
            ensure_ascii=False,
            sort_keys=True,
        ),
        "features_json": json.dumps(features, ensure_ascii=False, sort_keys=True),
        "documents_json": json.dumps(documents, ensure_ascii=False, sort_keys=True),
        "svd_files": "",
        "lifecycle": lifecycle(row.get("status")),
        "source_id": source_id,
        "source_url": SELECTOR_URL,
        "source_version": "live_api_snapshot",
        "observed_at": observed_at,
        "verification_status": "manufacturer_product_selector_api",
    }
    part = {
        "orderable_part_id": f"espressif::{slug(mpn)}",
        "device_id": device_id,
        "manufacturer": "Espressif",
        "family": "ESP",
        "series": series,
        "product_line": line,
        "device_name": device_name,
        "part_number": mpn,
        "manufacturer_suffix": device["manufacturer_variant_code"],
        "package_code": row.get("package", ""),
        "temperature_grade_code": "",
        "packing_code": "",
        "package_name": row.get("package", ""),
        "temperature_range": row.get("temperature_range", ""),
        "packing_form": "",
        "lifecycle": lifecycle(row.get("status")),
        "source_id": source_id,
        "source_url": SELECTOR_URL,
        "observed_at": observed_at,
        "verification_status": "manufacturer_product_selector_api",
        "decode_status": "official_mpn_record; suffix_not_inferred",
    }
    return device, part


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=root / "data" / "vendor-packs" / "espressif")
    parser.add_argument("--cache-dir", type=Path, default=root / "cache" / "espressif")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--timeout", type=float, default=45.0)
    args = parser.parse_args()

    fetcher = OfficialFetcher(
        cache_dir=args.cache_dir,
        allowed_hosts={"products.espressif.com"},
        refresh=args.refresh,
        timeout=args.timeout,
    )
    observed_at = utc_now()
    errors: list[dict[str, Any]] = []
    snapshots = []
    payloads: dict[str, dict[str, Any]] = {}
    for scope, product_type in (("chip", "wireless_soc"), ("module", "wireless_module")):
        try:
            snapshot = fetcher.fetch(
                f"{API_ROOT}/user/products/select?type={scope}&language=en",
                cache_name=f"{scope}-products-en.json",
                method="POST",
                body=json.dumps({"ids": "all"}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            snapshots.append(snapshot)
            payloads[scope] = json.loads(snapshot.payload.decode("utf-8"))
        except Exception as exc:
            errors.append({"scope": scope, "source_url": SELECTOR_URL, "item": "all", "error": f"{type(exc).__name__}: {exc}"})

    devices: dict[str, dict[str, Any]] = {}
    parts: dict[str, dict[str, Any]] = {}
    provenance: list[dict[str, Any]] = []
    for scope, payload in payloads.items():
        source_id = f"espressif-product-selector:{scope}:en"
        product_type = "wireless_soc" if scope == "chip" else "wireless_module"
        for index, raw in enumerate(payload.get("tableData", [])):
            mpn = str(raw.get("mpn") or "").strip().upper()
            if not mpn or not re.fullmatch(r"ESP[A-Z0-9-]+", mpn):
                errors.append({"scope": scope, "source_url": SELECTOR_URL, "item": mpn or f"row:{index}", "error": "missing_or_invalid_official_mpn"})
                continue
            try:
                device, part = row_to_records(raw, product_type, source_id, observed_at)
            except Exception as exc:
                errors.append({"scope": scope, "source_url": SELECTOR_URL, "item": mpn, "error": f"{type(exc).__name__}: {exc}"})
                continue
            if integer(raw.get("uart")) in ("", 0):
                errors.append(
                    {
                        "scope": scope,
                        "source_url": SELECTOR_URL,
                        "item": mpn,
                        "error": "official_record_missing_positive_uart_count",
                    }
                )
                continue
            if device["device_id"] in devices and devices[device["device_id"]] != device:
                errors.append({"scope": scope, "source_url": SELECTOR_URL, "item": mpn, "error": "conflicting_duplicate_mpn"})
                continue
            devices[device["device_id"]] = device
            parts[part["orderable_part_id"]] = part
            for field_name, source_value in raw.items():
                if source_value in (None, ""):
                    continue
                provenance.append(
                    {
                        "record_type": "device",
                        "record_id": device["device_id"],
                        "field_name": field_name,
                        "source_id": source_id,
                        "source_url": SELECTOR_URL,
                        "source_path": f"tableData[{index}].{field_name}",
                        "source_value_json": json.dumps(source_value, ensure_ascii=False),
                        "observed_at": observed_at,
                        "verification_status": "manufacturer_product_selector_api",
                    }
                )

    source_rows = [
        {
            "source_id": f"espressif-product-selector:{scope}:en",
            "source_type": "manufacturer_product_selector_api",
            "publisher": "Espressif Systems",
            "title": f"ESP Product Selector {scope} dataset",
            "url": SELECTOR_URL,
            "version": f"sha256:{next((item.sha256 for item in snapshots if item.path.name.startswith(scope)), '')}",
            "observed_at": observed_at,
            "verification_scope": "Official MPN, lifecycle, processor, memory, package, wireless and peripheral fields returned by the selector API.",
        }
        for scope in payloads
    ]
    device_rows = sorted(devices.values(), key=lambda row: (row["series"], row["product_line"], row["device_name"]))
    part_rows = sorted(parts.values(), key=lambda row: row["part_number"])
    write_csv(args.output_dir / "device-variants.csv", DEVICE_FIELDS, device_rows)
    write_csv(args.output_dir / "orderable-parts.csv", PART_FIELDS, part_rows)
    write_csv(args.output_dir / "sources.csv", SOURCE_FIELDS, source_rows)
    write_csv(args.output_dir / "import-errors.csv", ERROR_FIELDS, errors)
    write_csv(args.output_dir / "field-provenance.csv", FIELD_PROVENANCE_FIELDS, provenance)

    statuses = sorted({row["lifecycle"] for row in device_rows})
    series = sorted({row["series"] for row in device_rows})
    socs = sum(row["product_type"] == "wireless_soc" for row in device_rows)
    modules = sum(row["product_type"] == "wireless_module" for row in device_rows)
    report = {
        "status": "ok" if not errors else "partial",
        "manufacturer": "Espressif",
        "source": SELECTOR_URL,
        "official_api_scopes": sorted(payloads),
        "series": series,
        "series_count": len(series),
        "soc_variants": socs,
        "module_variants": modules,
        "device_variants": len(device_rows),
        "official_mpns": len(part_rows),
        "uart_coverage": {
            "records_with_positive_uart_count": sum(
                any(
                    item.get("type") == "UART" and number(item.get("n")) and number(item.get("n")) > 0
                    for item in json.loads(row["features_json"])
                )
                for row in device_rows
            ),
            "total_records": len(device_rows),
            "minimum_uart_count": min(
                (
                    int(number(item.get("n")) or 0)
                    for row in device_rows
                    for item in json.loads(row["features_json"])
                    if item.get("type") == "UART"
                ),
                default=0,
            ),
            "maximum_uart_count": max(
                (
                    int(number(item.get("n")) or 0)
                    for row in device_rows
                    for item in json.loads(row["features_json"])
                    if item.get("type") == "UART"
                ),
                default=0,
            ),
        },
        "lifecycle_states": statuses,
        "field_provenance_records": len(provenance),
        "import_errors": len(errors),
        "snapshot_sha256": {item.path.name: item.sha256 for item in snapshots},
        "completeness_claim": "All records returned by both official API scopes at observation time; not a claim about unpublished products.",
    }
    write_json(args.output_dir / "import-report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Enrich Microchip devices from the exact-device ATDF files in official DFPs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
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
PROVENANCE_FIELDS = [
    "record_type", "record_id", "field_name", "source_id", "source_url",
    "source_path", "source_value_json", "observed_at", "verification_status",
]
CONFLICT_FIELDS = [
    "device_id", "device_name", "field_name", "previous_value", "atdf_value",
    "resolution", "source_path",
]
UNMATCHED_FIELDS = ["device_id", "device_name", "reason", "candidate_paths"]
MANIFEST_FIELDS = ["source_id", "pack_url", "cache_path", "sha256", "bytes", "atdf_files"]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def local_name(value: str) -> str:
    return value.rsplit("}", 1)[-1]


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
    values = [item for item in value.split(";") if item]
    if addition not in values:
        values.append(addition)
    return ";".join(values)


def pack_url(source: dict[str, str]) -> str:
    parsed = urllib.parse.urlparse(source["url"])
    filename = PurePosixPath(parsed.path).name
    stem = filename[:-5] if filename.lower().endswith(".pdsc") else filename
    path = str(PurePosixPath(parsed.path).parent / f"{stem}.{source['version']}.pack")
    return urllib.parse.urlunparse(parsed._replace(path=path))


def download(
    source: dict[str, str], cache_dir: Path, *, refresh: bool, timeout: float, proxy: str
) -> dict[str, Any]:
    url = pack_url(source)
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != "packs.download.microchip.com":
        raise ValueError(f"not an approved Microchip HTTPS DFP URL: {url}")
    path = cache_dir / PurePosixPath(parsed.path).name
    if not path.exists() or refresh:
        handlers: list[Any] = []
        if proxy:
            handlers.append(urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
        opener = urllib.request.build_opener(*handlers)
        request = urllib.request.Request(url, headers={"User-Agent": "MCU-L-Catalog/0.5 (+official ATDF importer)"})
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                with opener.open(request, timeout=timeout) as response:
                    payload = response.read()
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
                break
            except OSError as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(0.75 * (2**attempt))
        else:
            assert last_error is not None
            raise last_error
    payload = path.read_bytes()
    return {
        "cmsis_source": source,
        "pack_url": url,
        "cache_path": path,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }


def one_text(parent: ET.Element, path: str, attribute: str) -> str:
    item = parent.find(path)
    return item.get(attribute, "") if item is not None else ""


def parse_int(value: str) -> int | None:
    if not value:
        return None
    try:
        return int(value, 0)
    except ValueError:
        try:
            return int(float(value))
        except ValueError:
            return None


def module_type(module_name: str, instance_name: str) -> str | None:
    module = module_name.upper().replace("-", "_")
    instance = instance_name.upper().replace("-", "_")
    if module in {"ADC", "AFEC"}:
        return "ADCUnits"
    if module == "DAC":
        return "DAC"
    if module in {"AC", "ACC", "CMP", "COMP"}:
        return "COMP"
    if module in {"OPAMP"}:
        return "OPAMP"
    if module in {"TC", "TCC", "TIMER", "RTT", "PIT", "PWM", "MCPWM", "CCT"}:
        return "Timer"
    if module in {"SERCOM", "FLEXCOM"}:
        return "Com"
    if module in {"UART"}:
        return "UART"
    if module in {"USART"}:
        return "USART"
    if module in {"SPI", "QSPI", "OSPI"}:
        return "SPI"
    if module in {"I2C", "TWI", "TWIHS"}:
        return "I2C"
    if module in {"I2S", "SSC"}:
        return "I2S"
    if module in {"CAN", "MCAN"}:
        return "CAN"
    if module in {"GMAC", "EMAC", "ETH"}:
        return "ETH"
    if module in {"SDHC", "HSMCI", "SDMMC"}:
        return "SDIO"
    if module in {"DMAC", "DMA", "XDMAC", "PDC"}:
        return "DMA"
    if module in {"TRNG", "RNG"}:
        return "RNG"
    if module in {"RTC"}:
        return "RTC"
    if module in {"WDT", "WDT0", "WDT1", "RSWDT"}:
        return "WDT"
    if module in {"PTC", "QTOUCH"}:
        return "Touch"
    if module in {"ISI", "CSI", "DCMI"}:
        return "Camera"
    if module in {"LCDC", "SLCDC", "LCD"}:
        return "GLCD"
    if module in {"EBI", "SMC", "SDRAMC", "HEFC"}:
        return "ExtBus"
    if module in {"AES", "ICM", "SHA", "PUKCC", "CRYPTO", "HSM"}:
        return "Crypto"
    if module in {"EIC", "AIC"}:
        return "ExtInt"
    if module in {"USB", "UOTGHS", "UDP", "UDPHS"}:
        # ATDF establishes the controller but not a universally comparable
        # host/device count, so retain it as a generic communication feature.
        return "ComOther"
    if module in {"PORT", "PIO"}:
        return None
    return module_name or instance_name or None


EXACT_COUNT_TYPES = (
    "Timer", "ADCUnits", "DAC", "IOs", "SPI", "I2C", "I2S", "USART",
    "UART", "CAN", "ETH", "SDIO", "DMA", "RNG", "RTC", "WDT", "COMP",
    "OPAMP", "Touch", "Camera", "GLCD", "ExtBus", "ExtInt", "Com",
)


def parse_atdf(payload: bytes, archive_path: str) -> dict[str, Any]:
    root = ET.fromstring(payload)
    device = root.find("./devices/device")
    if device is None or not device.get("name"):
        raise ValueError("ATDF has no devices/device name")
    device_name = device.get("name", "")
    variants = list(root.findall("./variants/variant"))
    pinouts = {item.get("name", ""): item for item in root.findall("./pinouts/pinout")}
    variant_pinouts = {item.get("pinout", "") for item in variants if item.get("pinout")}
    pin_counts = {
        len(list(pinouts[name])) for name in variant_pinouts if name in pinouts
    }
    gpio_pads: set[str] = set()
    package_pads: set[str] = set()
    for name in variant_pinouts:
        if name not in pinouts:
            continue
        for pin in pinouts[name]:
            pad = pin.get("pad", "")
            package_pads.add(pad)
            if re.fullmatch(r"P[A-Z]+\d+", pad, re.I):
                gpio_pads.add(pad)

    grouped: dict[str, set[str]] = defaultdict(set)
    grouped["IOs"] = set(gpio_pads)
    adc_pads: set[str] = set()
    peripherals = device.find("peripherals")
    if peripherals is not None:
        for module in peripherals:
            module_name = module.get("name", "")
            for instance in module:
                instance_name = instance.get("name", "")
                feature_type = module_type(module_name, instance_name)
                if feature_type:
                    grouped[feature_type].add(instance_name or module_name)
                if module_name.upper() not in {"ADC", "AFEC"}:
                    continue
                signals = instance.find("signals")
                if signals is not None:
                    for signal in signals:
                        group = signal.get("group", "").upper()
                        if group.startswith(("AIN", "AD", "VINP", "VINM")) and signal.get("pad", "") in package_pads:
                            adc_pads.add(signal.get("pad", ""))

    features: list[dict[str, str]] = []
    for feature_type in EXACT_COUNT_TYPES:
        instances = sorted(grouped.pop(feature_type, set()))
        if feature_type in {"SPI", "I2C", "USART", "UART"} and grouped.get("Com"):
            # SERCOM/FLEXCOM blocks can implement several serial protocols.
            # Dedicated protocol counts therefore remain unknown.
            continue
        count = len(gpio_pads) if feature_type == "IOs" else len(instances)
        features.append({
            "type": feature_type,
            "name": ", ".join(instances) if instances else "ATDF exact instance count",
            "count": str(count),
            "source_kind": "microchip_atdf",
        })
    features.append({
        "type": "ADCExternalPins", "name": "External package pads with ADC input signals",
        "count": str(len(adc_pads)), "source_kind": "microchip_atdf",
    })
    for feature_type, instances in sorted(grouped.items()):
        for instance in sorted(instances):
            features.append({
                "type": feature_type, "name": instance, "count": "1",
                "source_kind": "microchip_atdf",
            })

    flash = 0
    ram = 0
    regions: list[dict[str, str]] = []
    for segment in device.findall("./address-spaces/address-space/memory-segment"):
        segment_type = segment.get("type", "").lower()
        size = parse_int(segment.get("size", ""))
        name = segment.get("name", "")
        if size is None:
            continue
        name_upper = name.upper()
        external = segment.get("external", "").lower() == "true"
        selectable_flash = (
            segment_type == "flash"
            and not external
            and not any(
                token in name_upper
                for token in ("FUSE", "OTP", "USER", "CFG", "CAL", "BACKUP")
            )
        )
        selectable_ram = (
            segment_type == "ram"
            and not external
            and not any(
                token in name_upper
                for token in ("CACHE", "CMCC_", "TAGRAM", "VALIDRAM")
            )
        )
        if selectable_flash:
            flash += size
        if selectable_ram:
            ram += size
        if segment_type in {"flash", "ram"}:
            regions.append({key: value for key, value in {
                "name": name, "start": segment.get("start", ""),
                "size": segment.get("size", ""), "type": segment_type,
            }.items() if value})

    speeds = {parse_int(item.get("speedmax", "")) for item in variants}
    speeds.discard(None)
    packages = {item.get("package", "") for item in variants if item.get("package")}
    return {
        "device_name": device_name,
        "archive_path": archive_path,
        "features": features,
        "flash_bytes": str(flash) if flash else "",
        "ram_bytes": str(ram) if ram else "",
        "max_clock_hz": str(next(iter(speeds))) if len(speeds) == 1 else "",
        "package_types": ";".join(sorted(packages)) if len(packages) == 1 else "",
        "pin_counts": str(next(iter(pin_counts))) if len(pin_counts) == 1 else "",
        "memory_regions_json": json.dumps(regions, ensure_ascii=False, sort_keys=True),
        "variants": variants,
        "variant_pinout_count": len(variant_pinouts),
    }


def variant_ordercode(variant: ET.Element, device_name: str) -> str:
    values = [value for key, value in variant.attrib.items() if local_name(key) == "ordercode"]
    exact = [value for value in values if value.upper().startswith(device_name.upper())]
    return (exact[0] if exact else (values[-1] if values else "")).strip()


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--vendor-pack", type=Path, default=root / "data/vendor-packs/microchip")
    parser.add_argument("--cache-dir", type=Path, default=root / "cache/microchip/packs")
    parser.add_argument("--proxy", default="")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--timeout", type=float, default=90.0)
    args = parser.parse_args()
    vendor_pack = args.vendor_pack.resolve()
    cache_dir = args.cache_dir.resolve()
    observed_at = utc_now()

    devices = read_csv(vendor_pack / "device-variants.csv")
    source_rows = read_csv(vendor_pack / "sources.csv")
    cmsis_sources: dict[str, dict[str, str]] = {}
    for source in source_rows:
        if source.get("source_id", "").startswith("cmsis:Microchip."):
            cmsis_sources[source["source_id"]] = source
    downloads: list[dict[str, Any]] = []
    download_errors: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(
                download, source, cache_dir, refresh=args.refresh,
                timeout=args.timeout, proxy=args.proxy,
            ): source
            for source in cmsis_sources.values()
        }
        for future in as_completed(futures):
            source = futures[future]
            try:
                downloads.append(future.result())
            except Exception as exc:
                download_errors.append({"source_id": source["source_id"], "error": str(exc)})

    atdfs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    manifests: list[dict[str, Any]] = []
    atdf_parse_errors: list[dict[str, str]] = []
    atdf_sources: dict[str, dict[str, str]] = {}
    for result in downloads:
        cmsis = result["cmsis_source"]
        source_id = "microchip-atdf:" + cmsis["source_id"].removeprefix("cmsis:Microchip.")
        atdf_sources[source_id] = {
            "source_id": source_id, "source_type": "manufacturer_atdf_device_database",
            "publisher": "Microchip", "title": f"{cmsis['title']} exact-device ATDF files",
            "url": result["pack_url"], "version": cmsis["version"],
            "observed_at": observed_at,
            "verification_scope": (
                "Exact ATDF device name, memory segments, module instances, pinout, ADC signal pads, "
                "and manufacturer order codes contained in the official Microchip DFP."
            ),
        }
        count = 0
        try:
            with zipfile.ZipFile(result["cache_path"]) as archive:
                names = sorted(name for name in archive.namelist() if name.lower().endswith(".atdf"))
                for name in names:
                    try:
                        record = parse_atdf(archive.read(name), name)
                        record.update({
                            "source_id": source_id, "source_url": result["pack_url"],
                            "source_version": cmsis["version"],
                            "pack_sha256": result["sha256"],
                        })
                        atdfs[record["device_name"]].append(record)
                        count += 1
                    except Exception as exc:
                        atdf_parse_errors.append({"source_id": source_id, "path": name, "error": str(exc)})
        except (OSError, zipfile.BadZipFile) as exc:
            download_errors.append({"source_id": source_id, "error": str(exc)})
        manifests.append({
            "source_id": source_id, "pack_url": result["pack_url"],
            "cache_path": str(result["cache_path"]), "sha256": result["sha256"],
            "bytes": result["bytes"], "atdf_files": count,
        })

    provenance: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    unmatched: list[dict[str, str]] = []
    parts_by_id = {row["orderable_part_id"]: row for row in read_csv(vendor_pack / "orderable-parts.csv")}
    # Make reruns idempotent: ATDF parts are regenerated from the current packs.
    parts_by_id = {
        key: row for key, row in parts_by_id.items()
        if not row.get("source_id", "").startswith("microchip-atdf:")
    }
    field_updates: Counter[str] = Counter()

    def source_path(record: dict[str, Any]) -> str:
        return f"{record['source_url']}!/{record['archive_path']}"

    def add_provenance(row: dict[str, str], field: str, value: Any, record: dict[str, Any]) -> None:
        provenance.append({
            "record_type": "device", "record_id": row["device_id"], "field_name": field,
            "source_id": record["source_id"], "source_url": record["source_url"],
            "source_path": source_path(record),
            "source_value_json": json.dumps(value, ensure_ascii=False, sort_keys=True),
            "observed_at": observed_at, "verification_status": "manufacturer_device_database",
        })

    def update_scalar(row: dict[str, str], field: str, value: str, record: dict[str, Any]) -> None:
        if not value:
            return
        previous = row.get(field, "")
        if previous and previous != value:
            conflicts.append({
                "device_id": row["device_id"], "device_name": row["device_name"],
                "field_name": field, "previous_value": previous, "atdf_value": value,
                "resolution": "atdf_exact_device_preferred", "source_path": source_path(record),
            })
        if previous != value:
            row[field] = value
            field_updates[field] += 1
        add_provenance(row, field, value, record)

    for row in devices:
        candidates = atdfs.get(row["device_name"], [])
        if len(candidates) != 1:
            unmatched.append({
                "device_id": row["device_id"], "device_name": row["device_name"],
                "reason": "no_exact_atdf_match" if not candidates else "duplicate_exact_atdf_match",
                "candidate_paths": ";".join(source_path(item) for item in candidates),
            })
            continue
        record = candidates[0]
        for field in ("max_clock_hz", "flash_bytes", "ram_bytes", "package_types", "pin_counts", "memory_regions_json"):
            update_scalar(row, field, record[field], record)
        existing = [
            feature for feature in json.loads(row.get("features_json") or "[]")
            if feature.get("source_kind") != "microchip_atdf"
        ]
        row["features_json"] = json.dumps(existing + record["features"], ensure_ascii=False, sort_keys=True)
        field_updates["features_json"] += 1
        add_provenance(row, "features_json", record["features"], record)
        row["source_id"] = append_semicolon(row.get("source_id", ""), record["source_id"])
        row["source_version"] = append_semicolon(row.get("source_version", ""), record["source_version"])
        row["verification_status"] = "multi_source_manufacturer_device_database"

        for variant in record["variants"]:
            part_number = variant_ordercode(variant, row["device_name"])
            if not part_number:
                continue
            part_id = f"microchip::{part_number.lower()}"
            suffix = part_number[len(row["device_name"]):] if part_number.upper().startswith(row["device_name"].upper()) else ""
            temp_min = variant.get("tempmin", "")
            temp_max = variant.get("tempmax", "")
            parts_by_id[part_id] = {
                "orderable_part_id": part_id, "device_id": row["device_id"],
                "manufacturer": row["manufacturer"], "family": row["family"],
                "series": row["series"], "product_line": row["product_line"],
                "device_name": row["device_name"], "part_number": part_number,
                "manufacturer_suffix": suffix, "package_code": "",
                "temperature_grade_code": "", "packing_code": "",
                "package_name": variant.get("package", ""),
                "temperature_range": f"{temp_min}..{temp_max} °C" if temp_min and temp_max else "",
                "packing_form": "", "lifecycle": "unknown", "source_id": record["source_id"],
                "source_url": record["source_url"], "observed_at": observed_at,
                "verification_status": "manufacturer_device_database", "decode_status": "atdf_variant_record",
            }

    retained_sources = {
        row["source_id"]: row for row in source_rows
        if not row.get("source_id", "").startswith("microchip-atdf:")
    }
    retained_sources.update(atdf_sources)
    write_csv(vendor_pack / "device-variants.csv", DEVICE_FIELDS, devices)
    write_csv(vendor_pack / "orderable-parts.csv", PART_FIELDS, sorted(parts_by_id.values(), key=lambda row: row["part_number"]))
    write_csv(vendor_pack / "sources.csv", SOURCE_FIELDS, sorted(retained_sources.values(), key=lambda row: row["source_id"]))
    write_csv(vendor_pack / "atdf-field-provenance.csv", PROVENANCE_FIELDS, provenance)
    write_csv(vendor_pack / "atdf-conflicts.csv", CONFLICT_FIELDS, conflicts)
    write_csv(vendor_pack / "atdf-unmatched.csv", UNMATCHED_FIELDS, unmatched)
    write_csv(vendor_pack / "atdf-source-manifest.csv", MANIFEST_FIELDS, sorted(manifests, key=lambda row: row["source_id"]))
    report = {
        "status": "ok" if not download_errors else "partial",
        "input_devices": len(devices), "official_packs": len(cmsis_sources),
        "downloaded_or_cached_packs": len(downloads),
        "download_bytes": sum(item["bytes"] for item in downloads),
        "pack_download_errors": download_errors, "atdf_parse_errors": atdf_parse_errors,
        "atdf_device_names": len(atdfs), "matched_devices": len(devices) - len(unmatched),
        "unmatched_devices": len(unmatched), "field_updates": dict(sorted(field_updates.items())),
        "scalar_conflicts_resolved": len(conflicts), "provenance_records": len(provenance),
        "orderable_parts": sum(row.get("source_id", "").startswith("microchip-atdf:") for row in parts_by_id.values()),
        "observed_at": observed_at,
        "accuracy_policy": [
            "Only an ATDF device name exactly equal to the catalog device name is accepted.",
            "Memory and peripheral facts come from that exact ATDF, not from an SVD or sibling device.",
            "SERCOM/FLEXCOM is stored as a shared configurable serial resource, not multiplied into UART, SPI, and I2C counts.",
            "ADC external pin count is the intersection of ADC signal pads and the ATDF package pinout.",
            "Manufacturer order codes are copied only from ATDF variant elements.",
        ],
    }
    (vendor_pack / "atdf-import-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())

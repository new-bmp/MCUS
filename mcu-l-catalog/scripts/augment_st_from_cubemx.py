#!/usr/bin/env python3
"""Enrich the ST vendor pack from STM32CubeMX's per-device database.

This importer deliberately accepts only exact identifiers.  It does not copy
peripheral counts from family SVD files and it does not infer a small-package
part from a larger member of the same STM32 line.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SOURCE_URL = "https://www.st.com/en/development-tools/stm32cubemx.html"
DEVICE_FIELDS = [
    "device_id", "product_line_id", "manufacturer", "product_type",
    "architecture_class", "family", "series", "product_line", "device_name",
    "generic_device_name", "manufacturer_variant_code", "processor_cores",
    "max_clock_hz", "flash_bytes", "ram_bytes", "package_types", "pin_counts",
    "memory_regions_json", "features_json", "documents_json", "svd_files",
    "lifecycle", "source_id", "source_url", "source_version", "observed_at",
    "verification_status",
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
    "device_id", "device_name", "field_name", "previous_value", "cubemx_value",
    "resolution", "source_path",
]
UNMATCHED_FIELDS = ["device_id", "device_name", "reason", "candidate_refs"]
MANIFEST_FIELDS = ["source_path", "sha256", "matched_device_count"]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def children(element: ET.Element, name: str) -> list[ET.Element]:
    return [item for item in element if local_name(item.tag) == name]


def child_text(element: ET.Element, name: str) -> str:
    for item in element:
        if local_name(item.tag) == name:
            return (item.text or "").strip()
    return ""


def append_semicolon(value: str, addition: str) -> str:
    values = [item for item in value.split(";") if item]
    if addition not in values:
        values.append(addition)
    return ";".join(values)


def numeric_text(value: str) -> str:
    if not value:
        return ""
    return str(int(float(value)))


def scaled_text(value: str, scale: int) -> str:
    if not value:
        return ""
    return str(int(float(value) * scale))


def count_pinout(root: ET.Element) -> tuple[int, int]:
    pins = children(root, "Pin")
    adc_pins: set[str] = set()
    adc_input = re.compile(r"^ADC\d*_(?:IN\d+|IN[PN]\d+|VIN[MP]\d*)$", re.I)
    for pin in pins:
        if pin.get("Type", "").upper() != "I/O":
            continue
        if any(
            adc_input.match(signal.get("Name", ""))
            for signal in children(pin, "Signal")
        ):
            adc_pins.add(pin.get("Position") or pin.get("Name") or "")
    adc_pins.discard("")
    return len(pins), len(adc_pins)


SOFTWARE_IPS = {
    "ADV_TRACE", "COMMON_BLE", "COMMON_WPAN", "EXTMEM_LOADER",
    "EXTMEM_MANAGER", "EXTMEM_MANAGER_APPLI", "FATFS", "FILEX", "FREERTOS",
    "GUI_INTERFACE", "KMS", "LEVELX", "LIBJPEG", "LORAWAN", "LWIP",
    "MBEDTLS", "NETXDUO", "OPENAMP", "PDM2PCM", "RESMGR_UTILITY", "SIGFOX",
    "STM32_BLE", "STM32_WPAN", "SUBGHZ_PHY", "THREADX", "TINY_LPM",
    "TRACER_EMB", "TRUSTED-FIRMWARE-M", "USBX", "USB_DEVICE", "USB_HOST",
    "USB_DEVICE", "USB_HOST", "WMBUS",
}


def classify_ip(name: str, instance: str) -> str | None:
    """Map a concrete CubeMX hardware instance to MCU-L's stable taxonomy."""
    name_u = name.upper()
    instance_u = instance.upper()
    if name_u in SOFTWARE_IPS or instance_u in SOFTWARE_IPS:
        return None
    if re.fullmatch(r"ADC\d*", instance_u):
        return "ADCUnits"
    if re.fullmatch(r"DAC\d*", instance_u):
        return "DAC"
    if re.fullmatch(r"COMP\d*", instance_u):
        return "COMP"
    if re.fullmatch(r"OPAMP\d*", instance_u):
        return "OPAMP"
    if re.fullmatch(r"(?:TIM\d+|LPTIM\d+|HRTIM\d*|IRTIM\d*)", instance_u):
        return "Timer"
    if re.fullmatch(r"(?:SPI|HSPI)\d*", instance_u):
        return "SPI"
    if re.fullmatch(r"I2S\d*", instance_u):
        return "I2S"
    if re.fullmatch(r"(?:I2C|FMPI2C)\d*", instance_u):
        return "I2C"
    if re.fullmatch(r"I3C\d*", instance_u):
        return "I3C"
    if re.fullmatch(r"USART\d*", instance_u):
        return "USART"
    if re.fullmatch(r"(?:UART|LPUART)\d*", instance_u):
        return "UART"
    if re.fullmatch(r"(?:CAN|FDCAN)\d*", instance_u):
        return "CAN"
    if instance_u == "USB" or re.fullmatch(r"USB\d*", instance_u):
        return "USBD"
    if instance_u.startswith(("USB_OTG", "USB_DRD", "USB3DR")):
        return "USBOTG"
    if instance_u.startswith("USBH_"):
        return "USBH"
    if instance_u.startswith("ETH"):
        return "ETH"
    if instance_u.startswith(("SDIO", "SDMMC")):
        return "SDIO"
    if re.fullmatch(r"(?:DMA|BDMA|GPDMA|HPDMA|LPDMA|MDMA)\d*", instance_u):
        return "DMA"
    if re.fullmatch(r"RNG\d*", instance_u):
        return "RNG"
    if instance_u == "RTC":
        return "RTC"
    if instance_u in {"IWDG", "WWDG"}:
        return "WDT"
    if instance_u in {"TSC", "TOUCHSENSING"}:
        return "Touch"
    if instance_u.startswith(("DCMI", "DCMIPP", "CSI", "PSSI")):
        return "Camera"
    if instance_u.startswith(("LCD", "LTDC", "DSIHOST", "GFXMMU", "GPU")):
        return "GLCD"
    if instance_u.startswith(("FMC", "FSMC", "OCTOSPI", "QUADSPI", "XSPI")):
        return "ExtBus"
    if instance_u == "GPIO":
        return "IOs"
    if instance_u in {"AES", "CRYP", "SAES", "HASH", "PKA", "MCE", "OTFDEC"}:
        return "Crypto"
    return name or instance or None


EXACT_COUNT_TYPES = (
    "Timer", "ADCUnits", "DAC", "IOs", "SPI", "I2C", "I2S", "I3C",
    "USART", "UART", "CAN", "USBD", "USBH", "USBOTG", "ETH", "SDIO",
    "DMA", "RNG", "RTC", "WDT", "COMP", "OPAMP", "Touch", "Camera",
    "GLCD", "ExtBus",
)


def extract_features(root: ET.Element, io_count: int, adc_pin_count: int) -> list[dict[str, str]]:
    grouped: dict[str, set[str]] = defaultdict(set)
    for ip in children(root, "IP"):
        name = ip.get("Name", "").strip()
        instance = ip.get("InstanceName", "").strip()
        feature_type = classify_ip(name, instance)
        if feature_type:
            grouped[feature_type].add(instance or name)
    grouped["IOs"] = {f"{io_count} selectable I/O pins"} if io_count >= 0 else set()
    features: list[dict[str, str]] = []
    for feature_type in EXACT_COUNT_TYPES:
        instances = sorted(grouped.pop(feature_type, set()))
        count = io_count if feature_type == "IOs" else len(instances)
        features.append({
            "type": feature_type,
            "name": ", ".join(instances) if instances else "CubeMX exact instance count",
            "count": str(count),
            "source_kind": "cubemx_device_db",
        })
    features.append({
        "type": "ADCExternalPins",
        "name": "External package pins with ADC input signals",
        "count": str(adc_pin_count),
        "source_kind": "cubemx_device_db",
    })
    for feature_type, instances in sorted(grouped.items()):
        for instance in sorted(instances):
            features.append({
                "type": feature_type,
                "name": instance,
                "count": "1",
                "source_kind": "cubemx_device_db",
            })
    return features


def entry_signature(entry: ET.Element) -> tuple[Any, ...]:
    return (
        child_text(entry, "Frequency"),
        tuple((item.text or "").strip() for item in children(entry, "Flash")),
        child_text(entry, "Ram"),
        entry.get("PackageName", ""),
        child_text(entry, "IONb"),
        entry.get("Name", ""),
    )


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    default_db = Path.home() / "AppData/Local/Programs/STM32CubeMX/db"
    parser = argparse.ArgumentParser()
    parser.add_argument("--cubemx-db", type=Path, default=default_db)
    parser.add_argument("--vendor-pack", type=Path, default=root / "data/vendor-packs/st")
    args = parser.parse_args()

    db = args.cubemx_db.resolve()
    vendor_pack = args.vendor_pack.resolve()
    families_path = db / "mcu/families.xml"
    package_path = db / "package.xml"
    device_path = vendor_pack / "device-variants.csv"
    for required in (families_path, package_path, device_path):
        if not required.exists():
            parser.error(f"required input does not exist: {required}")

    package_root = ET.parse(package_path).getroot()
    version = ""
    for item in package_root.iter():
        if item.get("Release"):
            version = item.get("Release", "")
            break
    if not version:
        parser.error("CubeMX database release was not found in package.xml")
    source_id = f"st-cubemx-device-db:{version}"
    observed_at = utc_now()

    families_root = ET.parse(families_path).getroot()
    entries = [item for item in families_root.iter() if local_name(item.tag) == "Mcu"]
    indices: dict[str, dict[str, list[ET.Element]]] = {
        key: defaultdict(list) for key in ("RefName", "RPN", "Name")
    }
    for entry in entries:
        for key, index in indices.items():
            if entry.get(key):
                index[entry.get(key, "")].append(entry)

    xml_by_stem = {
        path.stem: path for path in (db / "mcu").glob("*.xml")
        if path.name.lower() != "families.xml"
    }
    devices = read_csv(device_path)
    provenance: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    unmatched: list[dict[str, Any]] = []
    matched_files: Counter[Path] = Counter()
    match_rules: Counter[str] = Counter()
    field_updates: Counter[str] = Counter()

    def add_provenance(row: dict[str, str], field: str, value: Any, source_path: Path) -> None:
        provenance.append({
            "record_type": "device",
            "record_id": row["device_id"],
            "field_name": field,
            "source_id": source_id,
            "source_url": SOURCE_URL,
            "source_path": str(source_path.resolve()),
            "source_value_json": json.dumps(value, ensure_ascii=False, sort_keys=True),
            "observed_at": observed_at,
            "verification_status": "manufacturer_device_database",
        })

    def update_scalar(row: dict[str, str], field: str, value: str, source_path: Path) -> None:
        if not value:
            return
        previous = row.get(field, "")
        if previous and previous != value:
            conflicts.append({
                "device_id": row["device_id"],
                "device_name": row["device_name"],
                "field_name": field,
                "previous_value": previous,
                "cubemx_value": value,
                "resolution": "cubemx_official_device_db_preferred",
                "source_path": str(source_path.resolve()),
            })
        if previous != value:
            row[field] = value
            field_updates[field] += 1
        add_provenance(row, field, value, source_path)

    for row in devices:
        name = row["device_name"]
        candidates: list[ET.Element] = []
        rule = ""
        for key in ("RefName", "RPN", "Name"):
            if indices[key].get(name):
                candidates = indices[key][name]
                rule = f"exact_{key.lower()}"
                break
        if not candidates and name.endswith("TR") and indices["RefName"].get(name[:-2]):
            candidates = indices["RefName"][name[:-2]]
            rule = "packing_suffix_removed:TR"
        if not candidates:
            unmatched.append({
                "device_id": row["device_id"], "device_name": name,
                "reason": "no_exact_identifier_match", "candidate_refs": "",
            })
            continue
        if len(candidates) != 1:
            unmatched.append({
                "device_id": row["device_id"], "device_name": name,
                "reason": "ambiguous_exact_identifier",
                "candidate_refs": ";".join(sorted({item.get("RefName", "") for item in candidates})),
            })
            conflicts.append({
                "device_id": row["device_id"], "device_name": name,
                "field_name": "cubemx_device_match",
                "previous_value": name,
                "cubemx_value": json.dumps([entry_signature(item) for item in candidates]),
                "resolution": "not_applied_ambiguous_package_or_variant",
                "source_path": str(families_path),
            })
            continue

        entry = candidates[0]
        mcu_xml = xml_by_stem.get(entry.get("Name", ""))
        if not mcu_xml:
            unmatched.append({
                "device_id": row["device_id"], "device_name": name,
                "reason": "matched_families_entry_missing_device_xml",
                "candidate_refs": entry.get("RefName", ""),
            })
            continue
        try:
            mcu_root = ET.parse(mcu_xml).getroot()
        except ET.ParseError as exc:
            unmatched.append({
                "device_id": row["device_id"], "device_name": name,
                "reason": f"device_xml_parse_error:{exc}",
                "candidate_refs": entry.get("RefName", ""),
            })
            continue

        match_rules[rule] += 1
        matched_files[mcu_xml] += 1
        pin_count, adc_pin_count = count_pinout(mcu_root)
        io_count = int(float(child_text(entry, "IONb") or 0))
        flash_values = [numeric_text((item.text or "").strip()) for item in children(entry, "Flash")]
        flash_values = [value for value in flash_values if value]
        flash_bytes = str(int(flash_values[0]) * 1024) if len(flash_values) == 1 else ""

        update_scalar(row, "max_clock_hz", scaled_text(child_text(entry, "Frequency"), 1_000_000), families_path)
        update_scalar(row, "flash_bytes", flash_bytes, families_path)
        update_scalar(row, "ram_bytes", scaled_text(child_text(entry, "Ram"), 1024), families_path)
        update_scalar(row, "package_types", entry.get("PackageName", ""), families_path)
        update_scalar(row, "pin_counts", str(pin_count), mcu_xml)

        existing_features = json.loads(row.get("features_json") or "[]")
        existing_features = [
            feature for feature in existing_features
            if feature.get("source_kind") != "cubemx_device_db"
        ]
        cubemx_features = extract_features(mcu_root, io_count, adc_pin_count)
        row["features_json"] = json.dumps(existing_features + cubemx_features, ensure_ascii=False, sort_keys=True)
        field_updates["features_json"] += 1
        add_provenance(
            row,
            "features_json",
            {
                "ip_instances": len(children(mcu_root, "IP")),
                "adc_external_pin_count": adc_pin_count,
                "features": cubemx_features,
            },
            mcu_xml,
        )
        row["source_id"] = append_semicolon(row.get("source_id", ""), source_id)
        row["source_version"] = append_semicolon(row.get("source_version", ""), version)
        row["verification_status"] = "multi_source_manufacturer_device_database"

    sources_path = vendor_pack / "sources.csv"
    sources = read_csv(sources_path)
    sources = [row for row in sources if row.get("source_id") != source_id]
    sources.append({
        "source_id": source_id,
        "source_type": "manufacturer_device_database",
        "publisher": "STMicroelectronics",
        "title": "STM32CubeMX MCU device database",
        "url": SOURCE_URL,
        "version": version,
        "observed_at": observed_at,
        "verification_scope": (
            "Exact CubeMX RefName/RPN/Name matches only; scalar selector fields, concrete IP "
            "instances, package pinout and ADC-capable external pins. No family-SVD inference."
        ),
    })

    write_csv(device_path, DEVICE_FIELDS, devices)
    write_csv(sources_path, SOURCE_FIELDS, sorted(sources, key=lambda row: row["source_id"]))
    write_csv(vendor_pack / "cubemx-field-provenance.csv", PROVENANCE_FIELDS, provenance)
    write_csv(vendor_pack / "cubemx-conflicts.csv", CONFLICT_FIELDS, conflicts)
    write_csv(vendor_pack / "cubemx-unmatched.csv", UNMATCHED_FIELDS, unmatched)
    manifest_rows = [
        {"source_path": str(families_path), "sha256": sha256(families_path), "matched_device_count": sum(match_rules.values())},
        {"source_path": str(package_path), "sha256": sha256(package_path), "matched_device_count": sum(match_rules.values())},
    ]
    manifest_rows.extend({
        "source_path": str(path.resolve()), "sha256": sha256(path),
        "matched_device_count": count,
    } for path, count in sorted(matched_files.items(), key=lambda item: str(item[0])))
    write_csv(vendor_pack / "cubemx-source-manifest.csv", MANIFEST_FIELDS, manifest_rows)
    manifest_digest = hashlib.sha256(
        "".join(f"{row['source_path']}\0{row['sha256']}\n" for row in manifest_rows).encode("utf-8")
    ).hexdigest()
    report = {
        "source_id": source_id,
        "source_version": version,
        "source_url": SOURCE_URL,
        "cubemx_db": str(db),
        "families_entries": len(entries),
        "input_devices": len(devices),
        "matched_devices": sum(match_rules.values()),
        "unmatched_devices": len(unmatched),
        "match_rules": dict(sorted(match_rules.items())),
        "field_updates": dict(sorted(field_updates.items())),
        "resolved_scalar_conflicts": sum(item["resolution"].endswith("preferred") for item in conflicts),
        "ambiguous_matches_not_applied": sum(item["resolution"].startswith("not_applied") for item in conflicts),
        "provenance_records": len(provenance),
        "source_files_used": len(manifest_rows),
        "source_manifest_sha256": manifest_digest,
        "observed_at": observed_at,
        "accuracy_policy": [
            "Only exact RefName, RPN, Name, or an explicitly recorded terminal TR packing suffix removal is accepted.",
            "Ambiguous identifier matches are not enriched.",
            "Peripheral counts come from the matched per-device MCU XML, never a family SVD.",
            "ADC converter units and external ADC-capable package pins are stored as separate facts.",
            "A blank remains unknown when the exact official database does not establish the value.",
        ],
    }
    (vendor_pack / "cubemx-import-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

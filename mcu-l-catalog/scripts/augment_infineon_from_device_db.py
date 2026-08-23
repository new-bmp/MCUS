#!/usr/bin/env python3
"""Enrich Infineon devices from the official ModusToolbox device-db.

The repository is Infineon's device database used by ModusToolbox.  Every
accepted record is a leaf part directory with info/base/studio views.  Exact
part names enrich existing CMSIS rows; MCU records absent from CMSIS are added
without inventing suffix combinations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import tarfile
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


COMMIT = "1e7589fe5ce5c25b68a5d8866c22c142df9b7555"
ARCHIVE_URL = f"https://codeload.github.com/Infineon/device-db/tar.gz/{COMMIT}"
SOURCE_URL = f"https://github.com/Infineon/device-db/tree/{COMMIT}"
SOURCE_ID_PREFIX = "infineon-device-db:"
SOURCE_ID = f"{SOURCE_ID_PREFIX}{COMMIT[:12]}"

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
    "device_id", "device_name", "field_name", "previous_value", "device_db_value",
    "resolution", "source_path",
]
UNMATCHED_FIELDS = ["device_id", "device_name", "reason"]
ADDED_FIELDS = [
    "device_id", "device_name", "domain", "database_family", "family", "series",
    "product_line", "source_path",
]
MANIFEST_FIELDS = ["source_id", "archive_url", "cache_path", "sha256", "bytes", "version", "commit"]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


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
    if addition and addition not in values:
        values.append(addition)
    return ";".join(values)


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def download_archive(
    path: Path, *, refresh: bool, proxy: str, timeout: float, retries: int
) -> None:
    if path.exists() and not refresh:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    handlers: list[Any] = []
    if proxy:
        handlers.append(urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
    opener = urllib.request.build_opener(*handlers)
    last_error: Exception | None = None
    for attempt in range(retries):
        start = temporary.stat().st_size if temporary.exists() else 0
        headers = {"User-Agent": "MCU-L-Catalog/0.5 (+official Infineon device-db importer)"}
        if start:
            headers["Range"] = f"bytes={start}-"
        request = urllib.request.Request(ARCHIVE_URL, headers=headers)
        try:
            with opener.open(request, timeout=timeout) as response:
                append = start > 0 and getattr(response, "status", None) == 206
                with temporary.open("ab" if append else "wb") as handle:
                    while True:
                        block = response.read(1024 * 1024)
                        if not block:
                            break
                        handle.write(block)
            with tarfile.open(temporary, "r:gz"):
                pass
            temporary.replace(path)
            return
        except (OSError, urllib.error.HTTPError, urllib.error.URLError, tarfile.TarError) as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(1.0 * (2**attempt))
    assert last_error is not None
    raise last_error


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def child_values(root: ET.Element) -> dict[str, str]:
    return {local_name(item.tag): (item.text or "").strip() for item in root}


def parse_count(value: str) -> int | None:
    if value in (None, "", "NA", "N", "None"):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric < 0 or not numeric.is_integer():
        return None
    return int(numeric)


def first_count(parameters: dict[str, str], names: Iterable[str]) -> tuple[int | None, str]:
    for name in names:
        value = parse_count(parameters.get(name, ""))
        if value is not None:
            return value, name
    return None, ""


def sum_counts(parameters: dict[str, str], names: Iterable[str]) -> tuple[int | None, str]:
    values: list[int] = []
    used: list[str] = []
    for name in names:
        value = parse_count(parameters.get(name, ""))
        if value is not None:
            values.append(value)
            used.append(name)
    return (sum(values), "+".join(used)) if values else (None, "")


def yes(value: str) -> bool:
    return value.strip().upper() in {"Y", "YES", "TRUE", "1", "PRESENT", "SUPPORTED"}


def package_pin_count(package: str) -> str:
    values = [int(item) for item in re.findall(r"(?<!\d)(\d{2,4})(?!\d)", package)]
    return str(values[0]) if values else ""


def normalized_core(value: str) -> str:
    compact = re.sub(r"[^A-Za-z0-9+]", "", value).upper().removeprefix("ARM")
    mapping = {
        "CORTEXM0": "Cortex-M0", "CM0": "Cortex-M0",
        "CORTEXM0P": "Cortex-M0+", "CM0+": "Cortex-M0+", "CM0P": "Cortex-M0+",
        "CORTEXM3": "Cortex-M3", "CM3": "Cortex-M3",
        "CORTEXM4": "Cortex-M4", "CM4": "Cortex-M4",
        "CORTEXM7": "Cortex-M7", "CM7": "Cortex-M7",
        "CORTEXM33": "Cortex-M33", "CM33": "Cortex-M33",
        "CORTEXR4": "Cortex-R4", "CR4": "Cortex-R4",
    }
    return mapping.get(compact, value.strip())


def processor_cores(base: dict[str, str], parameters: dict[str, str]) -> list[dict[str, str]]:
    raw = base.get("mcu", "")
    values = [item for item in raw.split(",") if item and item.lower() not in {"none", "na"}]
    if not values and parameters.get("CPU", "") not in {"", "None", "NA"}:
        values = [parameters["CPU"]]
    clock = str((parse_count(parameters.get("Speed_MHz", "")) or 0) * 1_000_000)
    result = []
    for index, value in enumerate(values):
        core = normalized_core(value)
        result.append({
            "Pname": f"Core{index}", "Dcore": core,
            "Dclock": clock if clock != "0" else "",
        })
    return result


def add_feature(
    features: list[dict[str, str]], feature_type: str, name: str,
    count: int | None = None, *, value: str = "", parameter: str = "",
) -> None:
    if count is None and not parameter:
        return
    # An explicit zero in an exact-device parameter is authoritative.  Keep it
    # so the enrichment stage does not fall back to a family/SVD superset.
    if count is not None and count < 0:
        return
    feature = {
        "type": feature_type, "name": name,
        "source_kind": "infineon_device_db",
    }
    if count is not None:
        feature["count"] = str(count)
    if value:
        feature["m"] = value
    if parameter:
        feature["source_parameter"] = parameter
    features.append(feature)


def extract_features(parameters: dict[str, str], capabilities: set[str]) -> list[dict[str, str]]:
    features: list[dict[str, str]] = []

    timer_count, timer_source = first_count(parameters, ["TimerCounterPWM_Count"])
    if timer_count is None:
        timer_count, timer_source = sum_counts(parameters, ["16Bit_TimerPWM_Count", "32Bit_TimerPWM_Count"])
    if timer_count is None:
        timer_count, timer_source = sum_counts(parameters, ["CCU4_Count", "CCU8_Count"])
    add_feature(features, "Timer", "Timer/counter/PWM blocks", timer_count, parameter=timer_source)

    adc_units, adc_source = first_count(parameters, ["SAR12_Count", "ADC_Count"])
    add_feature(features, "ADCUnits", "ADC converter units", adc_units, parameter=adc_source)
    adc_channels, adc_channel_source = first_count(parameters, ["ADCCH_Count", "ADC_CH_Count"])
    add_feature(
        features, "ADC", "ADC source channel count; not external ADC package pins",
        adc_channels, parameter=adc_channel_source,
    )
    dac_count, dac_source = first_count(parameters, ["DAC_Count"])
    if dac_count is None:
        dac_count, dac_source = sum_counts(parameters, ["DAC8_Count", "DAC12_Count", "AudioDAC_Count"])
    add_feature(features, "DAC", "DAC converter blocks", dac_count, parameter=dac_source)

    gpio_count, gpio_source = first_count(parameters, ["IO_Count"])
    add_feature(features, "IOs", "Selectable general-purpose I/O", gpio_count, parameter=gpio_source)

    serial_count, serial_source = first_count(parameters, ["SCB_Count", "USIC_Count"])
    add_feature(
        features, "Com", "Configurable serial communication blocks (SCB/USIC)",
        serial_count, parameter=serial_source,
    )
    for capability, feature_type, label in (
        ("uart", "UART", "UART supported by configurable serial blocks; simultaneous block count unknown"),
        ("i2c", "I2C", "I2C supported by configurable serial blocks; simultaneous block count unknown"),
        ("spi", "SPI", "SPI supported by configurable serial blocks; simultaneous block count unknown"),
    ):
        if capability in capabilities:
            add_feature(
                features, feature_type, label, None,
                parameter=f"DeviceCapabilities:{capability}",
            )
    for feature_type, label, names in (
        ("UART", "Dedicated UART-capable SCB channels", ["SCB_uart_Count"]),
        ("I2C", "Dedicated I2C-capable SCB channels", ["SCB_I2C_Count"]),
        ("SPI", "Dedicated SPI-capable SCB channels", ["SCB_SPI_Count"]),
        ("I2S", "I2S blocks", ["I2S_Count", "I2S_tx_Count", "I2S_rx_Count"]),
        ("I3C", "I3C blocks", ["I3C_Count"]),
        ("LIN", "LIN channels", ["LIN_CH_Count", "LIN_Count"]),
        ("ETH", "Ethernet controllers", ["ETH_Count"]),
        ("SDIO", "SDHC/SDMMC controllers", ["SDHC_COUNT", "SDMMC_Count"]),
        ("USBPD", "USB Power Delivery blocks", ["USBPD_Count"]),
    ):
        count, source = first_count(parameters, names)
        add_feature(features, feature_type, label, count, parameter=source)

    can_count, can_source = sum_counts(parameters, ["CAN20B_Count", "CANFD_CH_Count"])
    if can_count is None:
        can_count, can_source = first_count(parameters, ["CAN_Count"])
    add_feature(features, "CAN", "CAN/CAN FD channels", can_count, parameter=can_source)
    if yes(parameters.get("CAN_FD", "")):
        add_feature(features, "CANFD", "CAN FD supported", 1, parameter="CAN_FD")

    comparator_count, comparator_source = sum_counts(parameters, ["LPComp_Count", "Comp_Count", "ACMP_Count"])
    add_feature(features, "COMP", "Analog comparators", comparator_count, parameter=comparator_source)
    opamp_count, opamp_source = first_count(parameters, ["OpAmp_Count"])
    add_feature(features, "OPAMP", "Operational amplifiers", opamp_count, parameter=opamp_source)

    dma_channels, dma_source = sum_counts(parameters, ["DMAchannels_count", "DMACchannels_count", "AXI_DMACchannels_count"])
    add_feature(
        features, "DMAChannels", "DMA channels; not DMA controller count",
        dma_channels, parameter=dma_source,
    )

    if "USB_Type" in parameters:
        usb_type = parameters.get("USB_Type", "")
        add_feature(
            features, "USB", f"USB {usb_type}" if usb_type not in {"", "NA", "N"} else "No USB controller",
            0 if usb_type in {"", "NA", "N"} else 1, parameter="USB_Type",
        )
        if usb_type in {"", "NA", "N"}:
            add_feature(features, "USBD", "No USB device controller", 0, parameter="USB_Type")
            add_feature(features, "USBH", "No USB host controller", 0, parameter="USB_Type")
    for cap, feature_type, name in (
        ("usb_device", "USBD", "USB device capability"),
        ("usb_host", "USBH", "USB host capability"),
        ("rtc", "RTC", "Real-time clock"),
        ("rng", "RNG", "Hardware random number generator"),
        ("npu", "NPU", "Neural processing unit"),
        ("jpeg", "JPEG", "JPEG codec"),
    ):
        if cap in capabilities:
            add_feature(features, feature_type, name, 1, parameter=f"DeviceCapabilities:{cap}")

    usb_count, usb_source = first_count(parameters, ["USB_Count"])
    add_feature(features, "USB", "USB controllers", usb_count, parameter=usb_source)
    ext_bus_count, ext_bus_source = sum_counts(parameters, ["SMIF_Count", "EBU_Count"])
    add_feature(features, "ExtBus", "External memory/bus interfaces", ext_bus_count, parameter=ext_bus_source)
    ecat_count, ecat_source = first_count(parameters, ["ECAT_Count"])
    add_feature(features, "ETH", "EtherCAT controllers", ecat_count, parameter=ecat_source)
    camera_count, camera_source = first_count(parameters, ["Video-In_Count"])
    add_feature(features, "Camera", "Video input interfaces", camera_count, parameter=camera_source)
    display_count, display_source = first_count(parameters, ["Video-Out_Count"])
    add_feature(features, "GLCD", "Video output interfaces", display_count, parameter=display_source)

    if "Crypto" in parameters:
        add_feature(
            features, "Crypto", "Hardware cryptography" if yes(parameters["Crypto"]) else "No hardware cryptography",
            1 if yes(parameters["Crypto"]) else 0, parameter="Crypto",
        )
    security = parameters.get("Security", "")
    if security not in {"", "NA", "N", "None"}:
        add_feature(features, "Security", f"Infineon security subsystem: {security}", 1, parameter="Security")
    if "CapSense" in parameters:
        add_feature(
            features, "Touch", "Infineon CAPSENSE" if yes(parameters["CapSense"]) else "No CAPSENSE",
            1 if yes(parameters["CapSense"]) else 0, parameter="CapSense",
        )
    elif "capsense" in capabilities:
        add_feature(features, "Touch", "Infineon CAPSENSE", 1, parameter="DeviceCapabilities:capsense")
    if "IndSense" in parameters or "inductive" in capabilities:
        sensor_count = parse_count(parameters.get("IndSense_Sensors", ""))
        present = yes(parameters.get("IndSense", "")) or "inductive" in capabilities
        add_feature(
            features, "Touch", "Infineon inductive sensing" if present else "No inductive sensing",
            (sensor_count or 1) if present else 0, parameter="IndSense",
        )
    if "DirDriveLCD" in parameters:
        add_feature(
            features, "GLCD", "Direct-drive LCD" if yes(parameters["DirDriveLCD"]) else "No direct-drive LCD",
            1 if yes(parameters["DirDriveLCD"]) else 0, parameter="DirDriveLCD",
        )
    if "SMIF" in parameters:
        add_feature(
            features, "ExtBus", "Serial memory interface (SMIF)" if yes(parameters["SMIF"]) else "No SMIF",
            1 if yes(parameters["SMIF"]) else 0, parameter="SMIF",
        )

    for param, feature_type, label in (
        ("MATH_Count", "Math", "Hardware math units"),
        ("CORDIC", "CORDIC", "CORDIC accelerator"),
        ("JPEG_Count", "JPEG", "JPEG codec blocks"),
        ("CXPI_Count", "CXPI", "CXPI channels"),
        ("MIPI_Count", "MIPI", "MIPI interfaces"),
        ("UDB_Count", "UDB", "Universal Digital Blocks"),
    ):
        value = parameters.get(param, "")
        count = parse_count(value)
        if count is not None:
            add_feature(features, feature_type, label, count, parameter=param)
        elif yes(value):
            add_feature(features, feature_type, label, 1, parameter=param)
    if parameters.get("FLEXRAY", "") not in {"", "NA", "N", "0"}:
        add_feature(features, "FlexRay", "FlexRay supported", 1, parameter="FLEXRAY")
    bluetooth = parameters.get("Bluetooth", "")
    if bluetooth not in {"", "NA", "N", "0"}:
        add_feature(features, "Bluetooth", f"Bluetooth {bluetooth}", 1, parameter="Bluetooth")

    known_caps = {
        "adc", "capsense", "comp", "dma", "i2c", "i2s", "i3c", "lin", "spi", "uart",
        "can", "ethernet", "rtc", "rng", "usb_device", "usb_host", "sdhc", "sdio",
        "qspi", "std_crypto", "npu", "jpeg", "opamp", "lptimer", "low_power",
        "mcu_gp", "multi_core", "hal", "smart_io", "flash", "sram",
    }
    for capability in sorted(capabilities):
        if capability in known_caps or capability.startswith(("flash_", "sram_", "cat", "psoc", "xmc")):
            continue
        add_feature(
            features, "VendorCapability", f"Infineon capability: {capability}", 1,
            parameter=f"DeviceCapabilities:{capability}",
        )
    return features


def parse_database(archive: Path) -> tuple[str, list[dict[str, Any]], list[dict[str, str]]]:
    records: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    with tarfile.open(archive, "r:gz") as bundle:
        members = {item.name: item for item in bundle.getmembers() if item.isfile()}
        version_member = next((item for name, item in members.items() if name.endswith("/version.dat")), None)
        version = bundle.extractfile(version_member).read().decode("utf-8").strip() if version_member else COMMIT[:12]
        for path, member in members.items():
            if not path.endswith("/studio_5.0/view.xml"):
                continue
            pieces = path.split("/")
            if len(pieces) != 7 or pieces[1] != "devices":
                continue
            try:
                view = ET.fromstring(bundle.extractfile(member).read())
                parameters = {
                    item.get("name", ""): item.get("value", "")
                    for item in view.iter() if local_name(item.tag) == "Param"
                }
                part_number = parameters.get("PartNumber", "")
                types = {
                    item.get("name", "") for item in view.iter()
                    if local_name(item.tag) == "type"
                }
                if not part_number or "mcu" not in types:
                    continue
                capabilities = {
                    item.get("name", "") for item in view.iter()
                    if local_name(item.tag) == "DevCap" and item.get("name")
                }
                base_path = "/".join(pieces[:5] + ["base", "view.xml"])
                if base_path not in members:
                    raise ValueError("missing exact part base/view.xml")
                base = child_values(ET.fromstring(bundle.extractfile(members[base_path]).read()))
                records.append({
                    "part_number": part_number, "domain": pieces[2], "database_family": pieces[3],
                    "source_path": path, "base_path": base_path, "parameters": parameters,
                    "capabilities": capabilities, "base": base,
                    "features": extract_features(parameters, capabilities),
                    "cores": processor_cores(base, parameters),
                })
            except Exception as exc:
                errors.append({"path": path, "error": str(exc)})
    return version, records, errors


def common_prefix_length(left: str, right: str) -> int:
    count = 0
    for a, b in zip(left.upper(), right.upper()):
        if a != b:
            break
        count += 1
    return count


def generated_hierarchy(record: dict[str, Any]) -> tuple[str, str, str]:
    parameters = record["parameters"]
    database_family = record["database_family"]
    device_series = parameters.get("DeviceSeries", "") or parameters.get("DeviceDie", "") or database_family
    if record["domain"] == "Connectivity-BT-WIFI" or database_family.startswith(("CYW", "20", "30", "43", "55")):
        return "AIROC Connectivity", device_series, device_series
    if database_family == "FX3G2":
        return "EZ-USB", device_series, device_series
    if database_family.startswith("TVII"):
        return "TRAVEO T2G", device_series, device_series
    if database_family.startswith("PSC3") or database_family == "BOY2":
        return "PSC3", "PSC3", device_series
    return device_series, device_series, device_series


def is_orderable_part(part_number: str) -> bool:
    return bool(
        part_number
        and "x" not in part_number
        and not part_number.lower().endswith("-device")
        and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", part_number)
    )


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--vendor-pack", type=Path, default=root / "data/vendor-packs/infineon")
    parser.add_argument("--cache-dir", type=Path, default=root / "cache/infineon-device-db")
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--proxy", default="")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--retries", type=int, default=6)
    args = parser.parse_args()

    vendor_pack = args.vendor_pack.resolve()
    archive = (args.archive or args.cache_dir / f"device-db-{COMMIT}.tar.gz").resolve()
    if args.archive is None:
        download_archive(
            archive, refresh=args.refresh, proxy=args.proxy,
            timeout=args.timeout, retries=max(1, args.retries),
        )
    if not archive.exists():
        parser.error(f"archive does not exist: {archive}")

    observed_at = utc_now()
    version, records, parse_errors = parse_database(archive)
    previous_added = {row["device_id"] for row in read_csv(vendor_pack / "device-db-added-devices.csv")}
    devices = [
        row for row in read_csv(vendor_pack / "device-variants.csv")
        if row.get("device_id") not in previous_added
    ]
    for row in devices:
        row["features_json"] = json.dumps([
            item for item in json.loads(row.get("features_json") or "[]")
            if item.get("source_kind") != "infineon_device_db"
        ], ensure_ascii=False, sort_keys=True)
        row["source_id"] = ";".join(
            item for item in row.get("source_id", "").split(";")
            if item and not item.startswith(SOURCE_ID_PREFIX)
        )

    device_by_name = {row["device_name"].upper(): row for row in devices}
    record_by_name = {row["part_number"].upper(): row for row in records}
    templates: dict[str, list[dict[str, str]]] = defaultdict(list)
    for record in records:
        existing = device_by_name.get(record["part_number"].upper())
        if existing:
            templates[record["database_family"]].append(existing)

    provenance: list[dict[str, Any]] = []
    conflicts: list[dict[str, str]] = []
    added: list[dict[str, str]] = []
    field_updates: Counter[str] = Counter()

    def add_provenance(row: dict[str, str], field: str, value: Any, record: dict[str, Any]) -> None:
        provenance.append({
            "record_type": "device", "record_id": row["device_id"], "field_name": field,
            "source_id": SOURCE_ID, "source_url": SOURCE_URL,
            "source_path": record["source_path"],
            "source_value_json": json.dumps(value, ensure_ascii=False, sort_keys=True),
            "observed_at": observed_at, "verification_status": "manufacturer_device_database",
        })

    def update_scalar(row: dict[str, str], field: str, value: str, record: dict[str, Any]) -> None:
        if value in (None, ""):
            return
        previous = row.get(field, "")
        if previous and previous != value:
            conflicts.append({
                "device_id": row["device_id"], "device_name": row["device_name"],
                "field_name": field, "previous_value": previous, "device_db_value": value,
                "resolution": "official_exact_device_database_preferred",
                "source_path": record["source_path"],
            })
        if previous != value:
            row[field] = value
            field_updates[field] += 1
        add_provenance(row, field, value, record)

    for record in records:
        part_number = record["part_number"]
        row = device_by_name.get(part_number.upper())
        if row is None:
            family_templates = templates.get(record["database_family"], [])
            if family_templates:
                template = max(
                    family_templates,
                    key=lambda item: common_prefix_length(part_number, item["device_name"]),
                )
                family, series, product_line = template["family"], template["series"], template["product_line"]
                product_line_id = template["product_line_id"]
                product_type = template["product_type"]
            else:
                family, series, product_line = generated_hierarchy(record)
                product_line_id = f"infineon::{slug(family)}::{slug(series)}::{slug(product_line)}"
                product_type = "wireless_mcu" if record["domain"] == "Connectivity-BT-WIFI" else "general_purpose_mcu"
            cores = record["cores"]
            architecture = "+".join(sorted({item.get("Dcore", "") for item in cores if item.get("Dcore")}))
            row = {
                "device_id": f"infineon::{slug(part_number)}", "product_line_id": product_line_id,
                "manufacturer": "Infineon", "product_type": product_type,
                "architecture_class": architecture or "unknown", "family": family,
                "series": series, "product_line": product_line, "device_name": part_number,
                "generic_device_name": part_number, "manufacturer_variant_code": "",
                "processor_cores": json.dumps(cores, ensure_ascii=False, sort_keys=True),
                "max_clock_hz": "", "flash_bytes": "", "ram_bytes": "",
                "package_types": "", "pin_counts": "", "memory_regions_json": "[]",
                "features_json": "[]", "documents_json": "[]", "svd_files": "",
                "lifecycle": "unknown", "source_id": SOURCE_ID, "source_url": SOURCE_URL,
                "source_version": version, "observed_at": observed_at,
                "verification_status": "manufacturer_device_database",
            }
            devices.append(row)
            device_by_name[part_number.upper()] = row
            added.append({
                "device_id": row["device_id"], "device_name": part_number,
                "domain": record["domain"], "database_family": record["database_family"],
                "family": family, "series": series, "product_line": product_line,
                "source_path": record["source_path"],
            })

        parameters = record["parameters"]
        base = record["base"]
        clock = parse_count(parameters.get("Speed_MHz", ""))
        flash_kb = parse_count(parameters.get("Flash_KB", ""))
        ram_kb = parse_count(parameters.get("SRAM_KB", ""))
        update_scalar(row, "max_clock_hz", str(clock * 1_000_000) if clock is not None else "", record)
        update_scalar(row, "flash_bytes", str(flash_kb * 1024) if flash_kb is not None else base.get("flash", ""), record)
        update_scalar(row, "ram_bytes", str(ram_kb * 1024) if ram_kb is not None else base.get("ram", ""), record)
        package = parameters.get("Package", "") or base.get("package", "")
        update_scalar(row, "package_types", package, record)
        update_scalar(row, "pin_counts", package_pin_count(package), record)
        if not row.get("processor_cores") and record["cores"]:
            update_scalar(row, "processor_cores", json.dumps(record["cores"], ensure_ascii=False, sort_keys=True), record)

        availability = parameters.get("PartAvailability", "").lower()
        lifecycle = {"active": "active", "obsolete": "obsolete", "hidden": "hidden"}.get(availability, "")
        update_scalar(row, "lifecycle", lifecycle, record)

        current_features = [
            item for item in json.loads(row.get("features_json") or "[]")
            if item.get("source_kind") != "infineon_device_db"
        ]
        row["features_json"] = json.dumps(current_features + record["features"], ensure_ascii=False, sort_keys=True)
        field_updates["features_json"] += 1
        add_provenance(row, "features_json", record["features"], record)

        datasheet = parameters.get("DATASHEET", "")
        documents = json.loads(row.get("documents_json") or "[]")
        if datasheet.startswith("https://www.infineon.com/") and not any(item.get("url") == datasheet for item in documents):
            documents.append({"name": "datasheet", "title": f"{part_number} official product/datasheet page", "url": datasheet})
            row["documents_json"] = json.dumps(documents, ensure_ascii=False, sort_keys=True)
            field_updates["documents_json"] += 1
            add_provenance(row, "documents_json", documents[-1], record)

        row["source_id"] = append_semicolon(row.get("source_id", ""), SOURCE_ID)
        row["source_version"] = append_semicolon(row.get("source_version", ""), version)
        row["observed_at"] = observed_at
        row["verification_status"] = (
            "manufacturer_device_database" if row["source_id"] == SOURCE_ID
            else "multi_source_manufacturer_device_database"
        )

    parts_by_id = {
        row["orderable_part_id"]: row for row in read_csv(vendor_pack / "orderable-parts.csv")
        if not row.get("source_id", "").startswith(SOURCE_ID_PREFIX)
    }
    for record in records:
        part_number = record["part_number"]
        if not is_orderable_part(part_number):
            continue
        row = device_by_name[part_number.upper()]
        parameters = record["parameters"]
        temp_min, temp_max = parameters.get("TempMin_Celsius", ""), parameters.get("TempMax_Celsius", "")
        parts_by_id[f"infineon::{part_number.lower()}"] = {
            "orderable_part_id": f"infineon::{part_number.lower()}", "device_id": row["device_id"],
            "manufacturer": "Infineon", "family": row["family"], "series": row["series"],
            "product_line": row["product_line"], "device_name": row["device_name"],
            "part_number": part_number, "manufacturer_suffix": "", "package_code": "",
            "temperature_grade_code": "", "packing_code": "",
            "package_name": parameters.get("Package", ""),
            "temperature_range": f"{temp_min}..{temp_max} °C" if temp_min and temp_max else "",
            "packing_form": "", "lifecycle": row["lifecycle"], "source_id": SOURCE_ID,
            "source_url": SOURCE_URL, "observed_at": observed_at,
            "verification_status": "manufacturer_device_database",
            "decode_status": "official_device_db_part_record",
        }

    source_rows = {
        row["source_id"]: row for row in read_csv(vendor_pack / "sources.csv")
        if not row.get("source_id", "").startswith(SOURCE_ID_PREFIX)
    }
    source_rows[SOURCE_ID] = {
        "source_id": SOURCE_ID, "source_type": "manufacturer_device_database",
        "publisher": "Infineon", "title": "Infineon ModusToolbox device-db",
        "url": SOURCE_URL, "version": version, "observed_at": observed_at,
        "verification_scope": (
            "Exact leaf MCU part records from Infineon's ModusToolbox device database: availability, "
            "package, clock, flash, SRAM, device parameters, capabilities, and published part numbers."
        ),
    }

    unmatched = [
        {"device_id": row["device_id"], "device_name": row["device_name"], "reason": "no_exact_device_db_part_match"}
        for row in devices
        if row["device_name"].upper() not in record_by_name
        and row["device_id"] not in {item["device_id"] for item in added}
    ]
    payload = archive.read_bytes()
    manifest = [{
        "source_id": SOURCE_ID, "archive_url": ARCHIVE_URL, "cache_path": str(archive),
        "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload),
        "version": version, "commit": COMMIT,
    }]

    write_csv(vendor_pack / "device-variants.csv", DEVICE_FIELDS, sorted(devices, key=lambda row: row["device_name"]))
    write_csv(vendor_pack / "orderable-parts.csv", PART_FIELDS, sorted(parts_by_id.values(), key=lambda row: row["part_number"]))
    write_csv(vendor_pack / "sources.csv", SOURCE_FIELDS, sorted(source_rows.values(), key=lambda row: row["source_id"]))
    write_csv(vendor_pack / "device-db-field-provenance.csv", PROVENANCE_FIELDS, provenance)
    write_csv(vendor_pack / "device-db-conflicts.csv", CONFLICT_FIELDS, conflicts)
    write_csv(vendor_pack / "device-db-unmatched.csv", UNMATCHED_FIELDS, unmatched)
    write_csv(vendor_pack / "device-db-added-devices.csv", ADDED_FIELDS, added)
    write_csv(vendor_pack / "device-db-source-manifest.csv", MANIFEST_FIELDS, manifest)

    report = {
        "status": "ok" if not parse_errors else "partial",
        "database_version": version, "commit": COMMIT,
        "database_mcu_records": len(records), "input_catalog_devices": len(devices) - len(added),
        "exact_existing_matches": len(records) - len(added), "added_official_mcu_devices": len(added),
        "unmatched_catalog_devices": len(unmatched), "official_orderable_parts": len(parts_by_id),
        "field_updates": dict(sorted(field_updates.items())), "scalar_conflicts_resolved": len(conflicts),
        "provenance_records": len(provenance), "parse_errors": parse_errors,
        "observed_at": observed_at,
        "accuracy_policy": [
            "Only leaf records explicitly typed as mcu in the official device-db are accepted.",
            "Existing devices are joined only by case-insensitive exact part number.",
            "Missing CMSIS devices are added only from an official leaf PartNumber record.",
            "Part numbers containing lowercase x or the internal -device suffix are not treated as orderable parts.",
            "SCB and USIC are configurable serial blocks and are not multiplied into simultaneous UART/SPI/I2C counts.",
            "DMA channel count is kept separate from DMA controller count.",
            "ADC channel count is not treated as the number of external ADC-capable package pins.",
        ],
    }
    (vendor_pack / "device-db-import-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())

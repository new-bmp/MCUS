#!/usr/bin/env python3
"""Enrich Espressif products from official ESP-IDF per-SoC capability headers."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


COMMIT = "08e0d30a74ad0bfd5a34933142b80f45619ee410"
SOURCE_ID = f"espressif-idf-soc-caps:{COMMIT[:12]}"
SOURCE_URL = f"https://github.com/espressif/esp-idf/tree/{COMMIT}/components/soc"
TARGETS = {
    "ESP32": "esp32",
    "ESP32-C2/ESP8684": "esp32c2",
    "ESP32-C3": "esp32c3",
    "ESP32-C5": "esp32c5",
    "ESP32-C6": "esp32c6",
    "ESP32-C61": "esp32c61",
    "ESP32-H2": "esp32h2",
    "ESP32-H21": "esp32h21",
    "ESP32-H4": "esp32h4",
    "ESP32-P4": "esp32p4",
    "ESP32-S2": "esp32s2",
    "ESP32-S3": "esp32s3",
    "ESP32-S31": "esp32s31",
}
SOURCE_FIELDS = [
    "source_id", "source_type", "publisher", "title", "url", "version",
    "observed_at", "verification_scope",
]
PROVENANCE_FIELDS = [
    "record_type", "record_id", "field_name", "source_id", "source_url",
    "source_path", "source_value_json", "observed_at", "verification_status",
]
MANIFEST_FIELDS = ["target", "url", "cache_path", "sha256", "bytes", "commit"]
UNMATCHED_FIELDS = ["device_id", "device_name", "series", "reason"]
TIMER_COUNTS = {
    "esp32": 4, "esp32c2": 1, "esp32c3": 2, "esp32c5": 2,
    "esp32c6": 2, "esp32c61": 2, "esp32h2": 2, "esp32h21": 2,
    "esp32h4": 2, "esp32s2": 4, "esp32s3": 4, "esp32s31": 4,
}


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
    values = [item for item in value.split(";") if item and not item.startswith("espressif-idf-soc-caps:")]
    if addition not in values:
        values.append(addition)
    return ";".join(values)


def fetch(
    url: str, path: Path, *, refresh: bool, proxy: str, timeout: float, retries: int
) -> bytes:
    if path.exists() and not refresh:
        return path.read_bytes()
    handlers: list[Any] = []
    if proxy:
        handlers.append(urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
    opener = urllib.request.build_opener(*handlers)
    request = urllib.request.Request(
        url, headers={"User-Agent": "MCU-L-Catalog/0.5 (+official Espressif IDF importer)"}
    )
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with opener.open(request, timeout=timeout) as response:
                payload = response.read()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            return payload
        except (OSError, urllib.error.HTTPError, urllib.error.URLError) as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(0.75 * (2**attempt))
    assert last_error is not None
    raise last_error


def parse_headers(payload: bytes) -> dict[str, str]:
    text = payload.decode("utf-8")
    result: dict[str, str] = {}
    pattern = re.compile(r"^\s*#define\s+(SOC_[A-Z0-9_]+)(?:\s+(.+?))?\s*$")
    for raw in text.splitlines():
        line = raw.split("//", 1)[0].split("/*", 1)[0].rstrip()
        match = pattern.match(line)
        if match:
            result[match.group(1)] = (match.group(2) or "1").strip()
    return result


def integer(value: str) -> int | None:
    if not value:
        return None
    stripped = re.sub(r"(?<=\d)[uUlL]+\b", "", value).strip()
    while stripped.startswith("(") and stripped.endswith(")"):
        stripped = stripped[1:-1].strip()
    if re.fullmatch(r"0[xX][0-9a-fA-F]+|\d+", stripped):
        return int(stripped, 0)
    return None


def enabled(macros: dict[str, str], key: str) -> bool:
    value = integer(macros.get(key, ""))
    return value is not None and value > 0


def add_feature(
    result: list[dict[str, str]], feature_type: str, name: str,
    *, count: int | None = None, macro: str,
) -> None:
    row = {
        "type": feature_type, "name": name, "source_kind": "espressif_idf_soc_caps",
        "source_parameter": macro,
    }
    if count is not None:
        row["count"] = str(count)
    result.append(row)


def features(macros: dict[str, str], target: str) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    numeric = (
        ("GPIOPadCount", "SoC GPIO pad total; not module-exposed usable GPIO", "SOC_GPIO_PIN_COUNT"),
        ("UART", "UART controllers", "SOC_UART_NUM"),
        ("SPIPeripheralTotal", "Internal SPI peripheral total; Product Selector SPI count uses different semantics", "SOC_SPI_PERIPH_NUM"),
        ("I2C", "I2C controllers including HP/LP domains", "SOC_I2C_NUM"),
        ("I2S", "I2S controllers including HP/LP domains", "SOC_I2S_NUM"),
        ("CAN", "TWAI/CAN controllers", "SOC_TWAI_CONTROLLER_NUM"),
        ("USBOTG", "USB OTG controllers", "SOC_USB_OTG_PERIPH_NUM"),
        ("ADCUnits", "ADC converter units", "SOC_ADC_PERIPH_NUM"),
        ("LEDPWM", "LEDC PWM timers", "SOC_LEDC_TIMER_NUM"),
        ("MCPWM", "Motor-control PWM groups", "SOC_MCPWM_GROUPS"),
    )
    for feature_type, name, macro in numeric:
        value = integer(macros.get(macro, ""))
        if value is not None:
            add_feature(result, feature_type, name, count=value, macro=macro)

    if target in TIMER_COUNTS:
        add_feature(
            result, "Timer", "General-purpose hardware timers from timer-group register instances",
            count=TIMER_COUNTS[target], macro="timer_group_struct.h:instances*hw_timer[]",
        )
    elif enabled(macros, "SOC_GPTIMER_SUPPORTED"):
        add_feature(
            result, "TimerOther", "General-purpose timer supported; comparable timer count is not exposed by soc_caps.h",
            macro="SOC_GPTIMER_SUPPORTED",
        )
    if enabled(macros, "SOC_SYSTIMER_SUPPORTED"):
        add_feature(result, "SysTimer", "System timer", count=1, macro="SOC_SYSTIMER_SUPPORTED")

    dma_macros = [
        key for key in ("SOC_GDMA_SUPPORTED", "SOC_AHB_GDMA_SUPPORTED", "SOC_AXI_GDMA_SUPPORTED", "SOC_DW_GDMA_SUPPORTED", "SOC_CP_DMA_SUPPORTED")
        if enabled(macros, key)
    ]
    if dma_macros:
        add_feature(
            result, "DMA", "DMA engine present; controller/channel count not exposed by soc_caps.h",
            macro="+".join(dma_macros),
        )
    if enabled(macros, "SOC_DMA2D_SUPPORTED"):
        add_feature(result, "DMA2D", "2D DMA accelerator", count=1, macro="SOC_DMA2D_SUPPORTED")

    crypto = []
    for macro, label in (
        ("SOC_AES_SUPPORTED", "AES"), ("SOC_SHA_SUPPORTED", "SHA"),
        ("SOC_RSA_SUPPORTED", "RSA"), ("SOC_HMAC_SUPPORTED", "HMAC"),
        ("SOC_ECDSA_SUPPORTED", "ECDSA"), ("SOC_DS_SUPPORTED", "Digital Signature"),
    ):
        if enabled(macros, macro):
            crypto.append(label)
    if crypto:
        add_feature(result, "Crypto", "Hardware cryptography: " + ", ".join(crypto), count=1, macro="soc_caps crypto support macros")
    if enabled(macros, "SOC_RNG_SUPPORTED"):
        add_feature(result, "RNG", "Hardware random number generator", count=1, macro="SOC_RNG_SUPPORTED")

    for macro, feature_type, name in (
        ("SOC_USB_SERIAL_JTAG_SUPPORTED", "USB", "USB Serial/JTAG controller"),
        ("SOC_SDMMC_HOST_SUPPORTED", "SDIO", "SD/MMC host"),
        ("SOC_LCDCAM_CAM_SUPPORTED", "Camera", "LCD_CAM camera input"),
        ("SOC_LCD_RGB_SUPPORTED", "GLCD", "RGB LCD controller"),
        ("SOC_MIPI_DSI_SUPPORTED", "MIPI", "MIPI DSI"),
        ("SOC_RMT_SUPPORTED", "RMT", "Remote control transceiver"),
        ("SOC_LEDC_SUPPORTED", "LEDPWM", "LEDC PWM controller"),
        ("SOC_MCPWM_SUPPORTED", "MCPWM", "Motor-control PWM controller"),
    ):
        if enabled(macros, macro):
            add_feature(result, feature_type, name, count=1, macro=macro)
    add_feature(result, "VendorCapability", f"ESP-IDF target: {target}", count=1, macro=f"IDF_TARGET_{target.upper()}")
    return result


def fpu_value(macros: dict[str, str], target: str) -> str:
    """Return only an FPU state made explicit by the official target data."""
    if enabled(macros, "SOC_CPU_HAS_FPU"):
        return "FPU"
    # These official ESP-IDF target headers omit the macro because their
    # documented CPU ISA has no hardware floating-point extension. Keep new
    # or otherwise unspecified targets unknown instead of assuming a blanket
    # negative from an absent macro.
    no_fpu_targets = {
        "esp32c2", "esp32c3", "esp32c5", "esp32c6", "esp32c61",
        "esp32h2", "esp32h21", "esp32s2",
    }
    return "NO_FPU" if target in no_fpu_targets else ""


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--vendor-pack", type=Path, default=root / "data/vendor-packs/espressif")
    parser.add_argument("--cache-dir", type=Path, default=root / "cache/espressif-idf-soc-caps")
    parser.add_argument("--proxy", default="")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    vendor_pack = args.vendor_pack.resolve()
    observed_at = utc_now()
    target_macros: dict[str, dict[str, str]] = {}
    manifest = []
    errors = []
    def load_target(target: str) -> tuple[str, str, Path, bytes]:
        url = f"https://raw.githubusercontent.com/espressif/esp-idf/{COMMIT}/components/soc/{target}/include/soc/soc_caps.h"
        path = args.cache_dir / COMMIT / f"{target}-soc_caps.h"
        payload = fetch(
            url, path, refresh=args.refresh, proxy=args.proxy,
            timeout=args.timeout, retries=max(1, args.retries),
        )
        return target, url, path, payload

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(load_target, target): target for target in sorted(set(TARGETS.values()))}
        for future in as_completed(futures):
            target = futures[future]
            try:
                target, url, path, payload = future.result()
            except Exception as exc:
                url = f"https://raw.githubusercontent.com/espressif/esp-idf/{COMMIT}/components/soc/{target}/include/soc/soc_caps.h"
                errors.append({"target": target, "url": url, "error": str(exc)})
                continue
            target_macros[target] = parse_headers(payload)
            manifest.append({
                "target": target, "url": url, "cache_path": str(path),
                "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload), "commit": COMMIT,
            })

    def load_timer_header(target: str) -> tuple[str, str, Path, bytes]:
        url = f"https://raw.githubusercontent.com/espressif/esp-idf/{COMMIT}/components/soc/{target}/register/soc/timer_group_struct.h"
        path = args.cache_dir / COMMIT / f"{target}-timer_group_struct.h"
        payload = fetch(
            url, path, refresh=args.refresh, proxy=args.proxy,
            timeout=args.timeout, retries=max(1, args.retries),
        )
        return target, url, path, payload

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(load_timer_header, target): target for target in sorted(TIMER_COUNTS)}
        for future in as_completed(futures):
            target = futures[future]
            try:
                target, url, path, payload = future.result()
            except Exception as exc:
                url = f"https://raw.githubusercontent.com/espressif/esp-idf/{COMMIT}/components/soc/{target}/register/soc/timer_group_struct.h"
                errors.append({"target": target, "url": url, "error": str(exc)})
                target_macros.pop(target, None)
                continue
            timer_array = re.search(rb"hw_timer\s*\[(\d+)\]", payload)
            instances = set(re.findall(rb"extern\s+timg_dev_t\s+(TIMERG\d+)", payload))
            discovered = int(timer_array.group(1)) * len(instances) if timer_array and instances else 0
            if discovered != TIMER_COUNTS[target]:
                errors.append({
                    "target": target, "url": url,
                    "error": f"timer_instance_count_mismatch:expected={TIMER_COUNTS[target]}:discovered={discovered}",
                })
                target_macros.pop(target, None)
            manifest.append({
                "target": target + ":timer_group", "url": url, "cache_path": str(path),
                "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload), "commit": COMMIT,
            })

    devices = read_csv(vendor_pack / "device-variants.csv")
    provenance = []
    unmatched = []
    updates: Counter[str] = Counter()
    for row in devices:
        current = [
            item for item in json.loads(row.get("features_json") or "[]")
            if item.get("source_kind") != "espressif_idf_soc_caps"
        ]
        target = TARGETS.get(row.get("series", ""))
        if not target or target not in target_macros:
            unmatched.append({
                "device_id": row["device_id"], "device_name": row["device_name"],
                "series": row.get("series", ""),
                "reason": "series_not_present_in_selected_esp_idf_commit" if not target else "target_header_unavailable",
            })
            row["features_json"] = json.dumps(current, ensure_ascii=False, sort_keys=True)
            continue
        additions = features(target_macros[target], target)
        processors = json.loads(row.get("processor_cores") or "[]")
        fpu = fpu_value(target_macros[target], target)
        if fpu:
            for processor in processors:
                processor["Dfpu"] = fpu
            row["processor_cores"] = json.dumps(processors, ensure_ascii=False, sort_keys=True)
        row["features_json"] = json.dumps(current + additions, ensure_ascii=False, sort_keys=True)
        row["source_id"] = append_semicolon(row.get("source_id", ""), SOURCE_ID)
        row["source_version"] = append_semicolon(row.get("source_version", ""), COMMIT[:12])
        row["verification_status"] = "multi_source_manufacturer_device_database"
        updates[target] += 1
        provenance.append({
            "record_type": "device", "record_id": row["device_id"], "field_name": "features_json",
            "source_id": SOURCE_ID, "source_url": SOURCE_URL,
            "source_path": f"components/soc/{target}/include/soc/soc_caps.h",
            "source_value_json": json.dumps(additions, ensure_ascii=False, sort_keys=True),
            "observed_at": observed_at, "verification_status": "manufacturer_software_device_capabilities",
        })
        if fpu:
            provenance.append({
                "record_type": "device", "record_id": row["device_id"], "field_name": "processor_cores.Dfpu",
                "source_id": SOURCE_ID, "source_url": SOURCE_URL,
                "source_path": f"components/soc/{target}/include/soc/soc_caps.h",
                "source_value_json": json.dumps(fpu),
                "observed_at": observed_at, "verification_status": "manufacturer_software_device_capabilities",
            })

    sources = {
        row["source_id"]: row for row in read_csv(vendor_pack / "sources.csv")
        if not row.get("source_id", "").startswith("espressif-idf-soc-caps:")
    }
    sources[SOURCE_ID] = {
        "source_id": SOURCE_ID, "source_type": "manufacturer_software_device_capabilities",
        "publisher": "Espressif", "title": "ESP-IDF per-SoC capability headers",
        "url": SOURCE_URL, "version": COMMIT, "observed_at": observed_at,
        "verification_scope": (
            "Per-SoC hardware capability and count macros from ESP-IDF soc_caps.h. Product Selector rows are "
            "mapped to the IDF target by their official series; unsupported series remain unknown."
        ),
    }
    write_csv(vendor_pack / "device-variants.csv", list(devices[0].keys()), devices)
    write_csv(vendor_pack / "sources.csv", SOURCE_FIELDS, sorted(sources.values(), key=lambda row: row["source_id"]))
    write_csv(vendor_pack / "idf-soc-caps-provenance.csv", PROVENANCE_FIELDS, provenance)
    write_csv(vendor_pack / "idf-soc-caps-manifest.csv", MANIFEST_FIELDS, manifest)
    write_csv(vendor_pack / "idf-soc-caps-unmatched.csv", UNMATCHED_FIELDS, unmatched)
    report = {
        "status": "ok" if not errors else "partial", "commit": COMMIT,
        "targets_requested": len(set(TARGETS.values())), "targets_loaded": len(target_macros),
        "device_records_augmented": sum(updates.values()), "devices_unmatched": len(unmatched),
        "augmented_by_target": dict(sorted(updates.items())), "download_errors": errors,
        "observed_at": observed_at,
        "accuracy_policy": [
            "Only macros in the selected official ESP-IDF commit are used.",
            "Product Selector rows map to an ESP-IDF target only through their official series.",
            "General-purpose timer count is accepted only after checking timer-group instances and hw_timer array size.",
            "ESP32-P4 retains GPTimer support without a count because the selected target has no matching timer_group_struct.h path.",
            "LEDC and MCPWM resources remain separate from the general-purpose timer field.",
            "DMA presence is not reported as a controller or channel count.",
            "FPU state is copied from SOC_CPU_HAS_FPU or an explicitly reviewed target ISA; unreviewed targets stay unknown.",
            "ESP32-E22 and ESP8266 stay unknown because this ESP-IDF snapshot does not expose matching targets.",
        ],
    }
    (vendor_pack / "idf-soc-caps-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())

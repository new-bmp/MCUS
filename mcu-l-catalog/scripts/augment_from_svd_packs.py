#!/usr/bin/env python3
"""Augment device capabilities from explicitly bound official CMSIS SVDs.

The PDSC selects an SVD path for each device. This importer downloads the
matching official .pack archive, parses only those selected SVD files, and
adds peripheral-instance evidence without inferring across device families.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import hashlib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


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


def source_matches(source: dict[str, str], patterns: list[str]) -> bool:
    if not patterns:
        return True
    value = " ".join(
        source.get(key, "") for key in ("source_id", "publisher", "title", "url")
    ).lower()
    return any(fnmatch.fnmatchcase(value, pattern.lower()) for pattern in patterns)


def pack_url(source: dict[str, str]) -> str:
    url = source["url"]
    parsed = urllib.parse.urlparse(url)
    filename = PurePosixPath(parsed.path).name
    stem = filename[:-5] if filename.lower().endswith(".pdsc") else filename
    version = source["version"]
    path = str(PurePosixPath(parsed.path).parent / f"{stem}.{version}.pack")
    return urllib.parse.urlunparse(parsed._replace(path=path))


def fetch_pack(
    url: str,
    output: Path,
    *,
    refresh: bool,
    timeout: float,
    proxy: str,
) -> tuple[str, int]:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise ValueError(f"SVD pack must use official HTTPS URL: {url}")
    if output.exists() and not refresh:
        payload = output.read_bytes()
        return hashlib.sha256(payload).hexdigest(), len(payload)
    handlers: list[Any] = []
    if proxy:
        handlers.append(urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
    opener = urllib.request.build_opener(*handlers)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "MCU-L-Catalog/0.5 (+official SVD importer)"},
    )
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with opener.open(request, timeout=timeout) as response:
                payload = response.read()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(payload)
            return hashlib.sha256(payload).hexdigest(), len(payload)
        except (OSError, urllib.error.HTTPError, urllib.error.URLError) as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(0.75 * (2**attempt))
    assert last_error is not None
    raise last_error


def peripheral_type(name: str, group_name: str) -> str | None:
    value = f"{name} {group_name}".upper().replace("-", "_")
    compact = re.sub(r"[^A-Z0-9]", "", value)
    first = re.sub(r"[^A-Z0-9]", "", name.upper())
    rules: list[tuple[str, tuple[str, ...]]] = [
        ("ADCUnits", ("ADC", "SARADC")),
        ("DAC", ("DAC",)),
        ("USART", ("USART", "SERCOMUSART")),
        ("UART", ("UART", "LPUART", "LPUART")),
        ("I3C", ("I3C",)),
        ("I2C", ("I2C", "IIC", "TWI")),
        ("I2S", ("I2S", "SAI")),
        ("SPI", ("SPI", "QSPI", "OSPI", "OCTOSPI")),
        ("CAN", ("CAN", "FDCAN", "MCAN")),
        ("USBOTG", ("USBOTG", "USBOTGFS", "USBOTGHS", "OTGFS", "OTGHS")),
        ("USBD", ("USB", "USBD", "USBFS", "USBHS")),
        ("ETH", ("ETH", "ETHERNET", "GMAC", "EMAC")),
        ("SDIO", ("SDIO", "SDMMC", "SDHC")),
        ("DMA", ("DMA", "GPDMA", "BDMA", "MDMA", "XDMA")),
        ("RNG", ("RNG", "TRNG")),
        ("Crypto", ("CRYP", "CRYPTO", "AES", "HASH", "PKA", "HSM")),
        ("Camera", ("DCMI", "CSI", "PSSI")),
        ("LCD", ("LTDC", "LCD", "TFT")),
        ("COMP", ("COMP", "ACMP")),
        ("OPAMP", ("OPAMP",)),
        ("WDT", ("WDT", "IWDG", "WWDG", "WATCHDOG")),
        ("RTC", ("RTC",)),
    ]
    for feature_type, prefixes in rules:
        if any(first.startswith(prefix) or compact.startswith(prefix) for prefix in prefixes):
            return feature_type
    if re.match(r"^(?:TIM|TIMER|TMR|GPT|GPTM|LPTIM|HRTIM)\d", first):
        return "Timer"
    if re.match(r"^(?:GPIO|PORT)[A-Z0-9]", first):
        return "GPIOPort"
    return None


def parse_svd(payload: bytes) -> tuple[list[dict[str, Any]], int]:
    root = ET.fromstring(payload)
    instances: dict[str, set[str]] = defaultdict(set)
    peripheral_nodes = [node for node in root.iter() if local_name(node.tag) == "peripheral"]
    for peripheral in peripheral_nodes:
        values = {local_name(child.tag): (child.text or "").strip() for child in peripheral}
        name = values.get("name", "")
        if not name:
            continue
        feature_type = peripheral_type(name, values.get("groupName", ""))
        if not feature_type:
            continue
        dim = 1
        try:
            dim = max(1, int(values.get("dim", "1"), 0))
        except ValueError:
            dim = 1
        if "%s" in name or "%d" in name:
            for index in range(dim):
                instances[feature_type].add(name.replace("%s", str(index)).replace("%d", str(index)))
        else:
            instances[feature_type].add(name)
    features = []
    for feature_type, names in sorted(instances.items()):
        ordered = sorted(names, key=lambda value: [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", value)])
        features.append(
            {
                "type": feature_type,
                "count": str(len(ordered)),
                "n": str(len(ordered)),
                "name": "SVD peripheral instances: " + ", ".join(ordered),
                "instances": ordered,
                "source_kind": "cmsis_svd",
                "verification_status": "official_pack_svd",
            }
        )
    return features, len(peripheral_nodes)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vendor-pack", required=True, type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--source-pattern", action="append", default=[])
    parser.add_argument("--proxy", default="")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--max-packs", type=int)
    args = parser.parse_args()

    vendor_pack = args.vendor_pack.resolve()
    cache_dir = args.cache_dir or vendor_pack.parents[2] / "cache" / "svd-packs" / vendor_pack.name
    devices = read_csv(vendor_pack / "device-variants.csv")
    sources = read_csv(vendor_pack / "sources.csv")
    source_map = {row["source_id"]: row for row in sources}
    devices_by_source: dict[str, list[dict[str, str]]] = defaultdict(list)
    for device in devices:
        if not device.get("svd_files"):
            continue
        for source_id in device.get("source_id", "").split(";"):
            source = source_map.get(source_id)
            if source and source.get("source_type") == "cmsis_pdsc" and source_matches(source, args.source_pattern):
                devices_by_source[source_id].append(device)
                break

    selected = sorted(devices_by_source)
    if args.max_packs is not None:
        selected = selected[: args.max_packs]
    observed_at = utc_now()
    errors: list[dict[str, str]] = []
    reports: list[dict[str, Any]] = []
    svd_cache: dict[tuple[str, str], list[dict[str, Any]]] = {}
    added_sources: dict[str, dict[str, str]] = {}
    augmented_devices = 0

    for source_id in selected:
        source = source_map[source_id]
        url = pack_url(source)
        archive_name = PurePosixPath(urllib.parse.urlparse(url).path).name
        archive_path = cache_dir / archive_name
        try:
            sha256, archive_bytes = fetch_pack(
                url,
                archive_path,
                refresh=args.refresh,
                timeout=args.timeout,
                proxy=args.proxy,
            )
            pack_source_id = "svd:" + source_id.removeprefix("cmsis:")
            added_sources[pack_source_id] = {
                "source_id": pack_source_id,
                "source_type": "official_pack_svd",
                "publisher": source["publisher"],
                "title": source["title"] + " SVD files",
                "url": url,
                "version": f"{source['version']};sha256:{sha256}",
                "observed_at": observed_at,
                "verification_scope": "peripheral instances explicitly present in the device-bound SVD; not pin mux or analog-channel coverage",
            }
            with zipfile.ZipFile(archive_path) as archive:
                member_map = {name.replace("\\", "/").lower(): name for name in archive.namelist()}
                used_svd: set[str] = set()
                for device in devices_by_source[source_id]:
                    features = json.loads(device.get("features_json") or "[]")
                    features = [item for item in features if item.get("source_kind") != "cmsis_svd"]
                    matched_features: list[dict[str, Any]] = []
                    for requested in device["svd_files"].split(";"):
                        normalized = requested.replace("\\", "/").lstrip("./").lower()
                        member = member_map.get(normalized)
                        if member is None:
                            suffixes = [value for key, value in member_map.items() if key.endswith("/" + normalized) or key == normalized]
                            member = suffixes[0] if len(suffixes) == 1 else None
                        if member is None:
                            errors.append({"source_id": source_id, "device_name": device["device_name"], "svd_file": requested, "error": "svd_member_not_found"})
                            continue
                        cache_key = (sha256, member.lower())
                        if cache_key not in svd_cache:
                            parsed, peripheral_nodes = parse_svd(archive.read(member))
                            for item in parsed:
                                item["source_id"] = pack_source_id
                                item["source_file"] = member
                            svd_cache[cache_key] = parsed
                        matched_features.extend(dict(item) for item in svd_cache[cache_key])
                        used_svd.add(member)
                    if matched_features:
                        features.extend(matched_features)
                        device["features_json"] = json.dumps(features, ensure_ascii=False, sort_keys=True)
                        augmented_devices += 1
            reports.append(
                {
                    "source_id": source_id,
                    "pack_url": url,
                    "sha256": sha256,
                    "archive_bytes": archive_bytes,
                    "devices": len(devices_by_source[source_id]),
                    "unique_svd_files": len(used_svd),
                    "status": "ok",
                }
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            errors.append({"source_id": source_id, "device_name": "", "svd_file": "", "error": error})
            reports.append({"source_id": source_id, "pack_url": url, "status": "error", "error": error})

    if devices:
        write_csv(vendor_pack / "device-variants.csv", list(devices[0].keys()), devices)
    merged_sources = {row["source_id"]: row for row in sources}
    merged_sources.update(added_sources)
    if merged_sources:
        fields = ["source_id", "source_type", "publisher", "title", "url", "version", "observed_at", "verification_scope"]
        write_csv(vendor_pack / "sources.csv", fields, sorted(merged_sources.values(), key=lambda row: row["source_id"]))
    error_fields = ["source_id", "device_name", "svd_file", "error"]
    write_csv(vendor_pack / "svd-import-errors.csv", error_fields, errors)
    report = {
        "status": "ok" if not errors else "partial",
        "selected_packs": len(selected),
        "successful_packs": sum(row["status"] == "ok" for row in reports),
        "failed_packs": sum(row["status"] == "error" for row in reports),
        "augmented_device_records": augmented_devices,
        "unique_svd_payloads": len(svd_cache),
        "errors": len(errors),
        "proxy_mode": "explicit" if args.proxy else "environment_or_direct",
        "packs": reports,
    }
    (vendor_pack / "svd-import-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if any(row["status"] == "error" for row in reports) else 0


if __name__ == "__main__":
    raise SystemExit(main())

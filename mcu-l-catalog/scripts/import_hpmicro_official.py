#!/usr/bin/env python3
"""Import HPMicro MCU variants from official selector pages and HPM SDK.

The website selector supplies exact model rows and per-model values.  Exact
purchase codes are collected separately from the official product cards.  The
SDK is used only for vendor IP names on matching SoC definitions; it is never
used to copy a high-end SoC's peripheral counts onto a smaller sibling.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import os
import re
import shutil
import subprocess
import urllib.parse
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any

from vendor_import_common import (
    DEVICE_FIELDS,
    ERROR_FIELDS,
    FIELD_PROVENANCE_FIELDS,
    PART_FIELDS,
    SOURCE_FIELDS,
    OfficialFetcher,
    Snapshot,
    slug,
    utc_now,
    write_csv,
    write_json,
)


MANUFACTURER = "HPMicro"
SITEMAP_URL = "https://www.hpmicro.com/sitemap.xml"
PRODUCT_ROOT = "https://www.hpmicro.com/product-center/microcontroller"
GITHUB_API = "https://api.github.com/repos/hpmicro/hpm_sdk"
HPM_WEB_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) MCUS/0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.7",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.6",
}

FALLBACK_SELECTION_URLS = [
    f"{PRODUCT_ROOT}/hpm5e00/aaaab",
    f"{PRODUCT_ROOT}/hpm6p00/hpm6p00",
    f"{PRODUCT_ROOT}/hpm6e00/6e00-selection-table",
    f"{PRODUCT_ROOT}/hpm6800/selection-table",
    f"{PRODUCT_ROOT}/hpm5300/5300-selection-table",
    f"{PRODUCT_ROOT}/hpm6200/6200-selection-table",
    f"{PRODUCT_ROOT}/hpm6300/6300-selection-table",
    f"{PRODUCT_ROOT}/hpm6700/6700-6400-selection-table",
]

# Only exact product lines with their own SDK SoC directory are mapped.  A
# cut-down sibling is intentionally not assigned the generic high-end header.
SDK_SOC_PATHS = {
    "HPM5301": "HPM5300/HPM5301",
    "HPM5361": "HPM5300/HPM5361",
    "HPM53M1": "HPM5300/HPM53M1",
    "HPM5E31": "HPM5E00/HPM5E31",
    "HPM5E3Y": "HPM5E00/HPM5E3Y",
    "HPM6280": "HPM6200/HPM6280",
    "HPM6360": "HPM6300/HPM6360",
    "HPM6750": "HPM6700/HPM6750",
    "HPM6880": "HPM6800/HPM6880",
    "HPM6E80": "HPM6E00/HPM6E80",
    "HPM6P81": "HPM6P00/HPM6P81",
}

SDK_VENDOR_IPS = {
    "PSEC": ("Security", "HPMicro PSEC"),
    "PIXEL_MUX": ("VendorCapability", "HPMicro Pixelmux"),
    "PLA": ("VendorCapability", "HPMicro PLA"),
    "PLB": ("VendorCapability", "HPMicro PLB"),
    "SEI": ("VendorCapability", "HPMicro SEI"),
    "MMC": ("VendorCapability", "HPMicro MMC"),
    "QEI": ("VendorCapability", "HPMicro QEI"),
    "QEO": ("VendorCapability", "HPMicro QEO"),
    "ESC": ("VendorCapability", "HPMicro EtherCAT Slave Controller (ESC)"),
    "GPU": ("Accelerator", "HPMicro OpenVG GPU"),
    "JPEG": ("Accelerator", "HPMicro JPEG codec"),
    "FFA": ("Accelerator", "HPMicro FFA (FFT/filter accelerator)"),
    "MTG": ("VendorCapability", "HPMicro MTG"),
    "RDC": ("VendorCapability", "HPMicro RDC"),
    "VSC": ("VendorCapability", "HPMicro VSC"),
    "CLC": ("VendorCapability", "HPMicro CLC"),
    "LOBS": ("VendorCapability", "HPMicro LOBS"),
    "TFA": ("Accelerator", "HPMicro TFA"),
    "PDGO": ("VendorCapability", "HPMicro PDGO"),
    "PPI": ("VendorCapability", "HPMicro PPI"),
}

PRODUCT_LINE_FIELDS = [
    "product_line_id", "manufacturer", "product_type", "architecture_class",
    "family", "series", "product_line", "device_variant_count", "source_ids",
    "observed_at", "verification_status",
]


def clean_html(value: str) -> str:
    value = re.sub(r"(?is)<script[^>]*>.*?</script>", " ", value)
    value = re.sub(r"(?is)<style[^>]*>.*?</style>", " ", value)
    value = re.sub(r"(?is)<!--.*?-->", " ", value)
    value = re.sub(r"<[^>]+>", " ", value)
    return re.sub(r"\s+", " ", html.unescape(value)).strip()


def fetch_snapshot(
    fetcher: OfficialFetcher,
    url: str,
    *,
    cache_name: str,
    headers: dict[str, str] | None = None,
) -> Snapshot:
    """Fetch normally, with a Windows web-stack fallback for hpmicro.com.

    The public HPMicro server occasionally closes Python/curl TLS sessions
    without a close-notify alert while Windows' web stack still receives the
    complete response.  Cached bytes and their digest remain identical inputs
    to the rest of the importer.
    """
    try:
        return fetcher.fetch(url, cache_name=cache_name, headers=headers)
    except Exception:
        host = (urllib.parse.urlparse(url).hostname or "").lower()
        powershell = shutil.which("powershell") or shutil.which("powershell.exe")
        if host != "www.hpmicro.com" or not powershell:
            raise
        fetcher._validate_url(url)
        cache_path = fetcher.cache_dir / cache_name
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        environment = dict(os.environ)
        environment["MCUS_FETCH_URL"] = url
        environment["MCUS_FETCH_OUT"] = str(cache_path.resolve())
        command = (
            "$ErrorActionPreference='Stop';"
            "$headers=@{'User-Agent'='Mozilla/5.0 (Windows NT 10.0; Win64; x64) MCUS/0.8';"
            "'Accept-Language'='zh-CN,zh;q=0.9,en;q=0.6'};"
            "Invoke-WebRequest -UseBasicParsing -Headers $headers "
            "-Uri $env:MCUS_FETCH_URL -OutFile $env:MCUS_FETCH_OUT -TimeoutSec 60"
        )
        subprocess.run(
            [powershell, "-NoProfile", "-NonInteractive", "-Command", command],
            check=True,
            env=environment,
            stdout=subprocess.DEVNULL,
        )
        payload = cache_path.read_bytes()
        return Snapshot(
            url=url,
            path=cache_path,
            payload=payload,
            sha256=hashlib.sha256(payload).hexdigest(),
            observed_at=utc_now(),
        )


def official_rows(page: str) -> tuple[dict[int, str], list[dict[str, str]]]:
    """Parse the official div-based selection table without browser automation."""
    page = re.sub(r"(?is)<!--.*?-->", "", page)
    headers = {
        int(index): clean_html(label)
        for index, label in re.findall(
            r'<li class="mtcr_all_th mtcr_all_th(\d+)"[^>]*>.*?'
            r'<span class="mc_tab_sortbtn">\s*(.*?)\s*</span>',
            page,
            re.S,
        )
    }
    starts = list(re.finditer(r'<li class="mtbr_item" data-name="([^"]+)"', page))
    rows: list[dict[str, str]] = []
    for offset, match in enumerate(starts):
        end = starts[offset + 1].start() if offset + 1 < len(starts) else len(page)
        chunk = page[match.start():end]
        values = {
            int(index): clean_html(value)
            for index, value in re.findall(
                r'<div class="mc_tab_td mc_tab_td(\d+)">.*?'
                r'<div class="std_text14">\s*<p>(.*?)</p>',
                chunk,
                re.S,
            )
        }
        row = {headers[index]: value for index, value in values.items() if index in headers}
        row["型号"] = html.unescape(match.group(1)).strip().upper()
        rows.append(row)
    return headers, rows


def official_rows_json(payload: bytes) -> tuple[dict[int, str], list[dict[str, str]]]:
    """Read a compact DOM snapshot captured from the same official page."""
    record = json.loads(payload.decode("utf-8"))
    headers = {int(index): str(label) for index, label in record.get("headers", {}).items()}
    rows: list[dict[str, str]] = []
    for item in record.get("rows", []):
        row = {
            headers[int(index)]: str(value)
            for index, value in item.get("values", {}).items()
            if int(index) in headers
        }
        row["型号"] = str(item.get("model", "")).strip().upper()
        rows.append(row)
    return headers, rows


def column(row: dict[str, str], *tokens: str) -> str:
    for key, value in row.items():
        normalized = re.sub(r"\s+", "", key).lower()
        if all(token.lower() in normalized for token in tokens):
            return value.strip()
    return ""


def is_absent(value: str) -> bool:
    return value.strip().lower() in {"/", "-", "无", "none", "n/a", "不支持"}


def source_count(value: str, *, present_default: int = 1) -> int | None:
    value = clean_html(value)
    if not value:
        return None
    if is_absent(value):
        return 0
    if re.fullmatch(r"\d+", value):
        return int(value)
    match = re.match(r"\s*(\d+)\s*[x×*]", value, re.I)
    if match:
        return int(match.group(1))
    match = re.search(r"[x×*]\s*(\d+)", value, re.I)
    if match:
        return int(match.group(1))
    match = re.search(r"(\d+)\s*路", value)
    if match:
        return int(match.group(1))
    return present_default


def memory_bytes(value: str, *, default_unit: str = "") -> int | None:
    value = value.strip().replace(" ", "")
    if not value or is_absent(value):
        return None
    match = re.search(r"(\d+(?:\.\d+)?)\s*(KB|MB|K|M)?", value, re.I)
    if not match:
        return None
    number = float(match.group(1))
    unit = (match.group(2) or default_unit).upper()
    if unit in {"M", "MB"}:
        number *= 1024 * 1024
    elif unit in {"K", "KB"}:
        number *= 1024
    return int(number)


def frequency_hz(value: str) -> int | None:
    value = value.strip()
    match = re.search(r"(\d+(?:\.\d+)?)\s*(GHz|MHz)?", value, re.I)
    if not match:
        return None
    number = float(match.group(1))
    unit = (match.group(2) or "MHz").lower()
    return int(number * (1_000_000_000 if unit == "ghz" else 1_000_000))


def product_line_for(model: str) -> str:
    match = re.match(r"^(HPM[0-9A-Z]{4})", model.upper())
    return match.group(1) if match else model.upper()


def series_for(product_line: str) -> str:
    upper = product_line.upper()
    prefixes = (
        ("HPM5E", "HPM5E00"), ("HPM6P", "HPM6P00"),
        ("HPM6E", "HPM6E00"), ("HPM68", "HPM6800"),
        ("HPM67", "HPM6700"), ("HPM64", "HPM6400"),
        ("HPM53", "HPM5300"), ("HPM62", "HPM6200"),
        ("HPM63", "HPM6300"),
    )
    for prefix, series in prefixes:
        if upper.startswith(prefix):
            return series
    return upper[:5] + "00"


def pin_count(package: str) -> str:
    patterns = (
        r"(?:BGA|LQFP|eLQFP|QFN|WLCSP|CSP)[-_]?(\d+)",
        r"(\d+)\s*(?:BGA|LQFP|eLQFP|QFN|WLCSP|CSP)",
    )
    for pattern in patterns:
        match = re.search(pattern, package, re.I)
        if match:
            return match.group(1)
    return ""


def add_feature(
    features: list[dict[str, Any]],
    feature_type: str,
    name: str,
    count: int | None,
    *,
    raw: str = "",
    bits: int | None = None,
    source_kind: str = "hpmicro_product_selector",
    source_parameter: str = "",
) -> None:
    if count is None and not raw:
        return
    key = (feature_type, name, source_kind)
    if any((item.get("type"), item.get("name"), item.get("source_kind")) == key for item in features):
        return
    feature: dict[str, Any] = {
        "type": feature_type,
        "name": name,
        "source_kind": source_kind,
    }
    if count is not None:
        feature["count"] = str(count)
        feature["n"] = str(count)
    if bits is not None:
        feature["m"] = str(bits)
    elif raw and raw not in {str(count), "/"}:
        feature["m"] = raw
    if source_parameter:
        feature["source_parameter"] = source_parameter
    features.append(feature)


def selector_features(row: dict[str, str]) -> list[dict[str, Any]]:
    features: list[dict[str, Any]] = []

    def counted(feature_type: str, name: str, value: str, *, bits: int | None = None) -> None:
        add_feature(features, feature_type, name, source_count(value), raw=value, bits=bits, source_parameter=name)

    counted("I2S", "I²S controllers", column(row, "i²s"))
    counted("Audio", "Digital audio output", column(row, "数字音频输出"))
    counted("ExtBus", "DDR controller", column(row, "ddr"))
    counted("ExtBus", "SDRAM controller", column(row, "sdram"))
    counted("SDIO", "SD/eMMC controller", column(row, "sd/emmc"))
    counted("ExtBus", "HPMicro XPI", column(row, "xpi"))

    ethernet = column(row, "以太网")
    ethernet_links = [
        int(value)
        for value in re.findall(r"(\d+)\s*[x×*]\s*(?:1000|100|10)\b", ethernet, re.I)
    ]
    add_feature(
        features,
        "ETH",
        "Ethernet resources",
        sum(ethernet_links) if ethernet_links else source_count(ethernet),
        raw=ethernet,
        source_parameter="Ethernet resources",
    )
    if "EtherCAT" in ethernet:
        add_feature(features, "VendorCapability", "HPMicro EtherCAT Slave Controller (ESC)", 1, raw=ethernet)
    if "TSN" in ethernet:
        add_feature(features, "VendorCapability", "HPMicro TSN Ethernet switch", 1, raw=ethernet)
    phy_match = re.search(r"(\d+)\s*个?百兆PHY", ethernet, re.I)
    if phy_match:
        add_feature(features, "ETHPhy", "Integrated 100 Mbps Ethernet PHY", int(phy_match.group(1)), raw=ethernet)

    can = column(row, "can")
    counted("CAN", "CAN FD controllers" if "fd" in can.lower() else "CAN controllers", can)
    counted("USB", "High-speed USB with integrated PHY", column(row, "usb"))
    counted("SPI", "SPI controllers", column(row, "spi"))
    counted("I2C", "I²C controllers", column(row, "i²c"))
    counted("UART", "UART controllers", column(row, "uart"))
    counted("Audio", "PDM digital microphone interface", column(row, "pdm"))

    graphics_codec = column(row, "图形编解码器")
    if not is_absent(graphics_codec):
        add_feature(features, "Accelerator", f"HPMicro {graphics_codec}", source_count(graphics_codec), raw=graphics_codec)
    graphics_accel = column(row, "图形加速")
    if not is_absent(graphics_accel):
        generic_support = graphics_accel.strip().lower() in {"yes", "supported"} or len(graphics_accel.strip()) <= 4
        add_feature(
            features,
            "Accelerator",
            "HPMicro graphics accelerator" if generic_support else f"HPMicro {graphics_accel}",
            source_count(graphics_accel),
            raw="" if generic_support else graphics_accel,
        )
    counted("GLCD", "LCD/display interfaces", column(row, "lcd"))
    counted("Camera", "Camera interfaces", column(row, "摄像头接口"))

    crypto = column(row, "crypto")
    add_feature(features, "Crypto", f"Hardware cryptography: {crypto}" if not is_absent(crypto) else "Hardware cryptography", source_count(crypto), raw=crypto)
    security = column(row, "安全特性")
    add_feature(features, "Security", security or "Security features", source_count(security), raw=security)
    functional_safety = column(row, "功能安全")
    add_feature(features, "Security", f"Functional safety: {functional_safety}" if not is_absent(functional_safety) else "Functional safety", source_count(functional_safety), raw=functional_safety)

    motor = column(row, "电机控制")
    pwm_match = re.search(r"(\d+)\s*[x×]\s*8\s*通道\s*PWM(?:v2)?", motor, re.I)
    add_feature(features, "PWM", "HPMicro motor-control PWM", int(pwm_match.group(1)) if pwm_match else source_count(motor, present_default=0), raw=motor)
    motor_tokens = {
        "QEIv2": r"(\d+)\s*[x×]\s*QEIv2",
        "QEOv2": r"(\d+)\s*[x×]\s*QEOv2",
        "QEO": r"(\d+)\s*[x×]\s*QEO(?!v2)",
        "SEI": r"(\d+)\s*[x×]\s*SEI",
        "MMC": r"(\d+)\s*[x×]\s*MMC",
        "MTG": r"(\d+)\s*[x×]\s*MTG",
        "HALL": r"(\d+)\s*[x×]\s*(?:HALL|霍尔)",
    }
    for token, pattern in motor_tokens.items():
        match = re.search(pattern, motor, re.I)
        if match:
            add_feature(features, "VendorCapability", f"HPMicro {token}", int(match.group(1)), raw=motor)
    if "三相独立半桥驱动" in motor:
        add_feature(features, "VendorCapability", "HPMicro three-phase independent half-bridge driver", 1, raw=motor)

    timer = column(row, "gptimer")
    add_feature(features, "Timer", "HPMicro GPTimer modules", source_count(timer), raw=timer, bits=32)
    counted("WDT", "Watchdog timers", column(row, "wdg"))
    counted("RTC", "Real-time clock", column(row, "rtc"))

    # These official columns enumerate ADC converter blocks, not pins and not
    # aggregate analog channels.  ADC channels remain unknown unless a source
    # explicitly provides a channel count.
    counted("ADCUnits", "12-bit ADC converter units", column(row, "adc12bit"), bits=12)
    counted("ADCUnits", "16-bit ADC converter units", column(row, "adc16bit"), bits=16)
    counted("DAC", "12-bit DAC converter units", column(row, "dac12bit"), bits=12)
    counted("COMP", "Analog comparators", column(row, "比较器"))
    counted("AnalogOther", "Operational amplifiers", column(row, "运算放大器"))
    return features


def selector_urls(sitemap: bytes) -> list[str]:
    try:
        root = ET.fromstring(sitemap)
    except ET.ParseError:
        return FALLBACK_SELECTION_URLS
    urls: list[str] = []
    for element in root.iter():
        if not element.tag.endswith("loc") or not element.text:
            continue
        url = element.text.strip()
        path = urllib.parse.urlparse(url).path.strip("/").split("/")
        if len(path) == 4 and path[:2] == ["product-center", "microcontroller"]:
            urls.append(url)
    return sorted(set(urls or FALLBACK_SELECTION_URLS))


def purchase_cards(page: str) -> list[tuple[str, str]]:
    plain = clean_html(page)
    return sorted(set(
        (part.upper(), package)
        for part, package in re.findall(
            r"立即购买\s+(HPM[0-9A-Z]+?)\s+封装\s*[：:]\s*([A-Za-z0-9_×x.\-]+)",
            plain,
            re.I,
        )
    ))


def sdk_payload(payload: bytes) -> str:
    record = json.loads(payload.decode("utf-8"))
    return base64.b64decode(re.sub(r"\s+", "", record["content"])).decode("utf-8")


def sdk_features(source: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for token, (feature_type, name) in SDK_VENDOR_IPS.items():
        instances = re.findall(rf"(?m)^#define\s+HPM_{re.escape(token)}(?:\d+)?_BASE\b", source)
        if instances:
            add_feature(
                result,
                feature_type,
                name,
                len(instances),
                source_kind="hpmicro_sdk_soc_ip",
                source_parameter=f"HPM_{token}*_BASE",
            )
    return result


def device_row(
    model: str,
    row: dict[str, str],
    *,
    selection_url: str,
    selection_source_id: str,
    selection_version: str,
    series_url: str,
    series_source_id: str,
    series_text: str,
    observed_at: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    product_line = product_line_for(model)
    series = series_for(product_line)
    family = "HPM5" if product_line.startswith("HPM5") else "HPM6"
    core_text = column(row, "内核类型")
    core_count = 2 if "双核" in core_text else (1 if "单核" in core_text else None)
    max_clock_hz = frequency_hz(column(row, "最高主频")) or ""
    fpu = "DP_FPU" if "双精度浮点" in series_text else "unknown"
    dsp = "DSP" if "DSP" in series_text.upper() else "unknown"
    processor: dict[str, Any] = {
        "Darchitecture": "RISC-V",
        "Dcore": "HPMicro 32-bit RISC-V",
        "Dfpu": fpu,
        "Ddsp": dsp,
        "Dtz": "0",
        "DsourceDescription": core_text,
    }
    if core_count:
        processor["DcoreCount"] = str(core_count)
    if max_clock_hz:
        processor["Dclock"] = str(max_clock_hz)

    package = column(row, "封装形式")
    features = selector_features(row)
    documents = [
        {"title": f"HPMicro {series} official selector", "url": selection_url},
        {"title": f"HPMicro {series} series page", "url": series_url},
    ]
    device_id = f"hpmicro::{slug(model)}"
    source_ids = [selection_source_id, series_source_id]
    result = {
        "device_id": device_id,
        "product_line_id": f"hpmicro::{slug(product_line)}",
        "manufacturer": MANUFACTURER,
        "product_type": "general_purpose_mcu",
        "architecture_class": "RISC-V",
        "family": family,
        "series": series,
        "product_line": product_line,
        "device_name": model,
        "generic_device_name": product_line,
        "manufacturer_variant_code": model[len(product_line):],
        "processor_cores": json.dumps([processor], ensure_ascii=False, sort_keys=True),
        "max_clock_hz": max_clock_hz,
        "flash_bytes": memory_bytes(column(row, "flash"), default_unit="MB"),
        "ram_bytes": memory_bytes(column(row, "sram")),
        "package_types": package,
        "pin_counts": pin_count(package),
        "memory_regions_json": json.dumps([], ensure_ascii=False),
        "features_json": json.dumps(features, ensure_ascii=False, sort_keys=True),
        "documents_json": json.dumps(documents, ensure_ascii=False, sort_keys=True),
        "svd_files": "",
        "lifecycle": "active",
        "source_id": ";".join(source_ids),
        "source_url": selection_url,
        "source_version": selection_version,
        "observed_at": observed_at,
        "verification_status": "manufacturer_product_page",
    }
    auxiliary = {
        "temperature_range": column(row, "温度范围"),
        "packing_form": column(row, "包装方式"),
        "package": package,
    }
    return result, auxiliary


def mention_device(
    model: str,
    package: str,
    *,
    series_url: str,
    series_source_id: str,
    source_version: str,
    series_text: str,
    observed_at: str,
) -> dict[str, Any]:
    product_line = product_line_for(model)
    series = series_for(product_line)
    family = "HPM5" if product_line.startswith("HPM5") else "HPM6"
    processor = {
        "Darchitecture": "RISC-V",
        "Dcore": "HPMicro 32-bit RISC-V",
        "Dfpu": "DP_FPU" if "双精度浮点" in series_text else "unknown",
        "Ddsp": "DSP" if "DSP" in series_text.upper() else "unknown",
        "Dtz": "0",
        "DsourceDescription": "Official HPMicro series purchase card; exact core count not stated",
    }
    return {
        "device_id": f"hpmicro::{slug(model)}",
        "product_line_id": f"hpmicro::{slug(product_line)}",
        "manufacturer": MANUFACTURER,
        "product_type": "general_purpose_mcu",
        "architecture_class": "RISC-V",
        "family": family,
        "series": series,
        "product_line": product_line,
        "device_name": model,
        "generic_device_name": product_line,
        "manufacturer_variant_code": model[len(product_line):],
        "processor_cores": json.dumps([processor], ensure_ascii=False, sort_keys=True),
        "max_clock_hz": "",
        "flash_bytes": "",
        "ram_bytes": "",
        "package_types": package,
        "pin_counts": pin_count(package),
        "memory_regions_json": "[]",
        "features_json": "[]",
        "documents_json": json.dumps([{"title": f"HPMicro {series} series page", "url": series_url}], ensure_ascii=False),
        "svd_files": "",
        "lifecycle": "active",
        "source_id": series_source_id,
        "source_url": series_url,
        "source_version": source_version,
        "observed_at": observed_at,
        "verification_status": "manufacturer_product_page",
    }


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=root / "data" / "vendor-packs" / "hpmicro")
    parser.add_argument("--cache-dir", type=Path, default=root / "cache" / "hpmicro")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    observed = utc_now()
    fetcher = OfficialFetcher(
        cache_dir=args.cache_dir,
        allowed_hosts={"www.hpmicro.com", "api.github.com"},
        refresh=args.refresh,
        timeout=60,
        retries=4,
    )
    sources: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, str]] = []
    devices: dict[str, dict[str, Any]] = {}
    auxiliary: dict[str, dict[str, str]] = {}
    provenance: list[dict[str, Any]] = []
    purchase_records: list[tuple[str, str, str, str, str, str]] = []

    sitemap = fetch_snapshot(fetcher, SITEMAP_URL, cache_name="sitemap.xml", headers=HPM_WEB_HEADERS)
    sources["hpmicro:sitemap"] = {
        "source_id": "hpmicro:sitemap", "source_type": "manufacturer_sitemap",
        "publisher": "HPMicro", "title": "HPMicro official website sitemap",
        "url": SITEMAP_URL, "version": f"sha256:{sitemap.sha256}",
        "observed_at": sitemap.observed_at,
        "verification_scope": "Discovery of current official MCU selector and series pages.",
    }

    selection_pages = selector_urls(sitemap.payload)
    series_snapshots: dict[str, tuple[str, str, str, str, str]] = {}
    parsed_selection_pages = 0
    for selection_url in selection_pages:
        path = urllib.parse.urlparse(selection_url).path.strip("/").split("/")
        series_slug = path[2]
        leaf = path[3]
        series_url = f"{PRODUCT_ROOT}/{series_slug}"
        try:
            selection = fetch_snapshot(
                fetcher,
                selection_url,
                cache_name=f"selector-{series_slug}-{slug(leaf)}.html",
                headers=HPM_WEB_HEADERS,
            )
            page = selection.payload.decode("utf-8", errors="replace")
            _, rows = official_rows(page)
            selection_version = f"sha256:{selection.sha256}"
            compact_snapshot = args.cache_dir / f"selector-{series_slug}-{slug(leaf)}.json"
            if not rows and compact_snapshot.exists():
                compact_payload = compact_snapshot.read_bytes()
                _, rows = official_rows_json(compact_payload)
                selection_version = "official-dom-sha256:" + hashlib.sha256(compact_payload).hexdigest()
            if not rows:
                continue
            parsed_selection_pages += 1
            selection_source_id = f"hpmicro:selector:{series_slug}"
            sources[selection_source_id] = {
                "source_id": selection_source_id,
                "source_type": "manufacturer_product_selector_page",
                "publisher": "HPMicro",
                "title": f"HPMicro {series_slug.upper()} official selection table",
                "url": selection_url,
                "version": selection_version,
                "observed_at": selection.observed_at,
                "verification_scope": "Exact model rows and their per-row selector columns; no suffix expansion.",
            }
            if series_slug not in series_snapshots:
                overview = fetch_snapshot(
                    fetcher,
                    series_url,
                    cache_name=f"series-{series_slug}.html",
                    headers=HPM_WEB_HEADERS,
                )
                overview_text = clean_html(overview.payload.decode("utf-8", errors="replace"))
                series_source_id = f"hpmicro:series:{series_slug}"
                sources[series_source_id] = {
                    "source_id": series_source_id,
                    "source_type": "manufacturer_series_page",
                    "publisher": "HPMicro",
                    "title": f"HPMicro {series_slug.upper()} official series page",
                    "url": series_url,
                    "version": f"sha256:{overview.sha256}",
                    "observed_at": overview.observed_at,
                    "verification_scope": "Series architecture statements and exact purchase cards shown on the official page.",
                }
                series_snapshots[series_slug] = (
                    series_url,
                    series_source_id,
                    f"sha256:{overview.sha256}",
                    overview_text,
                    overview.payload.decode("utf-8", errors="replace"),
                )
            series_url, series_source_id, _, series_text, series_page = series_snapshots[series_slug]
            for raw in rows:
                model = raw.get("型号", "").upper()
                if not re.fullmatch(r"HPM[0-9A-Z]+", model):
                    errors.append({"scope": series_slug, "source_url": selection_url, "item": model, "error": "invalid exact model name"})
                    continue
                device, extra = device_row(
                    model,
                    raw,
                    selection_url=selection_url,
                    selection_source_id=selection_source_id,
                    selection_version=selection_version,
                    series_url=series_url,
                    series_source_id=series_source_id,
                    series_text=series_text,
                    observed_at=observed,
                )
                devices[device["device_id"]] = device
                auxiliary[device["device_id"]] = extra
                for field_name, value in raw.items():
                    if value not in (None, ""):
                        provenance.append({
                            "record_type": "device", "record_id": device["device_id"],
                            "field_name": field_name, "source_id": selection_source_id,
                            "source_url": selection_url, "source_path": f"selection column:{field_name}",
                            "source_value_json": json.dumps(value, ensure_ascii=False),
                            "observed_at": observed, "verification_status": "manufacturer_product_page",
                        })
            for part, package in purchase_cards(series_page):
                purchase_records.append((part, package, series_slug, series_url, series_source_id, series_snapshots[series_slug][2]))
        except Exception as exc:  # Keep other official series importable.
            errors.append({"scope": series_slug, "source_url": selection_url, "item": leaf, "error": repr(exc)})

    # Reproducible HPM SDK release metadata and exact-SoC vendor IP names.
    sdk_tag = ""
    sdk_release_url = f"{GITHUB_API}/releases/latest"
    try:
        release = fetcher.fetch(sdk_release_url, cache_name="hpm-sdk-latest-release.json")
        release_record = json.loads(release.payload.decode("utf-8"))
        sdk_tag = str(release_record.get("tag_name") or "")
        sources["hpmicro:sdk-release"] = {
            "source_id": "hpmicro:sdk-release", "source_type": "manufacturer_sdk_release",
            "publisher": "HPMicro", "title": "HPMicro HPM SDK latest official release",
            "url": "https://github.com/hpmicro/hpm_sdk/releases/latest",
            "version": sdk_tag or f"sha256:{release.sha256}", "observed_at": release.observed_at,
            "verification_scope": "SDK release version used for exact matching SoC IP headers.",
        }
    except Exception as exc:
        errors.append({"scope": "sdk", "source_url": sdk_release_url, "item": "latest-release", "error": repr(exc)})

    sdk_matches = 0
    if sdk_tag:
        for product_line, soc_path in SDK_SOC_PATHS.items():
            matched = [row for row in devices.values() if row["product_line"] == product_line]
            if not matched:
                continue
            api_url = f"{GITHUB_API}/contents/soc/{soc_path}/hpm_soc_ip.h?ref={urllib.parse.quote(sdk_tag)}"
            try:
                snapshot = fetcher.fetch(api_url, cache_name=f"sdk-{sdk_tag}-{product_line}-hpm_soc_ip.json")
                record = json.loads(snapshot.payload.decode("utf-8"))
                source = sdk_payload(snapshot.payload)
                source_id = f"hpmicro:sdk:{sdk_tag}:{product_line.lower()}"
                browser_url = str(record.get("html_url") or api_url)
                sources[source_id] = {
                    "source_id": source_id, "source_type": "manufacturer_sdk_device_header",
                    "publisher": "HPMicro", "title": f"HPM SDK {product_line} SoC IP header",
                    "url": browser_url, "version": f"{sdk_tag};git:{record.get('sha', '')}",
                    "observed_at": snapshot.observed_at,
                    "verification_scope": "Vendor IP block presence only for the exact matching SDK SoC definition.",
                }
                additions = sdk_features(source)
                if not additions:
                    continue
                for device in matched:
                    features = json.loads(device["features_json"] or "[]")
                    for addition in additions:
                        add_feature(
                            features,
                            addition["type"],
                            addition["name"],
                            int(addition["count"]),
                            source_kind="hpmicro_sdk_soc_ip",
                            source_parameter=addition.get("source_parameter", ""),
                        )
                    documents = json.loads(device["documents_json"] or "[]")
                    documents.append({"title": f"HPM SDK {product_line} SoC IP header", "url": browser_url})
                    device["features_json"] = json.dumps(features, ensure_ascii=False, sort_keys=True)
                    device["documents_json"] = json.dumps(documents, ensure_ascii=False, sort_keys=True)
                    device["source_id"] += ";" + source_id
                    device["svd_files"] = f"https://github.com/hpmicro/hpm_sdk/blob/{sdk_tag}/soc/{soc_path}/{product_line}_svd.xml"
                    sdk_matches += 1
            except Exception as exc:
                errors.append({"scope": "sdk", "source_url": api_url, "item": product_line, "error": repr(exc)})

    # Purchase cards use complete manufacturer strings.  The selector model is
    # the device variant; a final revision/grade digit remains a part suffix.
    parts: dict[str, dict[str, Any]] = {}
    purchase_only_devices = 0
    for part, package, series_slug, series_url, series_source_id, source_version in sorted(set(purchase_records)):
        candidates = [row for row in devices.values() if part.startswith(row["device_name"])]
        if candidates:
            device = max(candidates, key=lambda row: len(row["device_name"]))
        else:
            model = re.sub(r"(?<=[A-Z])\d$", "", part)
            _, _, _, series_text, _ = series_snapshots[series_slug]
            device = mention_device(
                model,
                package,
                series_url=series_url,
                series_source_id=series_source_id,
                source_version=source_version,
                series_text=series_text,
                observed_at=observed,
            )
            devices[device["device_id"]] = device
            auxiliary[device["device_id"]] = {"temperature_range": "", "packing_form": "", "package": package}
            purchase_only_devices += 1
        extra = auxiliary.get(device["device_id"], {})
        suffix = part[len(device["device_name"]):]
        part_id = f"hpmicro::{slug(part)}"
        parts[part_id] = {
            "orderable_part_id": part_id,
            "device_id": device["device_id"],
            "manufacturer": MANUFACTURER,
            "family": device["family"],
            "series": device["series"],
            "product_line": device["product_line"],
            "device_name": device["device_name"],
            "part_number": part,
            "manufacturer_suffix": suffix,
            "package_code": package,
            "temperature_grade_code": "",
            "packing_code": "",
            "package_name": package or extra.get("package", ""),
            "temperature_range": extra.get("temperature_range", ""),
            "packing_form": extra.get("packing_form", ""),
            "lifecycle": "active",
            "source_id": series_source_id,
            "source_url": series_url,
            "observed_at": observed,
            "verification_status": "manufacturer_product_page",
            "decode_status": "exact_official_purchase_card_no_suffix_semantic_inference",
        }

    lines: dict[str, dict[str, Any]] = {}
    for device in devices.values():
        line = lines.setdefault(device["product_line_id"], {
            "product_line_id": device["product_line_id"], "manufacturer": MANUFACTURER,
            "product_type": device["product_type"], "architecture_class": device["architecture_class"],
            "family": device["family"], "series": device["series"], "product_line": device["product_line"],
            "device_variant_count": 0, "source_ids": set(), "observed_at": observed,
            "verification_status": "manufacturer_product_page",
        })
        line["device_variant_count"] += 1
        line["source_ids"].update(device["source_id"].split(";"))
    product_lines = []
    for line in lines.values():
        line["source_ids"] = ";".join(sorted(line["source_ids"]))
        product_lines.append(line)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device_rows = sorted(devices.values(), key=lambda row: (row["series"], row["product_line"], row["device_name"]))
    part_rows = sorted(parts.values(), key=lambda row: row["part_number"])
    write_csv(args.output_dir / "device-variants.csv", DEVICE_FIELDS, device_rows)
    write_csv(args.output_dir / "orderable-parts.csv", PART_FIELDS, part_rows)
    write_csv(args.output_dir / "product-lines.csv", PRODUCT_LINE_FIELDS, sorted(product_lines, key=lambda row: (row["series"], row["product_line"])))
    write_csv(args.output_dir / "sources.csv", SOURCE_FIELDS, sorted(sources.values(), key=lambda row: row["source_id"]))
    write_csv(args.output_dir / "import-errors.csv", ERROR_FIELDS, errors)
    write_csv(args.output_dir / "field-provenance.csv", FIELD_PROVENANCE_FIELDS, provenance)

    field_coverage = Counter()
    for row in device_rows:
        for field in ("max_clock_hz", "flash_bytes", "ram_bytes", "package_types", "pin_counts"):
            if row.get(field) not in (None, ""):
                field_coverage[field] += 1
        features = json.loads(row["features_json"] or "[]")
        for feature_type in ("Timer", "ADCUnits", "DAC", "UART", "SPI", "I2C", "CAN", "USB", "ETH"):
            if any(item.get("type") == feature_type for item in features):
                field_coverage[feature_type] += 1

    report = {
        "status": "ok" if parsed_selection_pages and not [row for row in errors if row["scope"] != "sdk"] else "partial",
        "adapter": "import_hpmicro_official.py",
        "manufacturer": MANUFACTURER,
        "official_selection_pages_discovered": len(selection_pages),
        "official_selection_pages_parsed": parsed_selection_pages,
        "device_variants": len(device_rows),
        "selection_table_devices": len(device_rows) - purchase_only_devices,
        "purchase_only_devices": purchase_only_devices,
        "orderable_parts": len(part_rows),
        "product_lines": len(product_lines),
        "series": sorted({row["series"] for row in device_rows}),
        "sdk_release": sdk_tag,
        "sdk_exact_device_matches": sdk_matches,
        "field_coverage": dict(sorted(field_coverage.items())),
        "adc_semantics": "Official ADC 12-bit/16-bit columns are stored as converter units. ADC pin counts are not imported; channel counts remain unknown unless explicitly sourced.",
        "errors": len(errors),
        "completeness_claim": "Snapshot of every exact MCU row discoverable in the current official HPMicro sitemap selection tables plus exact official purchase-card strings. It is not a claim about unreleased, archived, or distributor-only parts.",
    }
    write_json(args.output_dir / "official-adapter-report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if parsed_selection_pages else 1


if __name__ == "__main__":
    raise SystemExit(main())

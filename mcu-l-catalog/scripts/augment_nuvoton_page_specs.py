#!/usr/bin/env python3
"""Extract explicit specifications from cached, exact Nuvoton product pages.

The Nuvoton product pages are model-bound by the existing official importer.
This pass only writes values stated in that exact page's key-feature text.  It
does not copy values from a family page or decode an ordering-code suffix.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def cache_name(url: str, suffix: str = ".html") -> str:
    path = url.split("?", 1)[0].rstrip("/").rsplit("/", 1)[-1] or "index"
    stem = re.sub(r"[^a-z0-9]+", "-", path.lower()).strip("-") or "index"
    return f"{stem}-{hashlib.sha256(url.encode()).hexdigest()[:14]}{suffix}"


def page_text(payload: str) -> str:
    value = re.sub(r"(?is)<script[^>]*>.*?</script>", " ", payload)
    value = re.sub(r"(?is)<style[^>]*>.*?</style>", " ", value)
    value = html.unescape(re.sub(r"<[^>]+>", " ", value))
    # Product pages use typographic dashes in facts such as ``32‑bit``.
    # Normalize separators only, so the exact-page parser can recognize the
    # stated bit width without changing numeric content.
    value = value.translate(str.maketrans({
        "\xa0": " ", "\u2010": "-", "\u2011": "-", "\u2013": "-",
        "\u2014": "-", "\u2212": "-",
    }))
    value = " ".join(value.split())
    # A handful of Nuvoton pages render three-digit quantities with an
    # inserted space (for example, ``1 10 I/O pins`` for 110 pins).
    return re.sub(r"(?<=\d)\s+(?=\d\d\s+(?:I/O|GPIO))", "", value)


def cached_page_index(pages_dir: Path) -> dict[str, Path]:
    """Map a cached official product page to the exact model in its title.

    Cache filenames intentionally only carry a URL hash.  The product title is
    therefore the reliable local identifier.  A title such as
    ``NUC120LE3DN - Nuvoton`` represents that exact orderable device page; a
    series landing page is excluded because its title does not end in the
    vendor suffix in this form.
    """
    index: dict[str, Path] = {}
    duplicates: set[str] = set()
    for path in sorted(pages_dir.glob("*.html")):
        try:
            payload = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        match = re.search(r"(?is)<title[^>]*>\s*([^<]+?)\s*-\s*Nuvoton\s*</title>", payload)
        if not match:
            continue
        model = " ".join(html.unescape(match.group(1)).split()).upper()
        # A part number has no whitespace and cannot be a series landing page.
        if not model or re.search(r"\s", model):
            continue
        if model in index:
            duplicates.add(model)
        else:
            index[model] = path
    for model in duplicates:
        index.pop(model, None)
    return index


def first_number(text: str, patterns: list[str], flags: int = re.I) -> float | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags)
        if not match:
            continue
        try:
            return float(match.group("value"))
        except (TypeError, ValueError):
            continue
    return None


NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}


def number_token(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return float(NUMBER_WORDS[value.lower()]) if value.lower() in NUMBER_WORDS else None


def whole(value: float | None) -> str:
    if value is None:
        return ""
    return str(int(value)) if value.is_integer() else str(value)


def feature(feature_type: str, name: str, count: float | None = None, *, bits: int | None = None) -> dict[str, str]:
    result: dict[str, str] = {
        "type": feature_type,
        "name": name,
        "source_kind": "nuvoton_official_exact_product_page",
        "verification_status": "official_exact_product_page",
    }
    if count is not None:
        result["count"] = whole(count)
        result["n"] = whole(count)
    if bits is not None:
        result["m"] = str(bits)
    return result


def normalize_core_name(value: str) -> str:
    value = re.sub(r"\s+", "", value.replace("®", ""))
    value = re.sub(r"^ArmCortex", "Cortex", value, flags=re.I)
    return value


def parse_specs(text: str) -> dict[str, Any]:
    specs: dict[str, Any] = {"features": []}
    core = re.search(r"\b(Cortex\s*[-®]?\s*M\d+[+]?|Arm\s+Cortex\s*[-®]?\s*M\d+[+]?)\s+(?:processor|core)", text, re.I)
    clock = first_number(text, [
        r"Max\s+frequency\s+of\s+(?P<value>[\d.]+)\s*MHz",
        r"(?P<value>[\d.]+)\s*MHz\s+maximum\s+frequency",
        r"(?P<value>[\d.]+)\s*MHz\s+CPU\s+frequency",
        r"runn?ing\s+up\s+to\s+(?P<value>[\d.]+)\s*MHz",
        r"(?:operating|core)\s+frequency\s+up\s+to\s+(?P<value>[\d.]+)\s*MHz",
        r"core[^.]{0,80}?up\s+to\s+(?P<value>[\d.]+)\s*MHz",
    ])
    voltage = re.search(r"(?:Operating\s+voltage\s*:\s*|Operating\s+voltage\s+from\s+|(?:Operating\s+)?Voltage\s+range\s*:\s*|operate\s+at\s+)(?P<min>[\d.]+)\s*V\s*(?:to|~|-)\s*(?P<max>[\d.]+)\s*V", text, re.I)
    flash_candidates = []
    composite_flash = re.search(r"(?P<main>[\d.]+)\s*KB\s*\+\s*(?P<extra>[\d.]+)\s*KB\s+Flash\s+Memory", text, re.I)
    if composite_flash:
        flash_candidates.append(float(composite_flash.group("main")) + float(composite_flash.group("extra")))
    for pattern in [
        r"(?P<value>[\d.]+)\s*K(?:B|bytes)\s+of\s+(?:embedded\s+)?flash\s+memory",
        r"(?P<value>[\d.]+)\s*K(?:B|bytes)\s+flash\s+memory",
        r"(?P<value>[\d.]+)\s*K(?:B|bytes)\s+embedded\s+program\s+Flash",
        r"(?P<value>[\d.]+)\s*K(?:B|bytes)\s+Flash\s+Memory",
    ]:
        flash_candidates.extend(float(m.group("value")) for m in re.finditer(pattern, text, re.I))
    flash = max(flash_candidates) if flash_candidates else None
    sram = first_number(text, [
        r"(?P<value>[\d.]+)\s*K(?:B|bytes)\s+of\s+(?:embedded\s+)?(?:S|s)ram",
        r"(?P<value>[\d.]+)\s*K(?:B|bytes)\s+of\s+SRAM",
        r"(?P<value>[\d.]+)\s*K(?:B|bytes)\s+SRAM",
    ])
    if core:
        specs["core"] = normalize_core_name(core.group(1))
    if clock is not None:
        specs["clock_hz"] = int(clock * 1_000_000)
    if voltage:
        specs["voltage"] = (float(voltage.group("min")), float(voltage.group("max")))
    if flash is not None:
        specs["flash_bytes"] = int(flash * 1024)
    if sram is not None:
        specs["ram_bytes"] = int(sram * 1024)

    # Engineering-memory facts are intentionally separate from the byte
    # totals.  They drive the detailed Flash/RAM section only when the exact
    # product page explicitly names the property.
    if re.search(r"(?:flash\s+memory[^.]{0,40}|flash[^.]{0,40}memory)dual[- ]?bank", text, re.I):
        specs["features"].append(feature("MemoryArchitecture", "Flash memory: dual-bank"))
    if re.search(r"flash\s+memory\s+with\s+(?:error\s+correction\s+code|ecc)", text, re.I):
        specs["features"].append(feature("MemoryArchitecture", "Flash memory with ECC"))
    if re.search(r"(?:sram|ram)[^.|]{0,80}\bwith\s+ecc\b", text, re.I):
        specs["features"].append(feature("MemoryArchitecture", "SRAM with ECC"))

    # Do not derive power figures from clock or package data.  These records
    # exist only when a current is explicitly published on the same model
    # page, including its named operating mode and stated conditions.
    for power in re.finditer(
        r"(?P<mode>normal\s+run\s+mode|normal\s+power[- ]?down\s+mode|"
        r"standby\s+power[- ]?down\s+mode|(?:RTC\s*\([^)]*\)))"
        r"[^:.]{0,60}:\s*(?P<current>[\d.]+\s*[munµμ]?A(?:\s*/\s*MHz)?)"
        r"(?:\s*(?P<condition>\([^)]*\)))?",
        text,
        re.I,
    ):
        detail = " ".join(value for value in (power.group("mode"), power.group("current"), power.group("condition") or "") if value)
        specs["features"].append(feature("Consumption", detail))

    adc = re.search(r"ADC\s*-\s*.*?Up\s+to\s+(?P<count>[\d.]+)\s+channels?.*?(?P<bits>8|10|12|14|16)-bit\s+resolution", text, re.I)
    if adc:
        specs["features"].append(feature("ADC", f"{adc.group('bits')}-bit ADC channels", float(adc.group("count")), bits=int(adc.group("bits"))))
    adc_rate = re.search(r"(?:ADC|SAR ADC).*?(?:up to\s+)?(?P<rate>[\d.]+)\s*(?P<unit>G|M|K)?SPS", text, re.I)
    if adc_rate:
        suffix = (adc_rate.group("unit") or "").upper()
        scale = {"G": 1_000_000_000, "M": 1_000_000, "K": 1_000}.get(suffix, 1)
        specs["features"].append(feature("ADCPerformance", f"ADC maximum {adc_rate.group('rate')} {suffix}SPS"))
        specs["adc_rate_hz"] = int(float(adc_rate.group("rate")) * scale)

    # Newer NuMicro pages use prose such as "One 12-bit SAR ADC with up to
    # 16 channels" or "3 units, 32 channels".  Keep converter units and
    # channel capacity as separate records.
    adc_block = re.search(r"(?P<unit>one|two|three|four|\d+)\s+(?:12|10|8|14|16)-bit\s+(?:SAR\s+)?ADC\s+with\s+(?:up\s+to\s+)?(?P<channels>\d+)\s+channels?", text, re.I)
    if adc_block:
        unit_count = number_token(adc_block.group("unit")); channel_count = number_token(adc_block.group("channels"))
        bits_match = re.search(r"(8|10|12|14|16)-bit", adc_block.group(0), re.I)
        bits = int(bits_match.group(1)) if bits_match else None
        specs["features"].append(feature("ADCUnits", "ADC converter units", unit_count, bits=bits))
        specs["features"].append(feature("ADC", f"{bits or ''}-bit ADC channels".strip(), channel_count, bits=bits))
    adc_units_channels = re.search(r"(?P<units>\d+)\s+units?\s*,\s*(?P<channels>\d+)\s+channels?", text, re.I)
    if adc_units_channels:
        specs["features"].append(feature("ADCUnits", "ADC converter units", number_token(adc_units_channels.group("units"))))
        specs["features"].append(feature("ADC", "ADC channels", number_token(adc_units_channels.group("channels"))))
    adc_single = re.search(r"(?P<units>one|two|three|four|\d+)\s+sets?\s+of\s+(?P<bits>8|10|12|14|16)-bit(?:,\s*)?(?:(?P<channels>\d+)\s+channel(?:s)?[^.]{0,30}?)?ADC", text, re.I)
    if adc_single:
        specs["features"].append(feature("ADCUnits", "ADC converter units", number_token(adc_single.group("units")), bits=int(adc_single.group("bits"))))
        if adc_single.group("channels"):
            specs["features"].append(feature("ADC", f"{adc_single.group('bits')}-bit ADC channels", number_token(adc_single.group("channels")), bits=int(adc_single.group("bits"))))
    adc_set_only = re.search(r"(?P<units>one|two|three|four|\d+)\s+sets?\s+of\s+(?P<bits>8|10|12|14|16)-bit[^.]{0,25}?ADC", text, re.I)
    if adc_set_only and not adc_single:
        specs["features"].append(feature("ADCUnits", "ADC converter units", number_token(adc_set_only.group("units")), bits=int(adc_set_only.group("bits"))))

    # Current M33 pages may enumerate each ADC as ``1 set of 12-bit,
    # 24-channel SAR ADC``.  A converter unit and its available channels are
    # two different data points and remain separate in the directory.
    for adc_detail in re.finditer(
        r"(?P<units>one|two|three|four|\d+)\s+sets?\s+of\s+"
        r"(?P<bits>8|10|12|14|16)-bit\s*,?\s*(?P<channels>\d+)\s*[- ]channel"
        r"(?:\s*,?\s*[\d.]+\s*(?:G|M|K)?sps)?\s*(?:SAR\s+)?ADC",
        text,
        re.I,
    ):
        units = number_token(adc_detail.group("units"))
        channels = number_token(adc_detail.group("channels"))
        bits = int(adc_detail.group("bits"))
        specs["features"].append(feature("ADCUnits", f"{bits}-bit ADC converter units", units, bits=bits))
        specs["features"].append(feature("ADC", f"{bits}-bit ADC channels", channels, bits=bits))

    patterns = [
        ("SPI", r"(?:Up\s+to\s+|(?P<set>\d+)\s+set(?:s)?\s+of\s+|(?P<set2>\d+)\s+set(?:s)?\s+)(?P<value>[\d.]+)?\s*SPI(?:s|\s+interfaces?)?"),
        ("I2C", r"(?:Up\s+to\s+|(?P<set>\d+)\s+set(?:s)?\s+of\s+|(?P<set2>\d+)\s+set(?:s)?\s+)(?P<value>[\d.]+)?\s*I(?:2C|²C)(?:s|\s+interfaces?)?"),
        ("UART", r"(?:Up\s+to\s+|(?P<set>\d+)\s+set(?:s)?\s+of\s+|(?P<set2>\d+)\s+set(?:s)?\s+)(?P<value>[\d.]+)?\s*UART(?:s|\s+interfaces?)?"),
        ("CAN", r"(?:Up\s+to\s+|(?P<set>\d+)\s+set(?:s)?\s+of\s+|(?P<set2>\d+)\s+set(?:s)?\s+)(?P<value>[\d.]+)?\s*(?:CAN|CAN\s+FD)(?:s|\s+controllers?)?"),
        ("USB", r"(?:Up\s+to\s+|(?P<set>\d+)\s+set(?:s)?\s+of\s+|(?P<set2>\d+)\s+set(?:s)?\s+)(?P<value>[\d.]+)?\s*USB(?:\s+2\.0)?"),
        ("PWM", r"(?:Up\s+to\s+)?(?P<value>[\d.]+)[ -](?:channel|channels)\s+PWM"),
        ("Timer", r"(?:Up\s+to\s+|(?P<set>\d+)\s+set(?:s)?\s+of\s+|(?P<set2>\d+)\s+set(?:s)?\s+)(?P<value>[\d.]+)?\s*(?:general-purpose\s+)?(?:\d+-bit\s+)?(?:timer|timers)"),
        ("IOs", r"(?:Up\s+to\s+)?(?P<value>[\d.]+)\s+I/O\s+pins"),
        ("IOs", r"(?:Up\s+to\s+)?(?P<value>[\d.]+)\s+GPIO(?:\s+pins?)?"),
    ]
    for kind, pattern in patterns:
        value = first_number(text, [pattern])
        if value is None:
            match = re.search(pattern, text, re.I)
            if match:
                for group in ("set", "set2"):
                    if match.groupdict().get(group):
                        value = number_token(match.group(group)); break
                if value is None and match.groupdict().get("value"):
                    value = number_token(match.group("value"))
        if value is not None:
            bits = None
            if kind == "Timer":
                width = re.search(r"(?P<bits>8|16|24|32)-bit\s+timers?", text, re.I)
                bits = int(width.group("bits")) if width else None
            name = f"Nuvoton {kind} instances"
            if kind == "CAN" and re.search(r"CAN[- ]?FD", text, re.I):
                name = "Nuvoton CAN FD instances"
            specs["features"].append(feature(kind, name, value, bits=bits))

    # Interface summaries commonly spell quantities out (for example, "up to
    # five UART interfaces") and may include "sets of" between the quantity
    # and peripheral name.
    word_patterns = {
        "UART": r"(?:Up\s+to\s+)?(?P<value>one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+(?:sets?\s+of\s+)?UART",
        "SPI": r"(?:Up\s+to\s+)?(?P<value>one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+(?:sets?\s+of\s+)?SPI",
        "I2C": r"(?:Up\s+to\s+)?(?P<value>one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+(?:sets?\s+of\s+)?I(?:2C|²C)",
        "CAN": r"(?:Up\s+to\s+)?(?P<value>one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+(?:sets?\s+of\s+)?CAN(?:-FD)?",
    }
    for kind, pattern in word_patterns.items():
        match = re.search(pattern, text, re.I)
        if match:
            value = number_token(match.group("value"))
            if value is not None:
                name = f"Nuvoton {kind} instances"
                if kind == "CAN" and re.search(r"CAN[- ]?FD", text, re.I):
                    name = "Nuvoton CAN FD instances"
                specs["features"].append(feature(kind, name, value))

    # Explicit timer forms found on automotive and motor-control pages.
    timer_forms = [
        (r"(?P<value>one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+sets?\s+of\s+(?P<bits>8|16|24|32)-bit\s+timers?", "Timer"),
        (r"(?P<value>one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+(?P<bits>8|16|24|32)-bit\s+timers?", "Timer"),
        (r"(?:General-purpose|general purpose)\s+(?P<bits>8|16|24|32)-bit\s+timers?\s*:\s*(?P<value>\d+)\s+(?:channels?|units?)", "Timer"),
        (r"(?P<value>one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+sets?\s+of\s+(?P<bits>8|16|24|32)-bit\s+timer", "Timer"),
    ]
    for pattern, kind in timer_forms:
        match = re.search(pattern, text, re.I)
        if match:
            value = number_token(match.group("value")); bits = int(match.group("bits"))
            if value is not None:
                specs["features"].append(feature(kind, f"Nuvoton {kind} instances", value, bits=bits))
                break

    spi_speed = first_number(text, [r"SPI(?:s)?\s*\([^)]*?up\s+to\s+(?P<value>[\d.]+)\s*MHz"])
    i2c_speed = first_number(text, [r"I(?:2C|²C)(?:s)?\s*\([^)]*?up\s+to\s+(?P<value>[\d.]+)\s*kHz"])
    uart_speed = first_number(text, [r"UART(?:s)?\s*\([^)]*?up\s+to\s+(?P<value>[\d.]+)\s*Mbps"])
    if spi_speed is not None:
        specs["features"].append(feature("IOSpeed", f"SPI maximum {whole(spi_speed)} MHz"))
    if i2c_speed is not None:
        specs["features"].append(feature("IOSpeed", f"I2C maximum {whole(i2c_speed)} kHz"))
    if uart_speed is not None:
        specs["features"].append(feature("IOSpeed", f"UART maximum {whole(uart_speed)} Mbps"))
    # Keep one exact-page quantity per peripheral label.  The same page often
    # repeats its feature list in an overview and a detailed section; taking
    # the largest explicitly stated count avoids double-counting while
    # preserving the page's upper bound.
    consolidated: dict[tuple[str, str], dict[str, str]] = {}
    for item in specs["features"]:
        key = (item.get("type", ""), item.get("name", ""))
        old = consolidated.get(key)
        if old is None or (number_token(item.get("count")) or -1) > (number_token(old.get("count")) or -1):
            consolidated[key] = item
    specs["features"] = list(consolidated.values())
    return specs


def add_features(row: dict[str, str], additions: list[dict[str, str]]) -> int:
    try:
        current = json.loads(row.get("features_json") or "[]")
    except json.JSONDecodeError:
        current = []
    if not isinstance(current, list):
        current = []
    keys = {(item.get("type"), item.get("name"), item.get("source_kind")) for item in current if isinstance(item, dict)}
    changed = 0
    for item in additions:
        key = (item["type"], item["name"], item["source_kind"])
        if key not in keys:
            current.append(item); keys.add(key); changed += 1
    if changed:
        row["features_json"] = json.dumps(current, ensure_ascii=False, sort_keys=True)
    return changed


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=root / "data" / "combined")
    parser.add_argument("--cache-dir", type=Path, default=root / "cache" / "nuvoton-official")
    args = parser.parse_args()
    data = args.data_dir.resolve(); cache = args.cache_dir.resolve()
    devices = read_csv(data / "device-variants.csv")
    page_by_model = cached_page_index(cache / "pages")
    processed = updated = feature_updates = 0
    errors: list[dict[str, str]] = []
    for row in devices:
        if row.get("manufacturer") != "Nuvoton":
            continue
        model = row.get("device_name", "").upper()
        path = page_by_model.get(model)
        if path is None:
            continue
        try:
            specs = parse_specs(page_text(path.read_text(encoding="utf-8", errors="replace")))
            processed += 1
            # Normalize values imported from older passes as well as the page
            # value, so FPU and core ranking use the canonical Cortex-M name.
            try:
                existing_processors = json.loads(row.get("processor_cores") or "[]")
            except json.JSONDecodeError:
                existing_processors = []
            if isinstance(existing_processors, list):
                for processor in existing_processors:
                    if isinstance(processor, dict) and processor.get("Dcore"):
                        processor["Dcore"] = normalize_core_name(str(processor["Dcore"]))
                row["processor_cores"] = json.dumps(existing_processors, ensure_ascii=False, sort_keys=True)
            # Rebuild this evidence source on every run.  This prevents an
            # earlier, less-specific parser result from surviving alongside a
            # corrected exact-page value and being double-counted downstream.
            try:
                existing_features = json.loads(row.get("features_json") or "[]")
            except json.JSONDecodeError:
                existing_features = []
            if isinstance(existing_features, list):
                row["features_json"] = json.dumps(
                    [item for item in existing_features if item.get("source_kind") != "nuvoton_official_exact_product_page"],
                    ensure_ascii=False,
                    sort_keys=True,
                )
            before = (row.get("processor_cores"), row.get("max_clock_hz"), row.get("flash_bytes"), row.get("ram_bytes"))
            if specs.get("core"):
                try: processors = json.loads(row.get("processor_cores") or "[]")
                except json.JSONDecodeError: processors = []
                if not isinstance(processors, list) or not processors: processors = [{}]
                processors[0]["Dcore"] = specs["core"]; processors[0]["Dclock"] = str(specs.get("clock_hz") or row.get("max_clock_hz") or "")
                processors[0].setdefault("DsourceKind", "nuvoton_official_exact_product_page")
                row["processor_cores"] = json.dumps(processors, ensure_ascii=False, sort_keys=True)
            for key in ("clock_hz", "flash_bytes", "ram_bytes"):
                if specs.get(key) is not None:
                    row[{"clock_hz": "max_clock_hz", "flash_bytes": "flash_bytes", "ram_bytes": "ram_bytes"}[key]] = str(specs[key])
            if specs.get("voltage"):
                lo, hi = specs["voltage"]
                voltage_feature = feature("VCC", f"Operating voltage: {lo:g}-{hi:g} V")
                voltage_feature.update({"n": f"{lo:g}", "m": f"{hi:g}"})
                add_features(row, [voltage_feature])
            feature_updates += add_features(row, specs["features"])
            if specs.get("adc_rate_hz"):
                feature_updates += add_features(row, [feature("ADCPerformance", f"ADC maximum {specs['adc_rate_hz']} SPS")])
            if before != (row.get("processor_cores"), row.get("max_clock_hz"), row.get("flash_bytes"), row.get("ram_bytes")):
                updated += 1
        except Exception as exc:
            errors.append({"device_name": model, "error": f"{type(exc).__name__}: {exc}"})
    write_csv(data / "device-variants.csv", devices)
    report = {"generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(), "cached_exact_pages": len(page_by_model), "pages_processed": processed, "device_rows_updated": updated, "features_added": feature_updates, "errors": errors}
    (data / "nuvoton-page-spec-augmentation-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

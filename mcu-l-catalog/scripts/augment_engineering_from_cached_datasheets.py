#!/usr/bin/env python3
"""Extract engineering parameters from cached, manufacturer-published datasheets.

The catalog intentionally keeps this pass narrow.  It consumes only text
already extracted from official PDFs and assigns a document to a product line
only when its normalized filename starts with that official product-line ID.
It never fills an unavailable value from a neighbouring family.  The derived
records are qualitative performance/architecture evidence, rather than an
attempt to replace a device's exact peripheral inventory.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SOURCES = (
    ("Artery", "artery", "artery-documents/pdf", r"ds-(at32[a-z0-9]+)\.pdf\.txt$"),
    ("GigaDevice", "gigadevice", "gigadevice-datasheets/pdf", r"(gd32[a-z0-9]+)-datasheet-.*\.pdf\.txt$"),
    ("STMicroelectronics", "st", "st-datasheets/pdf", r"(stm32[a-z0-9]+)-[a-f0-9]+\.txt$"),
    ("Allwinner", "allwinner", "allwinner", r"(xr\d+)[^/]*\.txt$"),
)

OFFICIAL_DOCUMENT_URLS = {
    ("STMicroelectronics", "STM32F429"): (
        "STM32F427xx / STM32F429xx Datasheet",
        "https://www.st.com/resource/en/datasheet/stm32f429zi.pdf",
    ),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def source_id_for(vendor: str, product_line: str, filename: str) -> str:
    safe = re.sub(r"[^a-z0-9]+", "-", filename.lower()).strip("-")
    return f"{vendor}:datasheet-engineering:{product_line.lower()}:{safe}"


def product_line_for(stem: str, product_lines: set[str]) -> str | None:
    upper = stem.upper()
    candidates = [line for line in product_lines if upper.startswith(line.upper())]
    return max(candidates, key=len) if candidates else None


def compact_text(value: str) -> str:
    return " ".join(value.replace("\u00d7", "x").replace("\u2011", "-").split())


def head_text(path: Path) -> str:
    # Feature summaries are at the front of all three vendors' datasheets.
    # Avoid characterisation tables later in the document, where a number can
    # describe a test point instead of the converter's advertised maximum.
    raw = path.read_text(encoding="utf-8", errors="ignore")
    return compact_text(raw[:24_000])


def rate_hz(raw: str, unit: str) -> int:
    unit = unit.lower().replace(" ", "")
    factor = {"gsps": 1_000_000_000, "g/s": 1_000_000_000, "msps": 1_000_000,
              "m/s": 1_000_000, "ksps": 1_000, "k/s": 1_000, "sps": 1}.get(unit)
    if factor is None:
        raise ValueError(f"unsupported sample-rate unit: {unit}")
    return round(float(raw) * factor)


def engineering_feature(kind: str, name: str, source_document_id: str, **extra: str) -> dict[str, str]:
    record = {
        "type": kind,
        "name": name,
        "source_kind": "manufacturer_datasheet",
        "source_document_id": source_document_id,
        "verification_status": "manufacturer_datasheet",
    }
    record.update(extra)
    return record


def first_sentence(text: str, match: re.Match[str], radius: int = 260) -> str:
    start = max(0, text.rfind(".", max(0, match.start() - radius)) + 1)
    end_candidates = [position for position in (text.find(".", match.end()), text.find("\u2022", match.end()), text.find("\n", match.end())) if position >= 0]
    end = min(end_candidates) if end_candidates else min(len(text), match.end() + radius)
    return compact_text(text[start:end])


def snippet(text: str, match: re.Match[str], before: int = 80, after: int = 180) -> str:
    """Return an evidence fragment without crossing a feature-list bullet."""
    start = max(0, match.start() - before)
    end = min(len(text), match.end() + after)
    value = text[start:end]
    for delimiter in ("\u2022", "\u25a0", "\uf06e"):
        if delimiter in value:
            # Preserve the portion containing the actual match.
            relative = match.start() - start
            left = value.rfind(delimiter, 0, relative)
            right = value.find(delimiter, relative)
            value = value[left + 1:right if right >= 0 else len(value)]
    return compact_text(value).lstrip(" :;,-\u2013\u2014")


def parse_adc(text: str) -> list[str]:
    """Return only explicit ADC performance clauses.

    A count is deliberately retained as prose.  Inserting it as an ADCUnits
    count would double count existing exact CubeMX/ATDF inventories.
    """
    results: list[str] = []
    patterns = (
        re.compile(
            r"(?:up\s+to\s+)?(?P<units>\d+)\s*[x*]\s*(?P<bits>\d+)\s*-?bit"
            r"[^.\u2022]{0,100}?(?P<rate>\d+(?:\.\d+)?)\s*(?P<unit>G(?:SPS|/s)|M(?:SPS|/s)|K(?:SPS|/s)|SPS)"
            r"[^.\u2022]{0,100}?(?:A\s*/\s*D|ADC|analog[- ]to[- ]digital)", re.I,
        ),
        re.compile(
            r"(?P<bits>\d+)\s*-?bit\s*,?\s*(?P<time>\d+(?:\.\d+)?)\s*(?P<timeunit>[mun\u03bc\u00b5]s)\s*"
            r"(?:A\s*/\s*D|ADC|analog[- ]to[- ]digital)", re.I,
        ),
        # Several vendor manuals state converter capability as a channel
        # count and resolution without publishing a sample-rate figure.
        re.compile(
            r"(?P<channels>\d+)\s+channels?\s+(?P<bits>\d+)\s*-?bit(?:\s+accuracy)?\s+ADC\b", re.I,
        ),
    )
    for pattern in patterns:
        for match in pattern.finditer(text):
            sentence = snippet(text, match)
            if "time" in match.groupdict() and match.group("time"):
                unit = match.group("timeunit").lower().replace("\u03bc", "u").replace("\u00b5", "u")
                seconds = float(match.group("time")) * {"ms": 1e-3, "us": 1e-6, "ns": 1e-9}[unit]
                rate = 1 / seconds
                summary = f"ADC: {match.group('bits')}-bit, conversion time {match.group('time')} {unit} (up to {rate / 1_000_000:g} MSPS); {sentence}"
            elif "rate" in match.groupdict() and match.group("rate"):
                rate = rate_hz(match.group("rate"), match.group("unit"))
                units = match.group("units")
                summary = f"ADC: {units} converter unit(s), {match.group('bits')}-bit, up to {rate / 1_000_000:g} MSPS; {sentence}"
            else:
                summary = f"ADC: {match.group('channels')} channel(s), {match.group('bits')}-bit; {sentence}"
            if summary not in results:
                results.append(summary)
    return results[:2]


def parse_dac(text: str) -> list[str]:
    results: list[str] = []
    pattern = re.compile(
        r"(?:up\s+to\s+)?(?P<units>\d+)\s*[x*]\s*(?P<bits>\d+)\s*-?bit\s*"
        r"(?:D\s*/\s*A|DAC|digital[- ]to[- ]analog)(?:\s+converter)?", re.I,
    )
    for match in pattern.finditer(text):
        sentence = snippet(text, match)
        summary = f"DAC: {match.group('units')} converter unit(s), {match.group('bits')}-bit; {sentence}"
        if summary not in results:
            results.append(summary)
    return results[:2]


def parse_timers(text: str) -> list[str]:
    results: list[str] = []
    # Do not allow a headline such as "32-bit MCU ... timers" to match. A
    # bit width must be adjacent to an explicitly named timer/counter block.
    patterns = (
        re.compile(
            r"(?:\d+\s*[x*]\s*)?\d+\s*-?\s*bit(?:\s+[A-Za-z/-]+){0,8}\s+"
            r"(?:timers?|tmrs?|counters?)\b", re.I,
        ),
        re.compile(
            r"\b(?:timers?|tmrs?|counters?)\b(?:\s+[A-Za-z/-]+){0,6}\s+"
            r"(?:\d+\s*[x*]\s*)?\d+\s*-?\s*bit", re.I,
        ),
    )
    for pattern in patterns:
        for match in pattern.finditer(text):
            candidate = snippet(text, match, 30, 220)
            summary = f"Timer architecture: {candidate}"
            if summary not in results:
                results.append(summary)
    return results[:3]


def parse_io_speed(text: str) -> list[str]:
    results: list[str] = []
    # Keep the GPIO token and frequency in the same short source fragment.
    # MCU clock or SPI frequency therefore cannot become an IO-speed claim.
    pattern = re.compile(
        r"(?:GPIO(?:s)?|fast\s+I/?O(?:s)?|general[- ]purpose\s+I/?O)[^.\u2022]{0,38}?"
        r"(?:up\s+to\s+|output\s+speed\s*[:=]?\s*|toggle\s+(?:rate|speed)\s*[:=]?\s*|(?:operating\s+)?frequency\s*[:=]?\s*|speed\s*[:=]?\s*)"
        r"[^.\u2022]{0,20}?(?P<speed>\d+(?:\.\d+)?)\s*(?P<unit>MHz)", re.I,
    )
    for match in pattern.finditer(text):
        sentence = snippet(text, match, 80, 140)
        summary = f"GPIO / I/O maximum speed: {match.group('speed')} {match.group('unit')}; {sentence}"
        if summary not in results:
            results.append(summary)
    return results[:2]


def parse_memory(text: str) -> list[str]:
    results: list[str] = []
    candidates = (
        ("RAM ECC", r"(?:SRAM|RAM)[^.\u2022]{0,90}\bECC\b|\bECC\b[^.\u2022]{0,90}(?:SRAM|RAM)"),
        ("RAM parity", r"(?:SRAM|RAM)[^.\u2022]{0,90}\bparity(?:\s+check)?\b|\bparity(?:\s+check)?\b[^.\u2022]{0,90}(?:SRAM|RAM)"),
        ("Flash ECC", r"(?:Flash|NVM)[^.\u2022]{0,90}\bECC\b|\bECC\b[^.\u2022]{0,90}(?:Flash|NVM)"),
        ("Dual-bank Flash", r"(?:dual|two)[ -]?bank[^.\u2022]{0,100}(?:Flash|flash)|(?:Flash|flash)[^.\u2022]{0,100}(?:dual|two)[ -]?bank"),
        ("Flash zero-wait", r"(?:Flash|flash)[^.\u2022]{0,100}(?:zero|0)[ -]?wait|(?:zero|0)[ -]?wait[^.\u2022]{0,100}(?:Flash|flash)"),
        ("Cache", r"(?:instruction|data|I-|D-)?\s*cache(?:\s+controller)?[^.\u2022]{0,100}"),
        ("TCM", r"(?:ITCM|DTCM|TCM)[^.\u2022]{0,120}"),
        ("Dedicated RAM", r"(?:dedicated|private|exclusive|per-core)[^.\u2022]{0,120}(?:SRAM|RAM|memory)"),
    )
    for label, expression in candidates:
        match = re.search(expression, text, re.I)
        if match:
            summary = f"{label}: {snippet(text, match)}"
            if summary not in results:
                results.append(summary)
    return results


def parse_voltage(text: str) -> list[dict[str, str]]:
    """Return explicit supply ranges, excluding absolute-maximum tables."""
    results: list[dict[str, str]] = []
    patterns = (
        re.compile(
            r"(?P<min>\d+(?:\.\d+)?)\s*(?:-|–|—|~|～)\s*(?P<max>\d+(?:\.\d+)?)\s*V"
            r"[^.\u2022]{0,55}(?:power\s+supply|supply\s+input|operating\s+condition|VCC|VDD)", re.I,
        ),
        re.compile(
            r"(?:power\s+supply|supply\s+input|operating\s+condition|VCC|VDD)[^.\u2022]{0,55}?"
            r"(?P<min>\d+(?:\.\d+)?)\s*(?:-|–|—|~|～)\s*(?P<max>\d+(?:\.\d+)?)\s*V", re.I,
        ),
    )
    for pattern in patterns:
        for match in pattern.finditer(text):
            lower = float(match.group("min"))
            upper = float(match.group("max"))
            if not (0 < lower <= upper <= 20):
                continue
            evidence = snippet(text, match, 90, 160)
            name = f"Operating supply range: {lower:g}-{upper:g} V; {evidence}"
            if not any(item["name"] == name for item in results):
                results.append({"name": name, "min": f"{lower:g}", "max": f"{upper:g}"})
    return results[:2]


def parse_features(text: str, source_document_id: str) -> list[dict[str, str]]:
    values: list[tuple[str, str]] = []
    values.extend(("ADCPerformance", item) for item in parse_adc(text))
    values.extend(("DACPerformance", item) for item in parse_dac(text))
    values.extend(("TimerArchitecture", item) for item in parse_timers(text))
    values.extend(("IOSpeed", item) for item in parse_io_speed(text))
    values.extend(("MemoryArchitecture", item) for item in parse_memory(text))
    records = [engineering_feature(kind, name, source_document_id) for kind, name in values]
    records.extend(
        engineering_feature("VCC", item["name"], source_document_id, n=item["min"], m=item["max"])
        for item in parse_voltage(text)
    )
    return records


def add_features(rows: list[dict[str, str]], manufacturer: str, product_line: str, features: list[dict[str, str]]) -> int:
    added = 0
    for row in rows:
        if row.get("manufacturer") != manufacturer or row.get("product_line") != product_line:
            continue
        existing = json.loads(row.get("features_json") or "[]")
        existing_keys = {(str(item.get("type") or ""), str(item.get("name") or "")) for item in existing if isinstance(item, dict)}
        for feature in features:
            key = (feature["type"], feature["name"])
            if key not in existing_keys:
                existing.append(feature)
                existing_keys.add(key)
                added += 1
        row["features_json"] = json.dumps(existing, ensure_ascii=False)
    return added


def add_document(rows: list[dict[str, str]], manufacturer: str, product_line: str) -> int:
    document = OFFICIAL_DOCUMENT_URLS.get((manufacturer, product_line))
    if not document:
        return 0
    title, url = document
    changed = 0
    for row in rows:
        if row.get("manufacturer") != manufacturer or row.get("product_line") != product_line:
            continue
        records = json.loads(row.get("documents_json") or "[]")
        if any(isinstance(item, dict) and str(item.get("url") or "") == url for item in records):
            continue
        records.append({
            "title": title,
            "url": url,
            "kind": "datasheet",
            "verification_status": "manufacturer_datasheet",
        })
        row["documents_json"] = json.dumps(records, ensure_ascii=False)
        changed += 1
    return changed


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=root / "data" / "combined")
    parser.add_argument("--cache-dir", type=Path, default=root / "cache")
    args = parser.parse_args()

    device_path = args.data_dir / "device-variants.csv"
    rows = read_csv(device_path)
    if not rows:
        raise SystemExit(f"no devices found: {device_path}")
    fields = list(rows[0].keys())
    product_lines = {row.get("product_line", "") for row in rows if row.get("product_line")}
    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "documents": [], "features_added": 0,
    }

    # Remove this pass's prior output before regenerating.  This makes parser
    # updates idempotent and avoids preserving evidence from an old rule.
    for row in rows:
        existing = json.loads(row.get("features_json") or "[]")
        row["features_json"] = json.dumps([
            item for item in existing
            if not (
                isinstance(item, dict)
                and ":datasheet-engineering:" in str(item.get("source_document_id") or "")
            )
        ], ensure_ascii=False)

    for manufacturer, vendor, relative_cache, filename_pattern in SOURCES:
        cache = args.cache_dir / relative_cache
        if not cache.exists():
            continue
        for path in sorted(cache.glob("*.txt")):
            match = re.match(filename_pattern, path.name, re.I)
            if not match:
                continue
            product_line = product_line_for(match.group(1), product_lines)
            if not product_line:
                continue
            source_document_id = source_id_for(vendor, product_line, path.name)
            features = parse_features(head_text(path), source_document_id)
            documents_added = add_document(rows, manufacturer, product_line)
            if not features and not documents_added:
                continue
            added = add_features(rows, manufacturer, product_line, features)
            if added:
                report["documents"].append({
                    "manufacturer": manufacturer, "product_line": product_line,
                    "file": path.name, "features": len(features), "features_added": added,
                    "documents_added": documents_added,
                })
                report["features_added"] += added

    write_csv(device_path, fields, rows)
    (args.data_dir / "engineering-datasheet-augmentation-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"documents": len(report["documents"]), "features_added": report["features_added"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

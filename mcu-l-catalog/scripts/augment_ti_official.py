#!/usr/bin/env python3
"""Augment TI devices from exact official product pages.

Package values are accepted only from explicit ``N-pin PACKAGE (code)`` text
on the product page for the exact catalog model.  The stable TI GPN document
URL is added only after it resolves to an HTTP 200 PDF.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = "https://www.ti.com"
USER_AGENT = "MCUS/1.0 (+https://github.com/new-bmp/MCUS; TI official importer)"
PACKAGE_RE = re.compile(
    r"\b(?P<pins>\d{1,3})\s*-\s*pin\s+"
    r"(?P<package>[A-Z][A-Z0-9-]{1,20})\s*\((?P<code>[A-Z0-9-]{1,12})\)",
    re.I,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


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


def cache_name(model: str) -> str:
    stem = re.sub(r"[^a-z0-9]+", "-", model.lower()).strip("-")
    return f"{stem}-{hashlib.sha256(model.encode()).hexdigest()[:12]}.html"


def request(url: str, timeout: float, method: str = "GET") -> tuple[bytes, str, str]:
    req = urllib.request.Request(
        url,
        method=method,
        headers={"User-Agent": USER_AGENT, "Accept": "*/*", "Connection": "close"},
    )
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                payload = response.read() if method != "HEAD" else b""
                return payload, response.geturl(), response.headers.get("Content-Type", "")
        except Exception as exc:  # pragma: no cover - remote availability
            last_error = exc
            if attempt < 2:
                time.sleep(1.0 * (2**attempt))
    assert last_error is not None
    raise last_error


def product_evidence(model: str, cache_dir: Path, timeout: float, refresh: bool) -> dict[str, Any]:
    product_url = f"{ROOT}/product/{model}"
    path = cache_dir / cache_name(model)
    if path.exists() and not refresh:
        payload = path.read_bytes()
    else:
        payload, final_url, content_type = request(product_url, timeout)
        if "text/html" not in content_type.lower() or f"/product/{model}".lower() not in final_url.lower():
            raise ValueError(f"unexpected TI product response: {final_url} ({content_type})")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    page = payload.decode("utf-8", errors="replace")
    text = " ".join(html.unescape(re.sub(r"<[^>]+>", " ", page)).split())
    if model.upper() not in text.upper():
        raise ValueError("exact model not present on TI product page")

    packages: list[str] = []
    pins: list[str] = []
    package_details: list[dict[str, str]] = []
    for match in PACKAGE_RE.finditer(text):
        pin_count = match.group("pins")
        package_family = match.group("package").upper().replace("-", "")
        package = f"{package_family}{pin_count}"
        if package not in packages:
            packages.append(package)
            package_details.append({
                "package": package,
                "pins": pin_count,
                "code": match.group("code").upper(),
            })
        if pin_count not in pins:
            pins.append(pin_count)

    datasheet_url = f"{ROOT}/lit/gpn/{model}"
    try:
        _payload, final_url, content_type = request(datasheet_url, timeout, method="HEAD")
        if "application/pdf" not in content_type.lower() and ".pdf" not in final_url.lower():
            datasheet_url = ""
    except (urllib.error.URLError, TimeoutError, ValueError):
        datasheet_url = ""
    return {
        "model": model,
        "product_url": product_url,
        "packages": packages,
        "pins": sorted(pins, key=int),
        "package_details": package_details,
        "datasheet_url": datasheet_url,
    }


def add_document(row: dict[str, str], record: dict[str, str]) -> bool:
    try:
        documents = json.loads(row.get("documents_json") or "[]")
    except json.JSONDecodeError:
        documents = []
    if not isinstance(documents, list):
        documents = []
    existing = next(
        (item for item in documents if isinstance(item, dict) and item.get("url") == record["url"]),
        None,
    )
    if existing is None:
        documents.append(record)
        changed = True
    else:
        before = dict(existing)
        existing.update(record)
        changed = existing != before
    if changed:
        row["documents_json"] = json.dumps(documents, ensure_ascii=False, sort_keys=True)
    return changed


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=root / "data" / "combined")
    parser.add_argument("--cache-dir", type=Path, default=root / "cache" / "ti-official")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    data = args.data_dir.resolve()
    devices = read_csv(data / "device-variants.csv")
    sources = read_csv(data / "sources.csv")
    targets = [
        row for row in devices
        if row.get("manufacturer") == "Texas Instruments"
        and (not row.get("package_types") or not row.get("documents_json"))
    ]
    evidence: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(
                product_evidence,
                row["device_name"],
                args.cache_dir.resolve(),
                args.timeout,
                args.refresh,
            ): row["device_name"]
            for row in targets
        }
        for future in as_completed(futures):
            model = futures[future]
            try:
                evidence[model.upper()] = future.result()
            except Exception as exc:  # pragma: no cover - remote availability
                errors.append(f"{model}: {type(exc).__name__}: {exc}")

    observed = utc_now()
    source_map = {row.get("source_id", ""): row for row in sources}
    package_updates = 0
    package_corrections = 0
    document_updates = 0
    datasheet_updates = 0
    for row in targets:
        item = evidence.get(row.get("device_name", "").upper())
        if not item:
            continue
        packages = ";".join(item["packages"])
        pins = ";".join(item["pins"])
        if packages and (row.get("package_types") != packages or row.get("pin_counts") != pins):
            if row.get("package_types"):
                package_corrections += 1
            else:
                package_updates += 1
            row["package_types"] = packages
            row["pin_counts"] = pins
        document_updates += int(add_document(row, {
            "title": f"TI {row['device_name']} official product page",
            "url": item["product_url"],
            "kind": "product_page",
            "verification_status": "official_exact_product_page",
        }))
        if item["datasheet_url"]:
            changed = add_document(row, {
                "title": f"TI {row['device_name']} data sheet",
                "url": item["datasheet_url"],
                "kind": "datasheet",
                "verification_status": "official_datasheet_http_200_pdf",
            })
            document_updates += int(changed)
            datasheet_updates += int(changed)
        source_id = f"ti:product-page:{row['device_name'].lower()}"
        source_ids = [value for value in row.get("source_id", "").split(";") if value]
        row["source_id"] = ";".join(dict.fromkeys([*source_ids, source_id]))
        row["verification_status"] = "multi_source_manufacturer_product_page"
        source_map[source_id] = {
            "source_id": source_id,
            "source_type": "manufacturer_product_page",
            "publisher": "Texas Instruments",
            "title": f"TI {row['device_name']} official product page",
            "url": item["product_url"],
            "version": "live",
            "observed_at": observed,
            "verification_scope": (
                "Exact TI product page: explicit package family, pin count and package code; "
                "stable GPN data-sheet URL separately verified as an HTTP 200 PDF."
            ),
        }

    write_csv(data / "device-variants.csv", devices)
    write_csv(data / "sources.csv", sorted(source_map.values(), key=lambda row: row.get("source_id", "")))
    remaining = [row["device_name"] for row in targets if not row.get("package_types")]
    report = {
        "generated_at": observed,
        "targets": len(targets),
        "exact_product_pages": len(evidence),
        "package_updates": package_updates,
        "package_corrections": package_corrections,
        "document_updates": document_updates,
        "datasheet_updates": datasheet_updates,
        "remaining_missing_packages": remaining,
        "errors": errors,
    }
    (data / "ti-official-augmentation-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: report[key] for key in (
        "targets", "exact_product_pages", "package_updates", "package_corrections",
        "document_updates", "datasheet_updates",
    )}, ensure_ascii=False, indent=2))
    if errors:
        print(json.dumps({"errors": errors}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

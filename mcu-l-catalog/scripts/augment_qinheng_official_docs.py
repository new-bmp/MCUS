#!/usr/bin/env python3
"""Add exact WCH manuals and the manufacturer-wide package library."""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


API = "https://api1.wch.cn/api/official/website/articles/getArticle"
WEB = "https://www.wch.cn"
USER_AGENT = "MCUS/1.0 (+https://github.com/new-bmp/MCUS; WCH official importer)"
PACKAGE_LIBRARY = {
    "title": "WCH all-product schematic symbols and package drawings",
    "url": f"{WEB}/downloads/SCHPCB.ZIP.html",
    "kind": "package_drawing",
    "version": "8.3",
    "verification_status": "official_all_product_package_library",
}


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


def fetch_article(alias: str, cache_dir: Path, timeout: float, refresh: bool) -> dict[str, Any]:
    path = cache_dir / f"{alias.lower()}.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    url = f"{API}?{urllib.parse.urlencode({'alias': alias + '.html'})}"
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
            "Origin": WEB,
            "Referer": f"{WEB}/",
            "Connection": "close",
        },
    )
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = response.read()
            parsed = json.loads(payload.decode("utf-8-sig"))
            if not isinstance(parsed, dict) or parsed.get("code") != 0 or not isinstance(parsed.get("data"), dict):
                raise ValueError(str(parsed.get("message") or "article not found"))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            return parsed
        except Exception as exc:  # pragma: no cover - remote availability
            last_error = exc
            if attempt < 2:
                time.sleep(1.0 * (2**attempt))
    assert last_error is not None
    raise last_error


def source_alias(row: dict[str, str]) -> str:
    parsed = urllib.parse.urlsplit(row.get("source_url", ""))
    name = parsed.path.rstrip("/").rsplit("/", 1)[-1]
    return re.sub(r"(?i)\.html$", "", name).upper()


def exact_model_in_file(model: str, item: dict[str, Any]) -> bool:
    evidence = " ".join(str(item.get(field) or "") for field in ("scope", "tag", "content", "name"))
    return re.search(rf"(?<![A-Z0-9]){re.escape(model)}(?![A-Z0-9])", evidence, re.I) is not None


def document_kind(item: dict[str, Any]) -> str:
    evidence = f"{item.get('name', '')} {item.get('content', '')}"
    if re.search(r"(?:技术手册|应用手册|reference\s+manual|user\s+manual)", evidence, re.I):
        return "reference_manual"
    return "datasheet"


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
    parser.add_argument("--cache-dir", type=Path, default=root / "cache" / "qinheng-official")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    data = args.data_dir.resolve()
    devices = read_csv(data / "device-variants.csv")
    targets = [row for row in devices if row.get("manufacturer") == "Qinheng"]
    aliases = sorted({source_alias(row) for row in targets if source_alias(row)})
    articles: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for alias in aliases:
        try:
            articles[alias] = fetch_article(alias, args.cache_dir.resolve(), args.timeout, args.refresh)
        except Exception as exc:  # pragma: no cover - remote availability
            errors.append(f"{alias}: {type(exc).__name__}: {exc}")
        time.sleep(0.08)

    manual_updates = 0
    package_library_updates = 0
    models_with_manual = set()
    for row in targets:
        alias = source_alias(row)
        data_row = articles.get(alias, {}).get("data") or {}
        files = data_row.get("filesList") if isinstance(data_row, dict) else []
        for item in files or []:
            if not isinstance(item, dict) or int(item.get("categoryId") or 0) != 27:
                continue
            name = str(item.get("name") or "").strip()
            if not name.lower().endswith(".pdf") or not exact_model_in_file(row["device_name"], item):
                continue
            changed = add_document(row, {
                "title": str(item.get("content") or name).split("，", 1)[0].strip(),
                "url": f"{WEB}/downloads/{urllib.parse.quote(name, safe='._-')}.html",
                "kind": document_kind(item),
                "version": str(item.get("version") or ""),
                "verification_status": "official_exact_model_scope",
            })
            manual_updates += int(changed)
            models_with_manual.add(row["device_name"])
        package_library_updates += int(add_document(row, PACKAGE_LIBRARY))

    write_csv(data / "device-variants.csv", devices)
    report = {
        "generated_at": utc_now(),
        "devices": len(targets),
        "article_aliases": len(aliases),
        "articles_loaded": len(articles),
        "exact_model_manual_updates": manual_updates,
        "models_with_exact_manual": len(models_with_manual),
        "package_library_updates": package_library_updates,
        "errors": errors,
    }
    (data / "qinheng-official-document-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

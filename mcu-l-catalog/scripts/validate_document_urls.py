#!/usr/bin/env python3
"""Audit catalog document links and remove only confirmed dead URLs.

HTTP failures caused by a vendor WAF or rate limiting are retained as
``blocked``/``rate_limited`` records.  They are not treated as 404s, because
silently deleting an official document that is temporarily inaccessible would
be less accurate than showing its verification state.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


USER_AGENT = "MCUS/1.0 (+https://github.com/new-bmp/MCUS; documentation audit)"
VALID_REDIRECT = {301, 302, 303, 307, 308}
DEAD_STATUS = {404, 410}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def normalise_url(url: str) -> str:
    parsed = urllib.parse.urlsplit(url.strip())
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        return ""
    path = re.sub(r"(?i)(?:\.pdf){2,}$", ".pdf", parsed.path)
    # Fragments select a client-side view and are not part of the HTTP resource.
    return urllib.parse.urlunsplit((parsed.scheme.lower(), parsed.netloc, path, parsed.query, ""))


def document_url(document: dict[str, Any]) -> str:
    value = str(document.get("url") or document.get("href") or "").strip()
    name = str(document.get("name") or "").strip()
    return value or (name if name.startswith(("http://", "https://")) else "")


def classify_error(exc: Exception) -> tuple[str, str]:
    code = getattr(exc, "code", None)
    if code in DEAD_STATUS:
        return "dead", str(code)
    if code == 429:
        return "rate_limited", "429"
    if code in {401, 403}:
        return "blocked", str(code)
    if isinstance(exc, TimeoutError) or "timed out" in str(exc).lower():
        return "unverified", "timeout"
    return "unverified", type(exc).__name__


def check_url(url: str, timeout: float) -> dict[str, Any]:
    checked_at = utc_now()
    headers = {"User-Agent": USER_AGENT, "Accept": "application/pdf,text/html;q=0.9,*/*;q=0.1"}
    final_url = url
    status: int | str = ""
    content_type = ""
    method = "HEAD"
    try:
        request = urllib.request.Request(url, headers=headers, method="HEAD")
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = response.status
            final_url = response.geturl()
            content_type = response.headers.get("Content-Type", "")
    except urllib.error.HTTPError as exc:
        status = exc.code
        final_url = exc.geturl() or url
        content_type = exc.headers.get("Content-Type", "") if exc.headers else ""
        # HEAD behaviour is not reliable enough to prove that a document is
        # dead.  Several vendor CDNs return 403/404 to HEAD while serving the
        # same stable URL with GET.  Confirm every HEAD HTTP failure with a
        # small ranged GET before classifying or removing the URL.
        try:
            method = "GET"
            request = urllib.request.Request(url, headers={**headers, "Range": "bytes=0-4095"}, method="GET")
            with urllib.request.urlopen(request, timeout=timeout) as response:
                status = response.status
                final_url = response.geturl()
                content_type = response.headers.get("Content-Type", "")
        except Exception as retry_exc:  # pragma: no cover - depends on remote host
            state, reason = classify_error(retry_exc)
            return {"url": url, "status": getattr(retry_exc, "code", status), "state": state,
                    "reason": reason, "final_url": getattr(retry_exc, "url", final_url),
                    "content_type": content_type, "method": method, "checked_at": checked_at}
    except Exception as exc:  # pragma: no cover - depends on remote host
        state, reason = classify_error(exc)
        return {"url": url, "status": getattr(exc, "code", ""), "state": state, "reason": reason,
                "final_url": getattr(exc, "url", final_url), "content_type": content_type,
                "method": method, "checked_at": checked_at}

    if isinstance(status, int) and 200 <= status < 400:
        state = "valid"
        reason = "http_ok" if status < 300 else "redirect_ok"
    elif status in DEAD_STATUS:
        state, reason = "dead", str(status)
    elif status == 429:
        state, reason = "rate_limited", "429"
    elif status in {401, 403}:
        state, reason = "blocked", str(status)
    else:
        state, reason = "unverified", str(status)
    return {"url": url, "status": status, "state": state, "reason": reason,
            "final_url": final_url, "content_type": content_type, "method": method,
            "checked_at": checked_at}


def document_rows(devices: list[dict[str, str]]) -> tuple[dict[str, list[tuple[str, int]]], list[str]]:
    references: dict[str, list[tuple[str, int]]] = {}
    malformed: list[str] = []
    for device in devices:
        try:
            docs = json.loads(device.get("documents_json") or "[]")
        except json.JSONDecodeError:
            docs = []
        if not isinstance(docs, list):
            continue
        for index, doc in enumerate(docs):
            if not isinstance(doc, dict):
                continue
            raw_url = document_url(doc)
            if not raw_url:
                continue
            url = normalise_url(raw_url)
            if not url:
                malformed.append(f"{device.get('device_id')}[{index}]")
                continue
            references.setdefault(url, []).append((device.get("device_id", ""), index))
    return references, malformed


def apply_dead_cleanup(data: Path, audits: dict[str, dict[str, Any]]) -> int:
    devices_path = data / "device-variants.csv"
    devices = read_csv(devices_path)
    removed = 0
    for row in devices:
        try:
            docs = json.loads(row.get("documents_json") or "[]")
        except json.JSONDecodeError:
            docs = []
        if not isinstance(docs, list):
            continue
        kept = []
        for doc in docs:
            if not isinstance(doc, dict):
                continue
            raw_url = document_url(doc)
            url = normalise_url(raw_url) if raw_url else ""
            audit = audits.get(url) if url else None
            if audit and audit.get("state") == "dead":
                removed += 1
                continue
            if url:
                doc["url"] = url
                if str(doc.get("name") or "").strip() == raw_url:
                    doc.pop("name", None)
                if audit:
                    doc["verification_status"] = audit["state"]
                    doc["http_status"] = audit.get("status", "")
                    doc["checked_at"] = audit.get("checked_at", "")
            kept.append(doc)
        row["documents_json"] = json.dumps(kept, ensure_ascii=False, sort_keys=True)
    if devices:
        write_csv(devices_path, list(devices[0].keys()), devices)
    return removed


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=root / "data" / "combined")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--timeout", type=float, default=25.0)
    parser.add_argument("--apply", action="store_true", help="remove only confirmed 404/410 document URLs")
    args = parser.parse_args()
    data = args.data_dir.resolve()
    output = args.output or data / "document-url-audit.json"
    devices = read_csv(data / "device-variants.csv")
    references, malformed = document_rows(devices)
    audits: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(check_url, url, args.timeout): url for url in references}
        for future in as_completed(futures):
            result = future.result()
            audits[result["url"]] = result

    removed = apply_dead_cleanup(data, audits) if args.apply else 0
    state_counts: dict[str, int] = {}
    for item in audits.values():
        state_counts[item["state"]] = state_counts.get(item["state"], 0) + 1
    report = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "data_dir": str(data),
        "unique_urls": len(audits),
        "document_references": sum(len(value) for value in references.values()),
        "malformed_references": malformed,
        "state_counts": state_counts,
        "confirmed_dead_removed": removed,
        "records": sorted(audits.values(), key=lambda item: item["url"]),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    csv_path = output.with_suffix(".csv")
    fields = ["url", "status", "state", "reason", "final_url", "content_type", "method", "checked_at"]
    write_csv(csv_path, fields, sorted(audits.values(), key=lambda item: item["url"]))
    print(json.dumps({key: report[key] for key in ("unique_urls", "document_references", "state_counts", "confirmed_dead_removed")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Shared primitives for auditable manufacturer catalog importers."""

from __future__ import annotations

import csv
import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


USER_AGENT = "MCU-L-Catalog/0.4 (+official-source importer; local audit cache)"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def slug(value: str) -> str:
    return "".join(character.lower() if character.isalnum() else "-" for character in value).strip("-")


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class Snapshot:
    url: str
    path: Path
    payload: bytes
    sha256: str
    observed_at: str


class OfficialFetcher:
    def __init__(
        self,
        *,
        cache_dir: Path,
        allowed_hosts: set[str],
        refresh: bool = False,
        timeout: float = 45.0,
        retries: int = 3,
    ) -> None:
        self.cache_dir = cache_dir
        self.allowed_hosts = {host.lower() for host in allowed_hosts}
        self.refresh = refresh
        self.timeout = timeout
        self.retries = retries

    def _validate_url(self, url: str) -> None:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme != "https" or (parsed.hostname or "").lower() not in self.allowed_hosts:
            raise ValueError(f"unapproved official source URL: {url}")

    def fetch(
        self,
        url: str,
        *,
        cache_name: str,
        method: str = "GET",
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> Snapshot:
        self._validate_url(url)
        cache_path = self.cache_dir / cache_name
        observed_at = utc_now()
        if cache_path.exists() and not self.refresh:
            payload = cache_path.read_bytes()
        else:
            request_headers = {"User-Agent": USER_AGENT, "Accept": "application/json,text/html;q=0.8,*/*;q=0.5"}
            request_headers.update(headers or {})
            request = urllib.request.Request(url, data=body, headers=request_headers, method=method)
            last_error: Exception | None = None
            for attempt in range(self.retries):
                try:
                    with urllib.request.urlopen(request, timeout=self.timeout) as response:
                        payload = response.read()
                    break
                except (OSError, urllib.error.HTTPError, urllib.error.URLError) as exc:
                    last_error = exc
                    if attempt + 1 < self.retries:
                        time.sleep(0.75 * (2**attempt))
            else:
                assert last_error is not None
                raise last_error
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(payload)
        return Snapshot(
            url=url,
            path=cache_path,
            payload=payload,
            sha256=hashlib.sha256(payload).hexdigest(),
            observed_at=observed_at,
        )


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

ERROR_FIELDS = ["scope", "source_url", "item", "error"]

FIELD_PROVENANCE_FIELDS = [
    "record_type", "record_id", "field_name", "source_id", "source_url",
    "source_path", "source_value_json", "observed_at", "verification_status",
]

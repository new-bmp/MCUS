#!/usr/bin/env python3
"""Extract text from cached official PDFs for downstream evidence parsers.

The catalog keeps the PDF as the authoritative artifact.  This companion
text file is only a deterministic parsing representation and is never used
as a source without the matching cached PDF.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pypdf import PdfReader


def extract(pdf: Path) -> str:
    reader = PdfReader(str(pdf))
    pages: list[str] = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n\n".join(pages).replace("\x00", "")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, default=Path(__file__).resolve().parents[1] / "cache")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    converted = 0
    failed = 0
    for pdf in sorted(args.cache_dir.rglob("*.pdf")):
        # Existing importer output uses the PDF stem, e.g. foo.pdf -> foo.txt.
        output = pdf.with_suffix(".txt")
        if output.exists() and not args.refresh:
            continue
        try:
            text = extract(pdf)
            if not text.strip():
                failed += 1
                continue
            output.write_text(text, encoding="utf-8")
            converted += 1
        except Exception:
            failed += 1
    print({"converted": converted, "failed": failed})
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

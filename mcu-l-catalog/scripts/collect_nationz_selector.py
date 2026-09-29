"""Cache the public selector requests issued by Nationz MCU category pages."""
import json
import re
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from vendor_import_common import OfficialFetcher

ROOT = "https://www.nationstech.com"
CACHE = Path(__file__).resolve().parents[1] / "cache/nationz-official"
fetcher = OfficialFetcher(cache_dir=CACHE, allowed_hosts={"www.nationstech.com"}, timeout=35)

def collect(category):
    url = ROOT + "/product/general/" + category + "/"
    page = fetcher.fetch(url, cache_name=category + ".html").payload.decode("utf-8")
    match = re.search(r"\btwo\s*:\s*(\d+)", page)
    if not match:
        raise ValueError("Missing selector category: " + category)
    body = urllib.parse.urlencode({"two": match[1], "key": ""}).encode()
    snap = fetcher.fetch(ROOT + "/api/getxh", cache_name=category + ".json", method="POST", body=body,
        headers={"X-Requested-With": "XMLHttpRequest", "Referer": url, "Content-Type": "application/x-www-form-urlencoded"})
    data = json.loads(snap.payload)
    assert data["code"] == 0 and data["data"]["body"], category
    return category, len(data["data"]["body"])

if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=3) as pool:
        for result in pool.map(collect, ["n32g", "n32h", "n32a", "n32l", "N32wbble", "n32m"]):
            print(result, flush=True)

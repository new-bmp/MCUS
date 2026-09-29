#!/usr/bin/env python3
"""Import exact orderable N32 MCU rows from the public Nationz selector.

Uses the requests issued by the official category pages, never generated
suffix combinations. PDSC records are deliberately not used as a fallback:
e.g. the N32G401 Pack incorrectly identifies a Cortex-M0.
"""
import argparse
import hashlib
import html
import json
import re
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from collect_nationz_selector import collect, CACHE, ROOT, fetcher
from vendor_import_common import DEVICE_FIELDS, PART_FIELDS, SOURCE_FIELDS, write_csv, write_json, utc_now

SOURCE_KIND = "nationz_product_selector"
CATEGORIES = ["n32g", "n32h", "n32a", "n32l", "N32wbble", "n32m"]

def clean(s):
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", str(s))).split())

def count(s):
    s = str(s).strip()
    if s == "-": return 0
    if s.lower() in {"yes", "y"}: return 1
    m = re.fullmatch(r"(\d+)(?:\([1-5]\)|\*)?", s)
    return int(m[1]) if m else None

def feature(kind, name, n=None, **kwargs):
    f = {"type": kind, "name": name, "source_kind": SOURCE_KIND, **kwargs}
    if n is not None: f.update(n=str(n), count=str(n))
    return f

def row_features(r):
    f = []
    for field, kind in [("I/O","IOs"),("PWM","PWM"),("RTC","RTC"),("I2C","I2C"),
            ("OPAMP","OPAMP"),("COMP","COMP"),("USB Device","USBD"),("USB HS OTG","USBOTG"),
            ("SDMMC","SDIO"),("TRNG","RNG"),("DVP","Camera"),("LCDC","GLCD"),("BLE","Bluetooth")]:
        if field in r: f.append(feature(kind, field, count(r[field])))
    timers = ["高精度定时器","高级定时器","通用定时器","基础定时器","低功耗定时器"]
    amounts = [count(r.get(k,"")) for k in timers]
    if all(v is not None for v in amounts):
        f.append(feature("Timer", "定时器合计（" + "；".join(f"{k} {r[k]}" for k in timers) + "）", sum(amounts)))
    for field, kind in [("USART/ISO7816/LIN","USART"),("UART/LIN","UART"),("LPUART","UART")]:
        if field in r: f.append(feature(kind, field, count(r[field])))
    for field, kind in [("CAN","CAN"),("CAN-FD","CAN"),("ETH 1000M","ETH"),("ETH 100M","ETH")]:
        if field in r: f.append(feature(kind, field, count(r[field])))
    spi = r.get("SPI/I2S", "").split("/")
    if len(spi) == 2:
        f += [feature("SPI","SPI (shared SPI/I2S)",count(spi[0])), feature("I2S","I2S (shared SPI/I2S)",count(spi[1]))]
    elif len(spi) == 1 and count(spi[0]) is not None:
        # A single value establishes SPI only; the omitted I2S count is unknown.
        f.append(feature("SPI","SPI (selector SPI/I2S column)",count(spi[0])))
    for field in ["QSPI", "xSPI", "FEMC"]:
        if field in r: f.append(feature("ExtBus", field, count(r[field])))
    dma = r.get("DMA/通道数", "").split("/")
    if len(dma) == 2:
        # The app's DMA summary is a channel count, not a controller count.
        f += [feature("DMAControllers", "DMA controllers", count(dma[0])), feature("DMA","DMA channels",count(dma[1]))]
    mdma = r.get("MDMA/Channels", "").split("/")
    if len(mdma) == 2:
        f += [feature("MDMAControllers", "MDMA controllers", count(mdma[0])), feature("MDMAChannels","MDMA channels (separate from DMA)",count(mdma[1]))]
    adc = re.fullmatch(r"(\d+)(?:\(5\))?x(\d+)bit", r.get("ADC个数x精度", ""), re.I)
    if adc:
        f += [feature("ADCUnits", "ADC converter units", int(adc[1]), m=adc[2]),
              feature("ADC", "ADC channels", count(r.get("ADC通道数","")), m=adc[2])]
        if "(5)" in r["ADC个数x精度"]:
            f.append(feature("VendorCapability", "ADC 硬件过采样至 16 bit（原生分辨率 12 bit）"))
    dac = re.fullmatch(r"(\d+)x(\d+)bit", r.get("DAC", ""), re.I)
    if dac: f.append(feature("DAC", "DAC 数量（官方选型表）", int(dac[1]), m=dac[2]))
    elif r.get("DAC") == "-": f.append(feature("DAC", "DAC 数量（官方选型表）", 0))
    volts = re.findall(r"\d+(?:\.\d+)?", r.get("工作电压", ""))
    if len(volts) == 2: f.append({"type":"VCC","n":volts[0],"m":volts[1],"source_kind":SOURCE_KIND})
    for field in ["DCDC", "Cordic", "DSMU", "FMAC", "2.5D GPU", "JPEG", "EtherCAT从站控制器"]:
        if count(r.get(field,"")):
            f.append(feature("VendorCapability",field,count(r[field])))
    for field in ["AES", "DES/3DES", "SM1", "SM3", "SM4", "SM7", "SHA1/SHA224/SHA256", "MD5", "CRC16", "CRC32"]:
        if count(r.get(field,"")): f.append(feature("Crypto" if not field.startswith("CRC") else "CRC",field,count(r[field])))
    return f

def processors(r):
    names = re.findall(r"M(?:0\+?|4F?|7)",r["内核"])
    clocks = [float(v)*1000000 for v in r["主频（MHz）"].split("&") if re.fullmatch(r"\d+(?:\.\d+)?",v)]
    result = []
    for i, name in enumerate(names):
        p = {"Dcore":"Cortex-"+name.removesuffix("F"),"Pname":name,"DcoreCount":"1"}
        if name.endswith("F"): p["Dfpu"]="1"
        if name in {"M0","M0+"}: p["Dfpu"]="0"
        if len(clocks) == len(names): p["Dclock"]=str(int(clocks[i]))
        result.append(p)
    return result, int(max(clocks)) if clocks else ""

def pdf_ok(url):
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent":"MCUS catalog", "Range":"bytes=0-7"}),timeout=30) as res:
            return res.read(5) == b"%PDF-"
    except Exception: return False

def documents_for_page(url):
    key = hashlib.sha256(url.encode()).hexdigest()[:16]
    page = fetcher.fetch(url,cache_name="pages/"+key+".html").payload.decode("utf-8",errors="replace")
    docs = [{"kind":"product_page","title":"国民技术官方产品与资料页","url":url}]
    pattern = r'<div class="pdfTitle">(.*?)</div>\s*<div class="pdfOther">(.*?)<div class="btns">(.*?)</div>'
    for title, info, links in re.findall(pattern,page,re.S):
        title=clean(title)
        kind = "datasheet" if "数据手册" in title else "reference_manual" if "用户手册" in title else None
        if not kind: continue
        # Only public file links are accepted; login-only controls stay on the product page.
        for href in re.findall(r'href="([^"]+)"',links):
            if not href.startswith("/api/dowfilebefore"): continue
            direct=urllib.parse.urljoin(ROOT,html.unescape(href))
            docs.append({"kind":kind,"title":title+("（英文）" if "l=en" in direct else "（中文）"),"url":direct})
    return url, docs

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=Path(__file__).resolve().parents[1]/"data/vendor-packs/nationz")
    args=parser.parse_args()
    with ThreadPoolExecutor(max_workers=3) as pool: list(pool.map(collect,CATEGORIES))
    all_rows=[]
    for cat in CATEGORIES:
        payload=json.loads((CACHE/(cat+".json")).read_text(encoding="utf-8"))
        all_rows.extend(payload["data"]["body"])
    unique={r["产品型号"]:r for r in all_rows}
    assert len(unique)==len(all_rows), "Duplicate selector rows require review"
    urls=sorted({urllib.parse.urljoin(ROOT,r["url"]) for r in all_rows})
    with ThreadPoolExecutor(max_workers=3) as pool: pages=dict(pool.map(documents_for_page,urls))
    check_path=CACHE/"document-checks.json"
    checks=json.loads(check_path.read_text(encoding="utf-8")) if check_path.exists() else {}
    pending=sorted({d["url"] for docs in pages.values() for d in docs if d["kind"]!="product_page" and d["url"] not in checks})
    with ThreadPoolExecutor(max_workers=3) as pool:
        checks.update(zip(pending,pool.map(pdf_ok,pending)))
    write_json(check_path,checks)
    observed=utc_now()
    digest=hashlib.sha256(json.dumps(all_rows,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
    sid="nationz:official-selector:"+digest[:16]
    devices=[]; parts=[]
    for name,r in sorted(unique.items()):
        assert re.fullmatch(r"N32[A-Z0-9-]+",name), name
        core,hz=processors(r)
        prefix=re.match(r"N32(?:WB|[A-Z])",name)[0]
        page=urllib.parse.urljoin(ROOT,r["url"])
        docs=[]
        for d in pages[page]:
            if d["kind"]=="product_page" or checks.get(d["url"]):
                docs.append({**d,"verification_status":"official_product_page" if d["kind"]=="product_page" else "official_pdf_header_verified"})
        package=clean(r["封装"])
        pin=re.match(r"[A-Z]+(\d+)(?:\+(\d+))?",package)
        pin_count=str(int(pin[1])+int(pin[2] or 0)) if pin else ""
        flash=int(float(r["Flash（KB）"])*1024)
        ram=int(float(r["SRAM（KB）"])*1024)
        row=dict.fromkeys(DEVICE_FIELDS,"")
        row.update(device_id="nationz::"+name.lower(),product_line_id="nationz::"+prefix.lower()+"::"+r["子系列"].lower(),
            manufacturer="Nationz",product_type="wireless_mcu" if prefix=="N32WB" else "low_power_mcu" if prefix=="N32L" else "high_performance_mcu" if prefix=="N32H" else "general_purpose_mcu",
            architecture_class=" + ".join(p["Dcore"] for p in core),family="N32",series=prefix,product_line=r["子系列"],
            device_name=name,generic_device_name=name,processor_cores=json.dumps(core),max_clock_hz=hz,flash_bytes=flash,ram_bytes=ram,
            package_types=package,pin_counts=pin_count,
            memory_regions_json=json.dumps([{"name":"Flash","type":"Flash","bytes":flash},{"name":"SRAM","type":"RAM","bytes":ram}]),
            features_json=json.dumps(row_features(r),ensure_ascii=False),documents_json=json.dumps(docs,ensure_ascii=False),
            lifecycle="unknown",source_id=sid,source_url=page,source_version="sha256:"+digest,observed_at=observed,
            verification_status="manufacturer_product_selector_api")
        devices.append(row)
        part={k:row.get(k,"") for k in PART_FIELDS}
        part.update(orderable_part_id="nationz::part::"+name.lower(),part_number=name,package_name=package,
            temperature_range=r["工作温度"],packing_form=r.get("SPQ(PCS)",""),decode_status="official_exact_model_row")
        parts.append(part)
    write_csv(args.output/"device-variants.csv",DEVICE_FIELDS,devices)
    write_csv(args.output/"orderable-parts.csv",PART_FIELDS,parts)
    write_csv(args.output/"sources.csv",SOURCE_FIELDS,[dict(source_id=sid,source_type="manufacturer_product_selector_api",publisher="Nationz Technologies",title="国民技术官方 MCU 选型表（6 类别）",url=ROOT+"/product/general/",version="sha256:"+digest,observed_at=observed,verification_scope="Exact model rows. NSING/Nationz sites cross-link; current Chinese selector takes precedence over conflicting PDSC fields.")])
    write_json(args.output/"selector-records.json",all_rows)
    report={"devices":len(devices),"product_lines":len({r['product_line'] for r in devices}),"datasheet_devices":sum(any(d['kind']=='datasheet' for d in json.loads(r['documents_json'])) for r in devices),"verified_pdf_urls":sum(checks.values()),"failed_pdf_urls":[k for k,v in checks.items() if not v],"completeness_claim":"Snapshot of all 273 exact rows returned by six public MCU selector categories; not a claim of all historical N32 models."}
    write_json(args.output/"official-adapter-report.json",report)
    print(json.dumps(report,ensure_ascii=False),flush=True)

if __name__=="__main__": main()

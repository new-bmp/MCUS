#!/usr/bin/env python3
"""Expand MM32 using header-aligned exact-model tables on official pages.

Keep older Pack devices. Never derive model suffixes or reuse a redirected
page's data based on its URL: only actual table rows identify a device.
"""
import argparse
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import augment_mindmotion_official as mm
from vendor_import_common import DEVICE_FIELDS, PART_FIELDS, SOURCE_FIELDS, write_csv, write_json, utc_now
from import_nationz_official import pdf_ok

ROOT=Path(__file__).resolve().parents[1]
KIND="mindmotion_product_selector"

def count(value):
    if value == "-": return 0
    if value.upper() in {"Y","YES"}: return 1
    m=re.fullmatch(r"(\d+)(?:\s*\(FD\))?",value)
    return int(m[1]) if m else None

def table_rows(payload):
    headers=[]
    for block in re.findall(r"<dl\b[^>]*>.*?</dl>",payload,re.I|re.S):
        cells=[mm.clean_text(c) for c in re.findall(r"<dd\b[^>]*>(.*?)</dd>",block,re.I|re.S)]
        if "Part No." in cells and "Core" in cells and "Flash (KB)" in cells:
            headers=cells
        elif headers and len(cells)==len(headers):
            row=dict(zip(headers,cells))
            if re.fullmatch(r"MM32[A-Z0-9]+",row.get("Part No.",""),re.I):
                yield row,block

def features(r):
    result=[]
    def add(kind,name,n=None,**kw):
        item={"type":kind,"name":name,"source_kind":KIND,**kw}
        if n is not None:item.update(count=str(n),n=str(n))
        result.append(item)
    mapping={"I/O#":"IOs","WDG":"WDT","RTC":"RTC","U(S)ART":"USART","LPUART":"UART","I²C":"I2C","I³C":"I3C","SPI":"SPI","I²S":"I2S","CAN 2.0B":"CAN","Ethernet":"ETH","SDIO":"SDIO","ACMP":"COMP","OpAmp":"OPAMP","QSPI":"ExtBus"}
    for field,kind in mapping.items():
        if field in r:add(kind,field+(" CAN-FD" if "(FD)" in r[field] else ""),count(r[field]))
    timer_fields=[k for k in r if k in {"Advanced TMR","GP TMR","Basic TMR","LP TMR"}]
    if timer_fields and all(count(r[k]) is not None for k in timer_fields):
        add("Timer","Timer total: "+", ".join(k+"="+r[k] for k in timer_fields),sum(count(r[k]) for k in timer_fields))
    for field in r:
        if field.startswith("ADC"):
            n=count(r[field]);bits=re.search(r"(\d+)\s*bit",field,re.I)
            channel_match=re.search(r'(\d+)\s*ch\b',r[field],re.I)
            # This selector does not label the quantity as converter units or
            # channels. Preserve raw semantics until a datasheet resolves it.
            if channel_match:
                add('ADC',field+' channels (原表 '+r[field]+')',int(channel_match[1]),**({'m':bits[1]} if bits else {}))
                add('ADCPerformance',field)
            elif n is not None and n>0:
                add("ADC",field+"（原表数量，口径未标明）",n,**({"m":bits[1]} if bits else {}))
                add("ADCPerformance",field)
        elif field.startswith("DAC"):
            n=count(r[field]);bits=re.search(r"(\d+)\s*bit",field,re.I)
            add("DAC",field,n,**({"m":bits[1]} if bits and n else {}))
        elif field.startswith("USB"):
            v=r[field]
            if v == "-":add("USB",field,0)
            elif v in {"D","D/H","D/H/O","OTG","H/D","H"}:add("USBOTG" if "/" in v or v=="OTG" else "USBD" if v=="D" else "USBH",field+" "+v,1)
    if r.get("Ext. Bus I/F") not in {None,"","-"}:add("ExtBus",r["Ext. Bus I/F"],1)
    if r.get("Security") not in {None,"","-"}:add("Crypto",r["Security"],1)
    return result

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--base",type=Path,default=ROOT/"data/combined")
    parser.add_argument("--output",type=Path,default=ROOT/"data/vendor-packs/mindmotion")
    args=parser.parse_args()
    cache=ROOT/"cache/mindmotion-current/pages"
    seed=mm.fetch(mm.SEED,cache,30,False)
    queued={mm.SEED};pending=set(mm.product_links(seed,mm.SEED));pages={mm.SEED:seed};errors=[]
    def get(url):
        try:return url,mm.fetch(url,cache,30,False),None
        except Exception as e:return url,"",str(e)
    with ThreadPoolExecutor(max_workers=4) as pool:
        while pending:
            batch=sorted(pending-queued)
            if not batch:break
            queued.update(batch);pending=set()
            for url,payload,error in pool.map(get,batch):
                if error:errors.append({"url":url,"error":error});continue
                pages[url]=payload
                pending.update(mm.product_links(payload,url))
    existing={r["device_name"]:r for r in mm.read_csv(args.base/"device-variants.csv") if r["manufacturer"]=="MindMotion"}
    old_count=len(existing)
    source_rows=[r for r in mm.read_csv(args.base/"sources.csv") if any(r['source_id'] in d['source_id'].split(';') for d in existing.values())]
    parts={r['part_number']:r for r in mm.read_csv(args.base/"orderable-parts.csv") if r['manufacturer']=="MindMotion"}
    observed=utc_now();seen={};conflicts=[];evidence=[]
    def page_priority(item):
        url,payload=item
        # Some old official URLs serve an unrelated default product. Prefer
        # discovered canonical pages whose manual title matches the URL.
        names=[m[0].lower() for d in mm.page_documents(payload) for m in [re.search(r'MM32[A-Z]+\d+',d['title'],re.I)] if m]
        return (not any('/'+name+'/' in url.lower() for name in names),url)
    for url,payload in sorted(pages.items(),key=page_priority):
        page_docs=mm.page_documents(payload)
        digest=hashlib.sha256(payload.encode()).hexdigest()
        sid="mindmotion:selector:"+digest[:16]
        scoped=False
        for r,block in table_rows(payload):
            name=r["Part No."].upper()
            if name in seen:
                if r!=seen[name]: conflicts.append({"model":name,"url":url,"first":seen[name],"other":r})
                continue
            seen[name]=r;scoped=True
            core=r.get("Core","").replace("ARM ","").replace("Arm ","")
            if re.fullmatch(r"M[0-9]+\+?",core,re.I):core="Cortex-"+core.upper()
            if core.upper().startswith("CORTEX-"):core="Cortex-"+core.split("-",1)[1].upper()
            if core.upper()=="STAR-MC1":core="STAR-MC1"
            if core in {"","-"}:continue
            numeric=[r.get(k,"") for k in ("Max Speed (MHz)","Flash (KB)","RAM (KB)")]
            if not all(re.fullmatch(r"\d+(?:\.\d+)?",v) for v in numeric):continue
            hz,flash,ram=[int(float(v)*u) for v,u in zip(numeric,[1000000,1024,1024])]
            old=existing.get(name,{})
            row={k:old.get(k,"") for k in DEVICE_FIELDS}
            family=re.match(r"MM32(?:SPIN|[A-Z])",name)[0]
            # Product page DS titles provide the product-line name, not suffix inference.
            line=next((m[0].upper() for d in page_docs for m in [re.search(r"MM32[A-Z]+\d+",d['title'],re.I)] if m),old.get('product_line',name))
            package=r.get('Package','').strip()
            pins=r.get('Pin Counts','').strip()
            if not pins.isdigit():
                explicit=re.fullmatch(r'[A-Z]+(\d+)',package)
                pins=explicit[1] if explicit else ''
            if re.fullmatch(r'[A-Z]+',package) and pins:package+=pins
            row.update(device_id=old.get('device_id',"mindmotion::"+name.lower()),product_line_id="mindmotion::"+family.lower()+"::"+line.lower(),
                manufacturer="MindMotion",family="MM32",series=family,product_line=line,device_name=name,generic_device_name=old.get('generic_device_name',name),
                product_type="low_power_mcu" if family=="MM32L" else "general_purpose_mcu",architecture_class=core,max_clock_hz=str(hz),flash_bytes=str(flash),ram_bytes=str(ram),
                package_types=package,pin_counts=pins,source_url=url,
                source_id=';'.join(dict.fromkeys([*old.get('source_id','').split(';'),sid])).strip(';'),source_version="sha256:"+digest,observed_at=observed,verification_status="manufacturer_product_page")
            prior=json.loads(old.get('processor_cores') or '[]')
            p=prior[0].copy() if len(prior)==1 and prior[0].get('Dcore')==core else {'Dcore':core,'DcoreCount':'1'}
            p['Dclock']=str(hz)
            row['processor_cores']=json.dumps([p])
            raw_features=json.loads(old.get('features_json') or '[]')
            incoming=features(r);types={f['type'] for f in incoming}
            if types.intersection({'USB','USBD','USBH','USBOTG'}):types.update({'USB','USBD','USBH','USBOTG'})
            # Replace overlapping quantities, keeping unrelated audited engineering evidence.
            retained=[f for f in raw_features if f.get('type') not in types]
            if 'ADC' in types:retained=[f for f in retained if f.get('type')!='ADCUnits' or f.get('source_kind')!='cmsis_svd']
            row['features_json']=json.dumps(retained+incoming,ensure_ascii=False)
            if not old or str(flash)!=old.get('flash_bytes') or str(ram)!=old.get('ram_bytes'):
                row['memory_regions_json']=json.dumps([{'name':'Flash','type':'Flash','bytes':flash},{'name':'SRAM','type':'RAM','bytes':ram}])
            row.setdefault('lifecycle','unknown')
            for d in page_docs:mm.add_document(row,d)
            mm.add_document(row,{'kind':'product_page','title':'MindMotion 官方精确型号选型表','url':url})
            drawing=mm.package_drawing(name,block)
            if drawing:mm.add_document(row,drawing)
            existing[name]=row
            part={k:row.get(k,'') for k in PART_FIELDS}
            part.update(orderable_part_id='mindmotion::part::'+name.lower(),part_number=name,package_name=row['package_types'],temperature_range=r.get('Operation Temp',''),decode_status='official_exact_model_row')
            parts[name]=part
            evidence.append({'model':name,'page':url,'fields':r})
        if scoped:source_rows.append(dict(source_id=sid,source_type='manufacturer_product_page',publisher='MindMotion',title='MindMotion exact-model selector',url=url,version='sha256:'+digest,observed_at=observed,verification_scope='Header-aligned exact model rows; ADC quantity retains unspecified semantics.'))
    for row in existing.values():
        prefix=re.match(r'MM32(?:SPIN|[A-Z])',row['device_name'])[0]
        row.update(family='MM32',series=prefix,product_line_id='mindmotion::'+prefix.lower()+'::'+row['product_line'].lower())
    check_path=ROOT/'cache/mindmotion-current/document-checks.json'
    checks=json.loads(check_path.read_text(encoding='utf-8')) if check_path.exists() else {}
    urls={d['url'] for row in existing.values() for d in json.loads(row.get('documents_json') or '[]')
          if d.get('kind') in {'datasheet','reference_manual'} and d.get('url','').startswith(mm.ROOT+'/download1.aspx')}
    pending=sorted(urls-set(checks))
    with ThreadPoolExecutor(max_workers=4) as pool: checks.update(zip(pending,pool.map(pdf_ok,pending)))
    write_json(check_path,checks)
    for row in existing.values():
        docs=json.loads(row.get('documents_json') or '[]')
        for d in docs:
            if d.get('url') in checks:
                if checks[d['url']]:d['verification_status']='official_pdf_header_verified'
                else:
                    d['kind']='source'
                    d['verification_status']='download_endpoint_unverified'
        row['documents_json']=json.dumps(docs,ensure_ascii=False)
    write_csv(args.output/'device-variants.csv',DEVICE_FIELDS,sorted(existing.values(),key=lambda r:r['device_name']))
    write_csv(args.output/'orderable-parts.csv',PART_FIELDS,sorted(parts.values(),key=lambda r:r['part_number']))
    write_csv(args.output/'sources.csv',SOURCE_FIELDS,list({r['source_id']:r for r in source_rows}.values()))
    write_json(args.output/'selector-records.json',evidence)
    write_json(args.output/'selector-conflicts.json',conflicts)
    report={'devices':len(existing),'added':len(existing)-old_count,'exact_selector_models':len(evidence),'pages':len(pages),'conflicts':len(conflicts),'errors':errors,'verified_pdf_urls':sum(checks.values()),'unverified_pdf_urls':[k for k,v in checks.items() if not v],'completeness_claim':'Current official product tables plus retained historical Pack variants; not all historical or future models.'}
    write_json(args.output/'official-adapter-report.json',report)
    print(json.dumps(report,ensure_ascii=False),flush=True)

if __name__=='__main__':main()

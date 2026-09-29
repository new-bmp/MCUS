#!/usr/bin/env python3
"""Add target-specific ESP-IDF memory guides and exact WCH product diagrams."""
import hashlib
import json
import re
import subprocess
from html.parser import HTMLParser
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from collect_memory_evidence import ROOT, read_rows, COMPILED, useful_quote


class Article(HTMLParser):
    def __init__(self):
        super().__init__(); self.parts=[]; self.text=[]; self.skip=0
    def handle_starttag(self,tag,attrs):
        if tag in ('script','style'): self.skip+=1
        if tag in ('p','li','h1','h2','h3','section','div'): self.flush()
    def handle_endtag(self,tag):
        if tag in ('script','style'): self.skip=max(0,self.skip-1)
        if tag in ('p','li','h1','h2','h3','section','div'): self.flush()
    def handle_data(self,data):
        if not self.skip: self.text.append(data)
    def flush(self):
        q=' '.join(''.join(self.text).split())
        if q: self.parts.append(q)
        self.text=[]

def download(url,path):
    if path.exists() and path.stat().st_size>100: return True
    process=subprocess.run(['curl.exe','-L','--fail','--silent','--show-error','--max-time','25',
                            '-A','Mozilla/5.0 (MCUS official memory references)','-o',str(path),url],capture_output=True,timeout=30)
    return process.returncode==0

def collect_esp(series,rows,cache):
    target={'ESP32-C2/ESP8684':'esp32c2','ESP8685':'esp32c3'}.get(series,series.lower().replace('-',''))
    url=f'https://docs.espressif.com/projects/esp-idf/en/stable/{target}/api-guides/memory-types.html'
    key=hashlib.sha256(url.encode()).hexdigest()[:16]
    path=cache/(key+'.html')
    if not download(url,path): return None
    parser=Article(); parser.feed(path.read_text(encoding='utf-8')); parser.flush()
    if not any('Memory Types' in p for p in parser.parts): return None
    facts=[]; seen=set()
    for quote in parser.parts:
        for k,g,t,a,b in COMPILED:
            if sum(f['key']==k for f in facts)>=3 or (k,quote) in seen: continue
            if useful_quote(quote,k,a,b):
                facts.append({'key':k,'group':g,'title':t,'quote':quote})
                seen.add((k,quote))
    # RAM layout paragraphs include IRAM / DRAM rather than the word SRAM.
    for quote in parser.parts:
        if re.search(r'\b(?:IRAM|DRAM)\b',quote) and 70<len(quote)<1400 and not re.search(r'cookie|copyright',quote,re.I):
            if sum(f['key']=='ram_banks' for f in facts)<4 and ('ram_banks',quote) not in seen:
                facts.append({'key':'ram_banks','group':'ram','title':'RAM 分区与总线','quote':quote})
                seen.add(('ram_banks',quote))
    selected=[r for r in rows if r['manufacturer']=='Espressif' and r['series']==series]
    return key,{'id':key,'url':url,'title':f'{series} ESP-IDF Memory Types','kind':'user_manual',
                'manufacturer':'Espressif','devices':[r['device_id'] for r in selected],
                'scope':[series+'（芯片内存与 SDK 使用说明；模组 Flash/PSRAM 容量依具体后缀）'],
                'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'facts':facts,'figures':[]}

def main():
    rows=read_rows(ROOT/'data/combined/device-variants.csv')
    cache=ROOT/'cache/memory-pages'; cache.mkdir(exist_ok=True)
    payload_path=ROOT/'data/combined/memory-evidence.json'
    payload=json.loads(payload_path.read_text(encoding='utf-8'))
    docs=payload['documents']
    # Replace this collector's vendor subset atomically on every run.
    docs={k:v for k,v in docs.items() if v['manufacturer'] not in ('Espressif','Qinheng')}
    payload['documents']=docs
    series=['ESP32','ESP32-S2','ESP32-S3','ESP32-C2/ESP8684','ESP32-C3','ESP32-C5','ESP32-C6','ESP32-C61','ESP32-H2','ESP32-P4']
    with ThreadPoolExecutor(max_workers=4) as pool:
        for result in pool.map(lambda s:collect_esp(s,rows,cache),series):
            if result:
                key,doc=result
                if key in docs: doc['devices']=sorted(set(doc['devices']+docs[key]['devices']))
                docs[key]=doc
                print(doc['title'],len(doc['facts']),flush=True)
    for path in (ROOT/'cache/qinheng-official').glob('ch32*.json'):
        article=json.loads(path.read_text(encoding='utf-8')).get('data') or {}
        content=article.get('content',''); alias=article.get('alias','')
        url=f'https://www.wch.cn/products/{alias}.html'
        selected=[r for r in rows if r['manufacturer']=='Qinheng' and r['source_url'].lower()==url.lower()]
        if not selected: continue
        doc_key=hashlib.sha256(url.encode()).hexdigest()[:16]
        match=re.search(r'####\s*系统框图\s*!\[[^\]]*\]\((https?://[^)]+)\)',content)
        figures=[]
        if match:
            image_url=match[1].replace('http://','https://')
            if not image_url.startswith('https://img.wch.cn/'): continue
            original=cache/(doc_key+Path(image_url).suffix)
            if download(image_url,original):
                # Normalize the official image to a portable PNG without
                # changing content; record the original URL alongside it.
                import pymupdf as fitz
                image=fitz.Pixmap(str(original))
                name=f'memory-arch-{doc_key}-p1.png'
                image.save(ROOT.parent/'mcu-l-android/assets'/name)
                figures=[{'title':f'WCH {alias} 官方系统框图','image':name,'source_image':image_url}]
        facts=[]
        for quote in content.splitlines():
            if not re.search(r'Flash|SRAM|RAM|闪存|存储器',quote,re.I): continue
            quote=quote.lstrip('- ').strip()
            if not 15<len(quote)<1000: continue
            is_flash=bool(re.search(r'Flash|闪存|ROM|只读',quote,re.I))
            fact_key='flash_layout' if is_flash else ('ram_banks' if re.search(r'bank|AHB|AXI|SRAM|RAM|内存|存储',quote,re.I) else 'ram_access')
            group='flash' if is_flash else 'ram'
            title='Flash 布局与启动区' if is_flash else '片上存储与分区'
            facts.append({'key':fact_key,'group':group,'title':title,'quote':quote})
            if is_flash and re.search(r'SRAM|(?<![A-Za-z])RAM',quote,re.I):
                facts.append({'key':'ram_banks','group':'ram','title':'片上存储与分区','quote':quote})
        if figures or facts:
            docs[doc_key]={'id':doc_key,'url':url,'title':f'WCH {alias} 产品说明','kind':'product_page','manufacturer':'Qinheng',
                       'devices':[r['device_id'] for r in selected],'scope':sorted({r['product_line'] for r in selected}),
                       'sha256':hashlib.sha256((original.read_bytes() if figures and original.exists() else content.encode())).hexdigest(),'facts':facts,'figures':figures}
    payload_path.write_text(json.dumps(payload,ensure_ascii=False,separators=(',',':'))+'\n',encoding='utf-8')
    print('Updated ESP / WCH sources',sum(d['manufacturer'] in ('Espressif','Qinheng') for d in docs.values()))

if __name__=='__main__': main()

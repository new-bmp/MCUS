#!/usr/bin/env python3
"""Collect page-level Flash/RAM evidence and official architecture figures.

Document-level statements retain their full conditions and model scope. They
are never flattened into unconditional device bank/wait/ECC/count fields.
Sources are attached by exact catalog documents or vendor document selectors;
we do not generate datasheet URLs from guessed device suffixes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import pymupdf as fitz

from augment_packages_from_official_sources import artery_document_records

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = ('st.com', 'renesas.com', 'microchip.com', 'infineon.com', 'gd32mcu.com',
            'arterytek.com', 'hpmicro.com', 'espressif.com', 'wch.cn', 'wch-ic.com',
            'nuvoton.com', 'geehy.com', 'mindmotion.com.cn', 'puyasemi.com',
            'nationstech.com', 'raspberrypi.com', 'raspberrypi.org', 'ti.com',
            'allwinnertech.com')

# Labels describe the evidence category, not an inferred device-wide boolean.
RULES = [
    ('flash_banks', 'flash', 'Bank / 分区与读写', r'\bflash\b|闪存', r'dual.?bank|single.?bank|two banks|read.while.write|bank swap|双区|双\s*bank|同时读写'),
    ('flash_latency', 'flash', '等待周期与加速条件', r'\bflash\b|闪存', r'wait.?state|zero.wait|latency|prefetch|\bART accelerator|NZW|等待|预取'),
    ('flash_ecc', 'flash', 'Flash 校验与纠错', r'\bflash\b|闪存', r'\bECC\b|error.correct|纠错|奇偶校验'),
    ('flash_erase', 'flash', '编程 / 擦除粒度', r'\bflash\b|闪存', r'(?:page|sector|block|mass).{0,25}eras|eras.{0,25}(?:page|sector|block)|programming.{0,20}(?:bit|byte|word)|页擦除|扇区|块擦除|编程粒度'),
    ('flash_endurance', 'flash', '擦写寿命与数据保持', r'\bflash\b|闪存|endurance|[km]?cycles', r'endurance|data retention|erase.{0,20}cycles|擦写寿命|数据保持'),
    ('flash_protection', 'flash', 'Flash 保护与安全', r'\bflash\b|闪存', r'read.?protect|write.?protect|proprietary code|PCROP|secure|读保护|写保护|安全'),
    ('flash_layout', 'flash', '代码 / 数据 / 启动区', r'\bflash\b|闪存', r'data flash|code flash|boot flash|bootloader|boot loader|EEPROM|数据闪存|程序闪存|启动'),
    ('flash_xip', 'flash', '外部 Flash 与 XIP', r'\bflash\b|闪存', r'external|off.chip|execute.in.place|\bXIP\b|QSPI|OctoSPI|FlexSPI|片外|外部'),
    ('ram_tcm', 'ram', 'TCM / CCM / 本地存储', r'(?<![A-Za-z0-9])[DI]?TCM(?![A-Za-z0-9])|(?<![A-Za-z0-9])CCM(?![A-Za-z0-9])|(?<![A-Za-z0-9])[ID]LM\d?(?![A-Za-z0-9])|tightly.coupled|紧耦合', r'RAM|memory|SRAM|bus|core|instruction|data|内存|存储|总线|核'),
    ('ram_ecc', 'ram', 'RAM 校验与纠错', r'\bS?RAM\d?\b|内存', r'\bECC\b|parity|error.correct|纠错|奇偶'),
    ('ram_retention', 'ram', 'RAM 低功耗保持', r'\bS?RAM\d?\b|内存', r'retention|retained|retain|backup|standby|low.power|保持|备份|待机|低功耗'),
    ('ram_access', 'ram', 'RAM 共享与访问限制', r'\bS?RAM\d?\b|\b[DI]?TCM\b|\bCCM\b|内存', r'DMA|shared|exclusive|private|dedicated|accessible|accessed|共享|独占|专用|访问'),
    ('ram_banks', 'ram', 'RAM 分区与总线', r'\bS?RAM\d?\b|内存', r'bank|AHB|AXI|SRAM[1-9]|D[123] domain|分区|总线|地址|configur'),
    ('ram_external', 'ram', '外部 / 封装内 RAM', r'PSRAM|SDRAM|pseudo.static|伪静态', r'memory|RAM|external|embedded|integrated|in.package|存储|内存|外部|集成'),
    ('memory_cache', 'system', '指令 / 数据 Cache', r'cache|缓存', r'instruction|data|[ID].cache|associat|write.through|write.back|指令|数据'),
]
COMPILED = [(k, g, t, re.compile(a, re.I), re.compile(b, re.I)) for k,g,t,a,b in RULES]
FIGURE = re.compile(r'^(?:Figure\s*\d+[.\-:]\s*.{0,70}(?:block diagram|system architecture|bus (?:matrix|architecture)))|^(?:图\s*\d+[.\-:]?\s*.{0,35}(?:框图|架构|结构图))', re.I)

def read_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))

def safe_url(url):
    p = urlsplit(url)
    return p.scheme == 'https' and any(p.hostname == h or (p.hostname or '').endswith('.'+h) for h in OFFICIAL)

def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def discover(rows):
    docs = {}
    def add(row, url, title, kind='datasheet', cached=None):
        url = re.sub(r'(?i)(?:\.pdf){2,}$','.pdf',url)
        if not safe_url(url):
            return
        key = hashlib.sha256(url.encode()).hexdigest()[:16]
        doc = docs.setdefault(key, {'id':key, 'url':url, 'title':title, 'kind':kind,
                                   'manufacturer':row['manufacturer'], 'devices':set(), 'scope':set()})
        doc['devices'].add(row['device_id'])
        doc['scope'].add(row['product_line'])
        if cached and cached.exists():
            doc['cached'] = str(cached)
    for row in rows:
        for d in json.loads(row.get('documents_json') or '[]'):
            if not isinstance(d, dict): continue
            url = d.get('url') or d.get('name') or ''
            title = d.get('title') or d.get('name') or row['product_line']
            if not isinstance(url, str) or not safe_url(url): continue
            if d.get('kind') in ('application_note', 'package_drawing'): continue
            # Some official Chinese vendor portals serve a PDF from a download
            # endpoint without a `.pdf` suffix.  Keep these exact links in the
            # evidence pipeline; fetch_document still requires the response to
            # have a real PDF signature before parsing it.
            parsed_url = urlsplit(url)
            download_endpoint = (
                row['manufacturer'] == 'Nationz' and parsed_url.hostname and
                parsed_url.hostname.endswith('nationstech.com') and
                ('dowfile' in parsed_url.path.lower() or 'download' in parsed_url.path.lower())
            ) or (
                row['manufacturer'] == 'MindMotion' and parsed_url.hostname and
                parsed_url.hostname.endswith('mindmotion.com.cn') and
                ('download' in parsed_url.path.lower() or parsed_url.path.lower().endswith('.aspx'))
            )
            if not (parsed_url.path.lower().endswith('.pdf') or '/document/dst/' in url or '/document/mah/' in url or download_endpoint): continue
            if re.search(r'errata|schematic|evaluation|EVK|Generic User Guide|HAL Drivers|low.layer driver|firmware|software|application note|应用笔记|cortex.m\d+.?(?:trm|dgug)', title+' '+url, re.I): continue
            kind = 'reference_manual' if 'reference' in title.lower() or '/reference_manual/' in url or d.get('kind')=='reference_manual' else 'datasheet'
            cached = None
            if row['manufacturer']=='STMicroelectronics' and '/datasheet/' in url:
                cached = ROOT/'cache/st-datasheets/pdf'/(Path(urlsplit(url).path).stem+'-'+hashlib.sha1(url.encode()).hexdigest()[:8]+'.pdf')
            if row['manufacturer']=='MicroPy MCU':
                cached = ROOT/'cache/micropy-mcu'/Path(urlsplit(url).path).name
            add(row, url, title, kind, cached)
    # Recover previously downloaded exact vendor selector documents even when a
    # later catalog merge omitted their document entries.
    by_line = defaultdict(list)
    for row in rows: by_line[(row['manufacturer'],row['product_line'].upper())].append(row)
    api = ROOT/'cache/artery-documents/Document.json'
    if api.exists():
        for line, entries in artery_document_records(api.read_bytes()).items():
            for d in entries:
                cache = ROOT/'cache/artery-documents/pdf'/('ds-'+line.lower()+'.pdf') if d['kind']=='datasheet' else None
                for row in by_line[('Artery', line)]: add(row,d['url'],d['title'],d['kind'],cache)
    report = json.loads((ROOT/'data/combined/package-augmentation-report.json').read_text(encoding='utf-8'))
    pdf_by_hash = {digest(p):p for p in (ROOT/'cache/gigadevice-datasheets/pdf').glob('*.pdf')}
    for d in report.get('documents',[]):
        match = re.search(r'(GD32[A-Z0-9]+?)x',d['title'],re.I)
        if not match: continue
        for row in by_line[('GigaDevice',match[1].upper())]: add(row,d['url'],d['title'],cached=pdf_by_hash.get(d.get('sha256')))
    for doc in docs.values():
        doc['devices']=sorted(doc['devices']); doc['scope']=sorted(doc['scope'])
    return docs

def fetch_document(doc, cache, offline):
    path = Path(doc.get('cached',cache/(doc['id']+'.pdf')))
    if path.exists() and path.read_bytes()[:4]==b'%PDF': return doc,path,None
    if offline: return doc,None,'not_cached'
    path = cache/(doc['id']+'.pdf')
    result = subprocess.run(['curl.exe','-L','--fail','--silent','--show-error','--max-time','28',
                             '-A','Mozilla/5.0 (MCUS official memory evidence)', '-o',str(path),doc['url']],capture_output=True,timeout=35)
    if result.returncode or not path.exists() or path.read_bytes()[:4]!=b'%PDF':
        return doc,None,'download_failed_or_not_pdf'
    return doc,path,None

def paragraphs(page):
    """Join physically adjacent wrapped lines without combining table columns."""
    lines=[]
    for block in page.get_text('dict')['blocks']:
        for line in block.get('lines',[]):
            q=''.join(s['text'] for s in line['spans']).strip()
            x0,y0,x1,y1=line['bbox']
            if not q or y0<page.rect.height*.06 or y1>page.rect.height*.945: continue
            lines.append((x0,y0,x1,y1,q))
    # A number of vendors export every wrapped line as a separate PDF block.
    # Reconstruct columns before joining those lines.  Bullet glyphs sharing
    # the same baseline belong to their following text, not a new column.
    mid=page.rect.width/2
    right=[l for l in lines if l[0]>mid and len(l[4])>15]
    left=[l for l in lines if l[2]<mid+12 and len(l[4])>15]
    two_columns=sum(any(abs(l[1]-r[1])<4 for l in left) for r in right)>=4
    groups=[[l for l in lines if l[0]<mid],[l for l in lines if l[0]>=mid]] if two_columns else [lines]
    all_paragraphs=[]
    for group in groups:
        group.sort(key=lambda l:(round(l[1]/3),l[0]))
        merged=[]
        for line in group:
            if merged and abs(line[1]-merged[-1][1])<3 and 0<=line[0]-merged[-1][2]<35:
                prev=merged.pop(); merged.append((prev[0],prev[1],line[2],max(prev[3],line[3]),prev[4]+' '+line[4]))
            else: merged.append(line)
        current=''; prev=None
        for line in merged:
            x0,y0,x1,y1,q=line
            heading=bool(re.match(r'^(?:\d+(?:\.\d+){1,4}\s|Table\s|Figure\s|[•●■]|[–−]\s)',q))
            same=prev and abs(x0-prev[0])<33 and -4<=y0-prev[3]<9
            if current and same and not heading and not re.search(r'[.!?。；;:]$',current) and len(current)<1500:
                current += ' '+q
            else:
                if current: all_paragraphs.append(current)
                current=q
            prev=line
        if current: all_paragraphs.append(current)
    return all_paragraphs

def useful_quote(q,key,a,b):
    minimum=20 if re.search(r'[\u4e00-\u9fff]',q) else 40
    if len(q)<minimum or len(q)>1600 or re.search(r'(?:\.\s*){4,}|…{2,}',q): return False
    if len(re.findall(r'0x[0-9a-f]+',q,re.I))>2 or q.startswith('Features '): return False
    if not re.search(r'[\u4e00-\u9fff]',q) and not re.search(r'[.!]$',q) and not re.match(r'^[•●■–−]',q): return False
    if re.match(r'^(?:of |registers\s*\+|and |or |which |sLib,)',q): return False
    if re.search(r'\b(?:is|the|maximum|from|and|with|of|can|which|to|by)\s*$',q): return False
    if re.match(r'^(?:Table\s+\d|Figure\s+\d|表\s*\d|图\s*\d)',q,re.I): return False
    if re.search(r'updated|modified|revision history|added to|changed from|replaced by|contents|list of (tables|figures)',q,re.I): return False
    if not a.search(q) or not b.search(q): return False
    if key=='ram_tcm' and re.search(r'AES|encryption|authentication|GCM',q,re.I): return False
    if key=='ram_tcm' and re.search(r'CoreMark|IDD|current consumption',q,re.I): return False
    if key=='flash_ecc' and re.search(r'NAND|external flash|off.chip',q,re.I): return False
    # A feature paragraph often mentions embedded Flash and an ECC-protected
    # SRAM in the same sentence.  Keep Flash ECC only when the two terms are
    # close enough to describe the same array, rather than the other memory.
    if key=='flash_ecc' and not any(abs(x.start()-y.start())<75 for x in a.finditer(q) for y in b.finditer(q)): return False
    if key=='flash_ecc' and any(
        re.search(r'\b(?:s?ram|memory)\b', q[min(x.start(), y.start()):max(x.end(), y.end())], re.I)
        for x in a.finditer(q) for y in b.finditer(q)
    ):
        # Mixed summaries such as "Flash ... SRAM where 64KB has ECC" do not
        # identify Flash ECC and must remain document context only.
        return False
    if key.startswith('ram_') and not any(abs(x.start()-y.start())<110 for x in a.finditer(q) for y in b.finditer(q)): return False
    if key=='ram_banks' and not re.search(r'(?:ram\d?|sram\d?)[^.;]{0,65}(?:bank|AHB|AXI|bus|分区|总线|地址|configur)|(?:bank|AHB|AXI)[^.;]{0,45}(?:ram|sram)',q,re.I): return False
    if key=='flash_xip' and not re.search(r'(?:external|off.chip|QSPI|XIP|片外|外部)[^.;]{0,55}(?:flash|闪存)|(?:flash|闪存)[^.;]{0,55}(?:external|off.chip|QSPI|XIP|片外|外部)',q,re.I): return False
    if key=='flash_endurance' and re.search(r'trademarks?|Adjacent Key Suppression|Total Endurance,|Microchip Technology Incorporated',q,re.I): return False
    if key=='flash_endurance' and not re.search(r'\b\d[\d,.]*\s*(?:k|m)?\s*(?:cycles?|次)|\b\d+\s*(?:years?|年)|data retention[^.]{0,100}\d|erase[^.]{0,80}cycles|擦写寿命|数据保持',q,re.I): return False
    if len(re.findall(r'\b(?:ADC|UART|USART|I2C|SPI|GPIO|TIM\d)\b',q,re.I))>5: return False
    return True

def architecture_caption(caption,doc):
    if not FIGURE.search(caption) or len(caption)>240: return False
    label=re.sub(r'^(?:Figure|图)\s*\d+(?:[.\-]\d+)*[.:：]?\s*','',caption,flags=re.I)
    if re.match(r'shows|illustrates|depicts',label,re.I): return False
    if re.search(r'power|supply|clock|oscillator|电源|供电|时钟|振荡|MCLK|CRC|RTC|sensorless|PMSM|SC/I2S|interrupt|radio system|GTZC|DMA block|定时|中断',label,re.I): return False
    if doc['kind']=='reference_manual':
        # Reference manuals routinely cover dies with different buses. Only
        # retain explicitly memory-related diagrams here; device datasheets
        # supply the overall architecture for the selected variant.
        return bool(re.search(r'memory|SRAM|Flash|存储',label,re.I))
    return bool(re.search(r'architecture|system|bus matrix|interconnect|MCU|memory|系统|架构|存储',label,re.I) or re.match(r'block diagram\b',label,re.I) or any(line.lower() in label.lower() for line in doc['scope']))

def extract_document(doc,path,assets):
    pdf = fitz.open(path)
    facts,figures=[],[]
    for pi in range(min(len(pdf),200)):
        page=pdf[pi]
        blocks=page.get_text('blocks')
        # PyMuPDF preserves paragraph boundaries; do not cross unrelated RAM
        # and Flash bullets to claim that both have the same ECC or latency.
        for raw in paragraphs(page):
            q=' '.join(raw.replace('\u00ad','').split())
            for key,group,title,a,b in COMPILED:
                if (pi==0 and len(q)<90) or not useful_quote(q,key,a,b): continue
                if doc['kind']=='reference_manual' and re.search(r'bank 2|BFB2',q,re.I) and not re.search(r'XL.density|dual.bank|STM32\w+',q,re.I): continue
                if any(f['key']==key and f['quote']==q for f in facts): continue
                facts.append({'key':key,'group':group,'title':title,'quote':q,'page':pi+1})
        # A real figure has graphics and its caption; table-of-contents hits
        # are excluded. Keep the full page, so notes and model restrictions stay.
        if len(figures)<3:
            caption=next((' '.join(str(b[4]).split()) for b in blocks if FIGURE.search(' '.join(str(b[4]).split())) and not re.search(r'(?:\.\s*){4,}',str(b[4]))),None)
            # A datasheet often contains several block diagrams.  Power,
            # clock and peripheral-only diagrams are useful elsewhere but do
            # not describe the MCU memory architecture requested here.
            if caption and not architecture_caption(caption,doc):
                caption=None
            if caption and len(caption)<240 and (len(page.get_drawings())>15 or page.get_images()):
                filename=f"memory-arch-{doc['id']}-p{pi+1}.png"
                target=assets/filename
                if not target.exists():
                    scale=1500/max(page.rect.width,page.rect.height)
                    page.get_pixmap(matrix=fitz.Matrix(scale,scale),alpha=False).save(target)
                figures.append({'title':caption,'page':pi+1,'image':filename})
                # H742 and H743, for example, have distinct figures within the
                # same datasheet. Expose only the figure naming this product.
                expanded=re.sub(r'([A-Z]+\d*[A-Z]+)(\d{2,4})/(\d{2,4})',r'\1\2 \1\3',caption,flags=re.I)
                named=[line for line in doc['scope'] if line.lower() in expanded.lower()]
                if named: figures[-1]['lines']=named
    # Prefer full descriptive paragraphs over a terse cover-page feature or
    # isolated table heading.  Keep distinct conditions verbatim.
    chosen=[]
    for key,group,title,a,b in COMPILED:
        candidates=[f for f in facts if f['key']==key]
        candidates.sort(key=lambda f:(
            not bool(re.search(r'[.!。]$',f['quote'])),
            f['page']<=2,
            len(f['quote'])<80,
            f['page']))
        chosen.extend(candidates[:3])
    facts=chosen
    result={k:v for k,v in doc.items() if k!='cached'}
    result.update(sha256=digest(path),pages=len(pdf),scanned_pages=min(len(pdf),200),facts=facts,figures=figures)
    pdf.close()
    return result

def parse_cached(doc,path,assets,cache,parser_hash):
    parsed_path=cache/(doc['id']+'-extracted.json')
    if parsed_path.exists():
        previous=json.loads(parsed_path.read_text(encoding='utf-8'))
        if previous.get('parser_hash')==parser_hash:
            previous.update({k:v for k,v in doc.items() if k!='cached'})
            return previous
    result=extract_document(doc,path,assets)
    result['parser_hash']=parser_hash
    parsed_path.write_text(json.dumps(result,ensure_ascii=False),encoding='utf-8')
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offline',action='store_true')
    parser.add_argument('--limit-new',type=int,default=260)
    parser.add_argument('--vendors',default='')
    args=parser.parse_args()
    rows=read_rows(ROOT/'data/combined/device-variants.csv')
    docs=discover(rows)
    cache=ROOT/'cache/memory-evidence'; cache.mkdir(parents=True,exist_ok=True)
    assets=ROOT.parent/'mcu-l-android/assets'
    selected=[]; news=Counter()
    for doc in sorted(docs.values(),key=lambda d:(d['kind']!='datasheet',-len(d['devices']),d['id'])):
        if args.vendors and doc['manufacturer'] not in args.vendors.split(','): continue
        if not doc.get('cached') and not (cache/(doc['id']+'.pdf')).exists():
            if sum(news.values())>=args.limit_new or news[doc['manufacturer']]>=120: continue
            news[doc['manufacturer']]+=1
        selected.append(doc)
    print(f"Discovered {len(docs)} official documents; selected {len(selected)}, new attempts {dict(news)}",flush=True)
    successes={}; errors=[]
    ready=[]
    with ThreadPoolExecutor(max_workers=6) as pool:
        pending={pool.submit(fetch_document,d,cache,args.offline):d for d in selected}
        for i,future in enumerate(as_completed(pending),1):
            original=pending[future]
            try:
                doc,path,error=future.result()
                if error:
                    errors.append({'url':doc['url'],'manufacturer':doc['manufacturer'],'error':error}); continue
                ready.append((doc,path))
            except Exception as e:
                errors.append({'url':original['url'],'error':str(e)})
            if i%50==0: print(f"Fetched/cached {i}/{len(selected)}",flush=True)
    parser_hash=digest(Path(__file__))
    # PyMuPDF documents must not be shared between threads. Independent worker
    # processes parse independent documents and checkpoint each verified result.
    with ProcessPoolExecutor(max_workers=4) as pool:
        pending={pool.submit(parse_cached,d,p,assets,cache,parser_hash):d for d,p in ready}
        for i,future in enumerate(as_completed(pending),1):
            original=pending[future]
            try:
                result=future.result()
                if result['facts'] or result['figures']: successes[original['id']]=result
            except Exception as e: errors.append({'url':original['url'],'error':str(e)})
            if i%40==0: print(f"Parsed {i}/{len(ready)}, retained {len(successes)}",flush=True)
    # Partial vendor runs merge only by document ID, preserving other work.
    output=ROOT/'data/combined/memory-evidence.json'
    if args.vendors and output.exists():
        previous=json.loads(output.read_text(encoding='utf-8'))['documents']
        for k,v in previous.items():
            if v['manufacturer'] not in args.vendors.split(','): successes[k]=v
    coverage={}
    for vendor in sorted({r['manufacturer'] for r in rows}):
        subset=[d for d in successes.values() if d['manufacturer']==vendor]
        coverage[vendor]={'total':sum(r['manufacturer']==vendor for r in rows),'documents':len(subset),
            'flash_devices':len({i for d in subset if any(f['group']=='flash' for f in d['facts']) for i in d['devices']}),
            'ram_devices':len({i for d in subset if any(f['group']=='ram' for f in d['facts']) for i in d['devices']}),
            'diagram_devices':len({i for d in subset if d['figures'] for i in d['devices']})}
    payload={'generated_at':datetime.now(timezone.utc).isoformat(),'scope':'official_document_evidence_with_conditions',
             'documents':dict(sorted(successes.items()))}
    output.write_text(json.dumps(payload,ensure_ascii=False,separators=(',',':'))+'\n',encoding='utf-8')
    report={'documents':len(successes),'facts':sum(len(d['facts']) for d in successes.values()),
            'diagrams':sum(len(d['figures']) for d in successes.values()),'coverage':coverage,'errors':errors}
    (ROOT/'data/combined/memory-evidence-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='errors'},ensure_ascii=False,indent=2),flush=True)
    print(f'Failed downloads/parses: {len(errors)}',flush=True)

if __name__=='__main__': main()

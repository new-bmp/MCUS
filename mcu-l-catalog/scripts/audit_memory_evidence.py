#!/usr/bin/env python3
"""Audit the final PDF + vendor-page snapshot, including displayed figures."""
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from collect_memory_evidence import ROOT, read_rows, safe_url


def main():
    source=ROOT/'data/combined/memory-evidence.json'
    payload=json.loads(source.read_text(encoding='utf-8'))
    docs=payload['documents']
    rows=read_rows(ROOT/'data/combined/device-variants.csv')
    devices={r['device_id']:r for r in rows}
    assets=ROOT.parent/'mcu-l-android/assets'
    failures=[]; pictures=set(); coverage={}; categories=Counter()
    for key,doc in docs.items():
        if key!=doc['id'] or not re.fullmatch('[a-f0-9]{16}',key): failures.append(f'invalid document id: {key}')
        if not safe_url(doc['url']): failures.append(f'nonofficial URL: {key}')
        if not doc['devices']: failures.append(f'no devices: {key}')
        for device in doc['devices']:
            if device not in devices or devices[device]['manufacturer']!=doc['manufacturer']:
                failures.append(f'invalid device link: {key} / {device}')
        for fact in doc['facts']:
            categories[fact['key']]+=1
            if not fact['quote'].strip(): failures.append(f'empty quote: {key}')
            if 'pages' in doc and not 1<=fact.get('page',0)<=doc['pages']: failures.append(f'page outside PDF: {key}')
        for figure in doc['figures']:
            image=figure['image']
            if not re.fullmatch(r'memory-arch-[a-f0-9]+-p\d+\.png',image): failures.append(f'unsafe image: {image}'); continue
            if not (assets/image).is_file(): failures.append(f'missing image: {image}')
            pictures.add(image)
    all_flash=set(); all_ram=set(); all_diagram=set(); any_evidence=set()
    for vendor in sorted({r['manufacturer'] for r in rows}):
        subset=[d for d in docs.values() if d['manufacturer']==vendor]
        flash={i for d in subset if any(f['group']=='flash' for f in d['facts']) for i in d['devices']}
        ram={i for d in subset if any(f['group']=='ram' for f in d['facts']) for i in d['devices']}
        diagram={i for d in subset for i in d['devices'] if any(not f.get('lines') or devices[i]['product_line'] in f['lines'] for f in d['figures'])}
        evidence={i for d in subset for i in d['devices']}
        coverage[vendor]={'total':sum(r['manufacturer']==vendor for r in rows),'documents':len(subset),
                         'flash_devices':len(flash),'ram_devices':len(ram),'diagram_devices':len(diagram),'with_evidence':len(evidence)}
        all_flash|=flash; all_ram|=ram; all_diagram|=diagram; any_evidence|=evidence
    report_path=ROOT/'data/combined/memory-evidence-report.json'
    previous=json.loads(report_path.read_text(encoding='utf-8')) if report_path.exists() else {}
    report={'generated_at':datetime.now(timezone.utc).isoformat(),'documents':len(docs),'facts':sum(len(d['facts']) for d in docs.values()),
            'diagrams':len(pictures),'flash_devices':len(all_flash),'ram_devices':len(all_ram),'diagram_devices':len(all_diagram),
            'devices_with_evidence':len(any_evidence),'coverage':coverage,'categories':dict(categories),'errors':previous.get('errors',[]),
            'audit_failures':failures,'scope':'Document excerpts with conditions; this is not per-field scalar completeness. PDF scan capped at 200 pages.'}
    report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('coverage','categories','errors')},ensure_ascii=False,indent=2))
    if failures: raise SystemExit(1)


if __name__=='__main__': main()

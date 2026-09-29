"""Selection-critical regression checks against cached official exact rows."""
import csv
import json
import unittest
from pathlib import Path
from import_nationz_official import processors, row_features
from import_mindmotion_selector import table_rows

ROOT=Path(__file__).resolve().parents[1]

def rows(vendor):
    with (ROOT/'data/vendor-packs'/vendor/'device-variants.csv').open(encoding='utf-8-sig',newline='') as f:
        return {r['device_name']:r for r in csv.DictReader(f)}

class VendorDataTests(unittest.TestCase):
    def test_g401_does_not_inherit_erroneous_pack_core(self):
        d=rows('nationz')['N32G401F8Q7']
        p=json.loads(d['processor_cores'])[0]
        self.assertEqual((p['Dcore'],p['Dfpu']),('Cortex-M4','1'))
        self.assertEqual(int(d['flash_bytes']),64*1024)
        self.assertEqual(d['pin_counts'],'20')

    def test_h785_multicore_and_adc_oversampling_are_distinct(self):
        d=rows('nationz')['N32H785XIB7']
        self.assertEqual([int(p['Dclock']) for p in json.loads(d['processor_cores'])],[600000000,300000000])
        f=json.loads(d['features_json'])
        units=next(v for v in f if v['type']=='ADCUnits')
        channels=next(v for v in f if v['type']=='ADC')
        self.assertEqual((units['count'],units['m'],channels['count']),('3','12','53'))
        self.assertEqual(d['pin_counts'],'265')

    def test_dma_controllers_are_not_reported_as_channels(self):
        f=row_features({'DMA/通道数':'3/24','MDMA/Channels':'1/16'})
        self.assertEqual(next(v['count'] for v in f if v['type']=='DMA'),'24')
        self.assertEqual(next(v['count'] for v in f if v['type']=='DMAControllers'),'3')
        self.assertEqual(next(v['count'] for v in f if v['type']=='MDMAChannels'),'16')

    def test_no_synthetic_suffixes_and_all_new_sources_retained(self):
        source=json.loads((ROOT/'data/vendor-packs/nationz/selector-records.json').read_text(encoding='utf-8'))
        self.assertEqual(set(rows('nationz')),{r['产品型号'] for r in source})
        for d in rows('nationz').values():
            self.assertTrue(d['source_id'] and d['source_version'])
            self.assertTrue(d['package_types'])

    def test_mm32_header_alignment_core_and_package(self):
        devices=rows('mindmotion')
        d=devices['MM32H5487L7PV']
        self.assertEqual((d['architecture_class'],d['max_clock_hz'],d['package_types']),('STAR-MC1','300000000','LQFP64'))
        self.assertEqual(devices['MM32G0003B1NV']['package_types'],'QFN20')
        self.assertEqual(devices['MM32G0003B1NV']['architecture_class'],'Cortex-M0+')
        channels=next(f for f in json.loads(devices['MM32G0003B1NV']['features_json']) if f['type']=='ADC')
        self.assertEqual((channels['count'],channels['m']),('9','12'))
        self.assertIn('channels',channels['name'])
        self.assertIn('MM32F103C8T',list(devices))
        self.assertTrue(all(not d['architecture_class'].startswith(('M0','M3')) for d in devices.values()))

    def test_unlabelled_adc_quantity_never_becomes_converter_count(self):
        d=rows('mindmotion')['MM32H5487L7PV']
        f=json.loads(d['features_json'])
        self.assertFalse(any(v['type']=='ADCUnits' for v in f))
        self.assertTrue(any(v['type']=='ADC' and v['count']=='18' for v in f))

if __name__=='__main__':unittest.main()

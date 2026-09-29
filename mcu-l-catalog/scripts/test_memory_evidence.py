import importlib.util
import json
import unittest
from pathlib import Path

import pymupdf as fitz
from collect_memory_evidence import COMPILED, FIGURE, paragraphs, useful_quote, architecture_caption, discover

root=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('generate_catalog',root/'mcu-l-android/scripts/generate_catalog.py')
generator=importlib.util.module_from_spec(spec); spec.loader.exec_module(generator)


class MemoryEvidenceTest(unittest.TestCase):
    def keys(self,quote):
        return {k for k,g,t,a,b in COMPILED if useful_quote(quote,k,a,b)}

    def test_graph_reference_is_not_a_graph_caption(self):
        self.assertIsNone(FIGURE.search('Figure 1 shows the general block diagram of the device family.'))
        self.assertIsNotNone(FIGURE.search('Figure 2. STM32H743xI/G block diagram'))
        doc={'kind':'datasheet','scope':['STM32H743']}
        self.assertFalse(architecture_caption('Figure 3-1 shows a functional block diagram of the device.',doc))
        self.assertFalse(architecture_caption('Figure 2. CRC calculation unit block diagram',doc))
        self.assertTrue(architecture_caption('Figure 2. STM32H743xI/G block diagram',doc))

    def test_subscript_line_overlap_keeps_wait_conditions(self):
        class Page:
            rect=fitz.Rect(0,0,595,842)
            def get_text(self,kind):
                return {'blocks':[{'lines':[
                    {'bbox':(120,440,520,454),'spans':[{'text':'The Flash memory needs 0 wait states below 24 MHz,'}]},
                    {'bbox':(120,453,520,465),'spans':[{'text':'1 wait state up to 48 MHz and 2 wait states above.'}]}]}]}
        q=paragraphs(Page())
        self.assertEqual(len(q),1)
        self.assertIn('2 wait states above.',q[0])

    def test_columns_do_not_mix_flash_and_ram_conditions(self):
        pdf=fitz.open(); page=pdf.new_page()
        for i in range(5):
            page.insert_text((50,100+16*i),f'Flash line {i} continues',fontsize=10)
            page.insert_text((340,100+16*i),f'RAM column {i} continues',fontsize=10)
        for q in paragraphs(page):
            self.assertFalse('Flash' in q and 'RAM' in q)
        pdf.close()

    def test_contents_crypto_and_ram_voltage_are_not_flash_attributes(self):
        self.assertFalse(self.keys('3.11 Chrom-ART Accelerator (DMA2D) . . . . . . . . . . . . . . . 34'))
        self.assertNotIn('ram_tcm',self.keys('The AES CCM mode is used for data authentication with a memory buffer.'))
        self.assertNotIn('flash_endurance',self.keys('Data retention supply voltage VDDDR 1.46 Note 5.5 V'))
        self.assertNotIn('flash_endurance',self.keys('Table 30. Flash memory endurance and data retention'))

    def test_conditions_and_negation_stay_verbatim(self):
        quote='The Flash memory uses 0 wait states up to 24 MHz, 1 wait state up to 48 MHz and 2 wait states above.'
        self.assertIn('flash_latency',self.keys(quote))
        self.assertIn('ram_access',self.keys('The CCM RAM is not accessible by the DMA controller.'))
        self.assertNotIn('flash_latency',self.keys('The SRAM is accessed at CPU speed with zero wait states.'))

    def test_sram_ecc_in_a_flash_summary_is_not_flash_ecc(self):
        quote=('The device has up to 2MB Flash memory, up to 256KB SRAM where '
               '64KB has ECC, and contains a QSPI interface.')
        self.assertNotIn('flash_ecc', self.keys(quote))

    def test_wrapped_paragraph_retains_last_line(self):
        pdf=fitz.open(); page=pdf.new_page()
        page.insert_text((95,100),'SRAM can be accessed by the CPU, but is',fontsize=11)
        page.insert_text((95,116),'not accessible by the DMA controller.',fontsize=11)
        page.insert_text((50,145),'2.2.4 External memory controller',fontsize=13)
        result=paragraphs(page)
        self.assertIn('SRAM can be accessed by the CPU, but is not accessible by the DMA controller.',result)
        pdf.close()

    def test_vendor_memory_formats_retain_addresses_types_and_aliases(self):
        source=[{'id':'IRAM1','size':'0x2000','start':'0x20000000','Pname':'CPU0'},
                {'name':'SRAM','bytes':32768,'type':'RAM'},
                {'name':'IRAM1_S','size':'0x2000','start':'0x30000000','alias':'IRAM1'},
                {'id':'FLASH','size':''}]
        result=generator.memory_regions(json.dumps(source))
        self.assertEqual(result[0],{'n':'IRAM1','s':8192,'start':'0x20000000','core':'CPU0'})
        self.assertEqual(result[1],{'n':'SRAM','s':32768,'type':'RAM'})
        self.assertEqual(result[2]['alias'],'IRAM1')
        self.assertNotIn('s',result[3])

    def test_official_vendor_download_endpoints_without_pdf_suffix_are_discovered(self):
        base={'device_id':'vendor::part','product_line':'PART','architecture_class':'Cortex-M4'}
        nationz={**base,'manufacturer':'Nationz','documents_json':json.dumps([{
            'kind':'datasheet','title':'Official datasheet',
            'url':'https://www.nationstech.com/api/dowfilebefore&did=7090'}])}
        mindmotion={**base,'device_id':'vendor::part2','manufacturer':'MindMotion','documents_json':json.dumps([{
            'kind':'datasheet','title':'Official datasheet',
            'url':'https://www.mindmotion.com.cn/download1.aspx?itemid=6080&typeid=5'}])}
        found=discover([nationz,mindmotion])
        self.assertEqual({doc['manufacturer'] for doc in found.values()},{'Nationz','MindMotion'})

if __name__=='__main__': unittest.main()

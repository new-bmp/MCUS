import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import vm from "node:vm";

function element() {
  return {
    textContent: "",
    innerHTML: "",
    hidden: false,
    dataset: {},
    classList: { add() {}, remove() {}, toggle() {} },
    style: { setProperty() {} },
    querySelector() { return null; },
    querySelectorAll() { return []; },
    addEventListener() {},
    remove() {},
    scrollIntoView() {},
    focus() {},
    appendChild() {},
    insertAdjacentHTML() {},
    getBoundingClientRect() { return { bottom: 0 }; },
    value: "",
  };
}

async function loadAssistant() {
  const modelSource = await readFile(new URL("../staticfiles/assistant-model.js", import.meta.url), "utf8");
  let appSource = await readFile(new URL("../staticfiles/app.js", import.meta.url), "utf8");
  appSource = appSource.replace(/\}\)\(\);\s*$/, "window.__MCUS_TEST__={aiParse,aiRecommend,safeHttpUrls};})();");

  const devices = [
    { id: "m33", m: "VendorA", f: "A", s: "A", l: "A", n: "M33-160", a: "Cortex-M33", c: "Cortex-M33", pt: "general_purpose_mcu", hz: 160000000, fl: 524288, ra: 262144, pin: "64", idx: 80, cov: 100, uart: 4, can: 2, tw: 16, tim: 1, adch: 16, adr: "16", pwr: [{ m: "run", v: 90, u: "uA", q: "typical", l: "Active current" }, { m: "run", v: 80, u: "uA_per_MHz", q: "typical", l: "Active current density" }, { m: "sleep", v: 800, u: "uA", q: "typical", l: "Sleep current" }], mem: [{ n: "Flash", s: 524288 }, { n: "ITCM", s: 65536 }, { n: "SRAM1", s: 196608 }], pi: [{ t: "Timer", n: "32-bit general purpose timer", d: "数量 1" }, { t: "ADC", n: "16-bit ADC", d: "采样率 2 MSPS" }] },
    { id: "m4", m: "VendorB", f: "B", s: "B", l: "B", n: "M4-240", a: "Cortex-M4", c: "Cortex-M4", pt: "general_purpose_mcu", hz: 240000000, fl: 524288, ra: 65536, pin: "64", idx: 82, cov: 100, uart: 4, can: 2, pi: [] },
    { id: "m7", m: "VendorC", f: "C", s: "C", l: "C", n: "M7-480", a: "Cortex-M7", c: "Cortex-M7", pt: "general_purpose_mcu", hz: 480000000, fl: 1048576, ra: 524288, pin: "144", idx: 95, cov: 100, uart: 8, can: 2, pi: [] },
    { id: "m0", m: "VendorD", f: "D", s: "D", l: "D", n: "M0-80", a: "Cortex-M0+", c: "Cortex-M0+", pt: "general_purpose_mcu", hz: 80000000, fl: 131072, ra: 32768, pin: "32", idx: 60, cov: 100, uart: 2, can: 1, pi: [] },
    { id: "at32", m: "Artery", f: "AT32", s: "AT32F", l: "AT32F435", n: "AT32F435CCT7", a: "Cortex-M4", c: "Cortex-M4", pt: "general_purpose_mcu", hz: 288000000, fl: 524288, ra: 65536, pin: "48", idx: 90, cov: 100, uart: 8, can: 2, usb: 2, pi: [] },
  ];
  const catalog = { meta: { devices: devices.length, manufacturers: 4, series: 4 }, coverage: [], devices };
  const elements = new Map();
  const document = {
    querySelector(selector) { if (!elements.has(selector)) elements.set(selector, element()); return elements.get(selector); },
    querySelectorAll() { return []; },
    addEventListener() {},
    documentElement: element(),
  };
  const localStorage = { getItem() { return null; }, setItem() {}, removeItem() {} };
  const location = { search: "", protocol: "file:", href: "file:///mcus" };
  const window = { MCU_CATALOG: catalog, MCUS_LOCAL_MODEL: null, MCUS_QUOTES_ENABLED: false, location, localStorage, visualViewport: null, addEventListener() {}, requestAnimationFrame(fn) { fn(); } };
  const context = { window, document, location, localStorage, URL, URLSearchParams, console, setTimeout, clearTimeout, requestAnimationFrame(fn) { fn(); }, AbortController, fetch() { throw new Error("not used"); } };
  vm.runInNewContext(modelSource, context);
  vm.runInNewContext(appSource, context);
  return window.__MCUS_TEST__;
}

test("assistant keeps M33 as a hard core constraint", async () => {
  const assistant = await loadAssistant();
  const result = assistant.aiRecommend("我要 M33 内核高频 MCU");
  assert.ok(result.results.length > 0);
  assert.ok(result.results.every(item => item.device.c.includes("Cortex-M33")));
});

test("assistant supports core alternatives and excludes M7", async () => {
  const assistant = await loadAssistant();
  const result = assistant.aiRecommend("高频但不要 M7，M33 或 M4 都可以");
  assert.deepEqual(Array.from(result.req.coreAny), ["cortex-m33", "cortex-m4"]);
  assert.deepEqual(Array.from(result.req.excludedCores), ["cortex-m7"]);
  assert.ok(result.results.every(item => !item.device.c.includes("Cortex-M7")));
});

test("assistant resolves Artery family prefixes", async () => {
  const assistant = await loadAssistant();
  const result = assistant.aiRecommend("ArteryTek AT32F4xx");
  assert.equal(result.req.vendor, "Artery");
  assert.ok(result.results.length > 0);
  assert.ok(result.results.every(item => item.device.m === "Artery"));
});

test("assistant handles upper bounds instead of reversing them", async () => {
  const assistant = await loadAssistant();
  const result = assistant.aiRecommend("主频 120MHz 以下，RAM 不超过 64KB");
  assert.equal(result.req.clockMax, 120000000);
  assert.equal(result.req.ramMax, 65536);
  assert.ok(result.results.every(item => item.device.hz <= 120000000 && item.device.ra <= 65536));
});

test("assistant combines domestic preference with serial and CAN requirements", async () => {
  const assistant = await loadAssistant();
  const result = assistant.aiRecommend("国产、至少 3 路 UART、CAN、不要 Wi-Fi");
  assert.equal(result.req.minimums.serial, 3);
  assert.equal(result.req.minimums.can, 1);
  assert.ok(result.results.every(item => (item.device.uart || 0) >= 3 && (item.device.can || 0) >= 1));
});

test("assistant parses technical timer and ADC constraints", async () => {
  const assistant = await loadAssistant();
  const parsed = assistant.aiParse("我要 M33 内核，至少一个 32 位定时器，16 位 ADC，采样率至少 1MSPS，不要 Wi-Fi");
  assert.equal(parsed.core, "cortex-m33");
  assert.equal(parsed.timerWidthMin, 32);
  assert.equal(parsed.minimums.tim, 1);
  assert.equal(parsed.adcResolution, 16);
  assert.equal(parsed.adcSampleRate, 1_000_000);
  assert.ok(parsed.excludedFeatures.includes("wifi"));
  assert.ok(parsed.excludedFeatures.includes("bluetooth"));
  const result = assistant.aiRecommend("M33，至少一个 32 位定时器，16 位 ADC，采样率至少 1MSPS");
  assert.equal(result.results[0].strict, true);
});

test("assistant parses flash and split RAM requirements without guessing", async () => {
  const assistant = await loadAssistant();
  const parsed = assistant.aiParse("M33，高频，双 Bank Flash，零等待，RAM 要有 ITCM 或 DTCM，最好有独占 RAM");
  assert.equal(parsed.core, "cortex-m33");
  assert.equal(parsed.flashBanks, 2);
  assert.equal(parsed.flashWaitStates, 0);
  assert.ok(parsed.flashArchitecture.includes("dualBank"));
  assert.deepEqual(Array.from(parsed.ramTypeAny), ["itcm", "dtcm"]);
  assert.equal(parsed.ramExclusive, true);
  assert.ok(parsed.technicalRequirements.includes("双 Bank Flash"));
});

test("assistant keeps local maximums when another feature has a maximum", async () => {
  const assistant = await loadAssistant();
  const parsed = assistant.aiParse("至少 3 路 UART，主频不超过 120MHz");
  assert.equal(parsed.minimums.serial, 3);
  assert.equal(parsed.clockMax, 120_000_000);
});

test("assistant understands colloquial speed preferences", async () => {
  const assistant = await loadAssistant();
  const parsed = assistant.aiParse("做电池供电的传感器，ADC 快一点，IO 翻转速度高一些，串口多一点");
  assert.ok(parsed.profiles.includes("low_power"));
  assert.ok(parsed.technicalPreferences.includes("fastAdc"));
  assert.ok(parsed.technicalPreferences.includes("fastIo"));
  assert.ok(parsed.preferences.includes("morePeripherals"));
});

test("assistant handles ranges and negated FPU intent", async () => {
  const assistant = await loadAssistant();
  const parsed = assistant.aiParse("2 到 3 路串口，主频 80MHz 以上，不要 FPU");
  assert.equal(parsed.minimums.serial, 2);
  assert.equal(parsed.maximums.serial, 3);
  assert.equal(parsed.clock, 80_000_000);
  assert.equal(parsed.fpu, false);
  assert.equal(parsed.fpuExcluded, true);
});

test("assistant keeps ADC speed separate from core clock", async () => {
  const assistant = await loadAssistant();
  const parsed = assistant.aiParse("16 位 ADC，ADC 速度 1M，主频 120MHz");
  assert.equal(parsed.adcResolution, 16);
  assert.equal(parsed.adcSampleRate, 1_000_000);
  assert.equal(parsed.clock, 120_000_000);
});

test("assistant separates ADC resolution from ADC unit count", async () => {
  const assistant = await loadAssistant();
  for (const prompt of ["16bitadc", "16-bit ADC", "16位ADC", "ADC 16位", "ADC16bit"]) {
    const parsed = assistant.aiParse(prompt);
    assert.equal(parsed.adcResolution, 16, prompt);
    assert.equal(parsed.minimums.adch, 1, prompt);
    assert.notEqual(parsed.minimums.adch, 16, prompt);
  }
  const countRequest = assistant.aiParse("至少 16 个 ADC");
  assert.equal(countRequest.adcResolution, null);
  assert.equal(countRequest.minimums.adch, 16);
});

test("assistant keeps ADC converter units separate from channels", async () => {
  const assistant = await loadAssistant();
  const parsed = assistant.aiParse("至少 2 个 ADC 转换器，至少 8 个 ADC 通道");
  assert.equal(parsed.minimums.adcu, 2);
  assert.equal(parsed.minimums.adch, 8);
});

test("assistant separates timer bit width from timer unit count", async () => {
  const assistant = await loadAssistant();
  for (const prompt of ["32bit timer", "32-bit timer", "32位定时器", "定时器 32位"]) {
    const parsed = assistant.aiParse(prompt);
    assert.equal(parsed.timerWidthMin, 32, prompt);
    assert.equal(parsed.minimums.tim, 1, prompt);
  }
  const countRequest = assistant.aiParse("至少 32 个定时器");
  assert.equal(countRequest.timerWidthMin, null);
  assert.equal(countRequest.minimums.tim, 32);
});

test("assistant parses typical power wording and current limits", async () => {
  const assistant = await loadAssistant();
  const typical = assistant.aiParse("典型功耗");
  assert.equal(typical.powerMentioned, true);
  assert.equal(typical.powerTypicalOnly, true);
  assert.ok(typical.preferences.includes("lowPower"));

  const run = assistant.aiParse("典型运行电流低于 100 uA");
  assert.equal(run.powerRunMax.value, 100);
  assert.equal(run.powerRunMax.rawValue, 100);
  assert.equal(run.powerRunMax.basis, "current");
  assert.equal(run.powerRunMax.unit, "ua");
  assert.equal(run.powerTypicalOnly, true);

  const sleep = assistant.aiParse("待机功耗不超过 1 mA");
  assert.equal(sleep.powerSleepMax.value, 1000);
  assert.equal(sleep.powerSleepMax.rawValue, 1);
  assert.equal(sleep.powerSleepMax.basis, "current");
  assert.equal(sleep.powerSleepMax.unit, "ma");
  assert.equal(sleep.powerTypicalOnly, false);
});

test("assistant recommends only devices with verified typical power for a typical-current request", async () => {
  const assistant = await loadAssistant();
  const result = assistant.aiRecommend("典型运行电流低于 100 uA");
  assert.ok(result.results.length > 0);
  assert.ok(result.results.every(item => item.device.pwr?.some(measurement => measurement.m === "run" && measurement.q === "typical")));
  assert.ok(result.results[0].strict);
});

test("multiple official source URLs are split into valid links", async () => {
  const assistant = await loadAssistant();
  assert.deepEqual(
    Array.from(assistant.safeHttpUrls("https://example.com/a;https://example.com/b\nnot-a-url;https://example.com/a")),
    ["https://example.com/a", "https://example.com/b"],
  );
});

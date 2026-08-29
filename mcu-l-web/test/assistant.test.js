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
    { id: "m33", m: "VendorA", f: "A", s: "A", l: "A", n: "M33-160", a: "Cortex-M33", c: "Cortex-M33", pt: "general_purpose_mcu", hz: 160000000, fl: 524288, ra: 262144, pin: "64", idx: 80, cov: 100, uart: 4, can: 2, pi: [] },
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

test("multiple official source URLs are split into valid links", async () => {
  const assistant = await loadAssistant();
  assert.deepEqual(
    Array.from(assistant.safeHttpUrls("https://example.com/a;https://example.com/b\nnot-a-url;https://example.com/a")),
    ["https://example.com/a", "https://example.com/b"],
  );
});

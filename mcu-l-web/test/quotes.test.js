import test from "node:test";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";

import worker, {
  ickeySign,
  md5Hex,
  normalizeIckeyOffers,
  resetIckeyTokenCache,
} from "../src/index.js";

const ENV = {
  MCUS_QUOTES_ENABLED: "true",
  ICKEY_API_BASE: "https://api.example.test",
  ICKEY_APP_ID: "mcus-app",
  ICKEY_APP_KEY: "secret-key",
};

test("MD5 and Ickey signing follow the documented canonical format", () => {
  assert.equal(md5Hex("abc"), "900150983cd24fb0d6963f7d28e17f72");
  const params = { appid: "app_id111", _t: 1721869200 };
  const canonical = "_t=1721869200&appid=app_id111|appkey1";
  const expected = createHash("md5").update(canonical).digest("hex");
  assert.equal(ickeySign(params, "appkey1"), expected);
});

test("Ickey offers keep exact MPNs and select the requested price tier", () => {
  const offers = normalizeIckeyOffers([
    {
      supplier: "云汉优选",
      sku: "1001",
      pro_name: "STM32F103C8T6",
      pro_maf: "STMicroelectronics",
      stock: 2519,
      moq: 1,
      spq: 1,
      nums: [1, 200, 1500],
      rmb: [8.5, 7.2, 6.4],
      lead_time_cn: "3-5工作日",
      detail_url: "https://www.ickey.cn/detail/1001/STM32F103C8T6.html",
    },
    {
      supplier: "错误变体",
      sku: "1002",
      pro_name: "STM32F103C8T6TR",
      nums: [1],
      rmb: [1],
    },
  ], "STM32F103C8T6", 250);

  assert.equal(offers.length, 1);
  assert.equal(offers[0].price, 7.2);
  assert.equal(offers[0].stock, 2519);
  assert.equal(offers[0].leadTime, "3-5工作日");
  assert.equal(offers[0].priceTiers.length, 3);
});

test("quote route stays disabled unless the deployment flag is enabled", async () => {
  const response = await worker.fetch(
    new Request("https://mcus.example/api/quotes?part=STM32F103C8T6"),
    {},
    { waitUntil() {} },
  );
  assert.equal(response.status, 404);
  assert.equal((await response.json()).code, "feature_disabled");
});

test("configured quote route obtains a token and returns normalized Ickey data", async () => {
  resetIckeyTokenCache();
  const originalFetch = globalThis.fetch;
  const calls = [];
  globalThis.fetch = async (url, options) => {
    calls.push({ url: String(url), body: String(options.body) });
    if (String(url).endsWith("/v2/new-token/create")) {
      return Response.json({
        success: true,
        errorCode: 0,
        message: "success",
        result: { token: "test-token", expireTime: 7200 },
      });
    }
    if (String(url).endsWith("/search-v1/products/get-single-goods-new")) {
      return Response.json({
        success: true,
        message: "success",
        errorCode: 0,
        result: [{
          supplier: "云汉在库",
          sku: "1003024504405",
          pro_name: "STM32F103C8T6",
          pro_maf: "STMicroelectronics",
          date_code: "24+",
          package: "LQFP48",
          stock: 3000,
          moq: 5,
          spq: 1,
          rmb: [8.5, 7.9, 7.2],
          nums: [5, 20, 100],
          lead_time_cn: "3-5工作日",
          detail_url: "https://www.ickey.cn/detail/1003024504405/STM32F103C8T6.html",
        }],
      });
    }
    throw new Error(`Unexpected request: ${url}`);
  };

  try {
    const response = await worker.fetch(
      new Request("https://mcus.example/api/quotes?part=STM32F103C8T6&quantity=20"),
      ENV,
      { waitUntil() {} },
    );
    assert.equal(response.status, 200);
    assert.equal(response.headers.get("cache-control"), "no-store");
    const payload = await response.json();
    assert.equal(payload.provider, "ickey");
    assert.equal(payload.quantity, 20);
    assert.equal(payload.quotes[0].price, 7.9);
    assert.equal(payload.quotes[0].moq, 5);
    assert.equal(calls.length, 2);
    assert.match(calls[0].body, /appid=mcus-app/);
    assert.match(calls[0].body, /sign=[0-9a-f]{32}/);
    assert.match(calls[1].body, /keyword=STM32F103C8T6/);
    assert.match(calls[1].body, /is_exact_match=1/);
    assert.match(calls[1].body, /pro_num=20/);
  } finally {
    globalThis.fetch = originalFetch;
    resetIckeyTokenCache();
  }
});

test("enabled quote route reports missing Ickey credentials without leaking details", async () => {
  const response = await worker.fetch(
    new Request("https://mcus.example/api/quotes?part=STM32F103C8T6"),
    { MCUS_QUOTES_ENABLED: "true" },
    { waitUntil() {} },
  );
  assert.equal(response.status, 503);
  assert.equal((await response.json()).code, "not_configured");
});

test("quote route handles CORS preflight without a response body", async () => {
  const response = await worker.fetch(
    new Request("https://mcus.example/api/quotes", { method: "OPTIONS" }),
    {},
    { waitUntil() {} },
  );
  assert.equal(response.status, 204);
  assert.equal(await response.text(), "");
});

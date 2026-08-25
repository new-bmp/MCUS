import test from "node:test";
import assert from "node:assert/strict";

import worker, { exactPartMatch, isChipListing, md5Hex, pickQuotes } from "../src/index.js";

test("MD5 signer uses the standard digest", () => {
  assert.equal(md5Hex("abc"), "900150983CD24FB0D6963F7D28E17F72");
  assert.equal(md5Hex("淘宝"), "12AD5C790444F88966C2FAF90E73D8C9");
});

test("exact part matching rejects longer look-alike models", () => {
  assert.equal(exactPartMatch("原装 STM32F429ZIT6 LQFP144 芯片", "STM32F429ZIT6"), true);
  assert.equal(exactPartMatch("STM32F429ZIT6TR 原装", "STM32F429ZIT6"), false);
  assert.equal(exactPartMatch("2PCS STM32F429ZIT6", "STM32F429ZIT6"), true);
});

test("chip filter excludes boards, modules, programmers and used pulls", () => {
  const part = "STM32F429ZIT6";
  assert.equal(isChipListing(`${part} 原装芯片`, part), true);
  assert.equal(isChipListing(`${part} 核心板`, part), false);
  assert.equal(isChipListing(`${part} 开发套件`, part), false);
  assert.equal(isChipListing(`${part} 拆机`, part), false);
});

test("quote selection keeps three distinct stores with valid prices", () => {
  const part = "STM32F429ZIT6";
  const items = [
    { title: `${part} 原装芯片`, shop_title: "店铺甲", seller_id: "1", zk_final_price: "28.50", item_id: "a", item_url: "https://item.taobao.com/item.htm?id=a" },
    { title: `${part} 核心板`, shop_title: "开发板店", seller_id: "2", zk_final_price: "12.00", item_id: "b" },
    { title: `${part} 芯片`, shop_title: "店铺甲", seller_id: "1", zk_final_price: "27.00", item_id: "c" },
    { title: `${part} 现货`, shop_title: "店铺乙", seller_id: "3", zk_final_price: "31.00", item_id: "d" },
    { title: `${part} 原装`, shop_title: "店铺丙", seller_id: "4", zk_final_price: "29.00", item_id: "e" },
    { title: `${part} 芯片`, shop_title: "店铺丁", seller_id: "5", zk_final_price: "35.00", item_id: "f" },
  ];
  const quotes = pickQuotes(items, part);
  assert.deepEqual(quotes.map((quote) => quote.shop), ["店铺甲", "店铺丙", "店铺乙"]);
  assert.equal(quotes.length, 3);
});

test("quote route stays disabled in 1.0.2", async () => {
  const context = { waitUntil() {} };
  const response = await worker.fetch(
    new Request("https://mcus.example/api/quotes?part=STM32F429ZIT6"),
    {},
    context,
  );
  assert.equal(response.status, 404);
  assert.equal((await response.json()).code, "feature_disabled");
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

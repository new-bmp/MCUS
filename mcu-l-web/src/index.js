const TAOBAO_API = "https://eco.taobao.com/router/rest";
const CACHE_SECONDS = 600;
const QUOTES_ENABLED = false;
const BLOCKED_LISTING_TERMS = [
  "开发板", "核心板", "最小系统", "学习板", "评估板", "扩展板", "底板",
  "开发套件", "套件", "模块", "模组", "成品板", "烧录器", "下载器", "仿真器",
  "二手", "拆机", "demo board", "development board", "evaluation board",
  "eval board", "starter kit", "breakout board", "dev board",
];

function json(data, status = 200, extraHeaders = {}) {
  return new Response(JSON.stringify(data), {
    status,
    headers: {
      "content-type": "application/json; charset=utf-8",
      "access-control-allow-origin": "*",
      "access-control-allow-methods": "GET, OPTIONS",
      "access-control-allow-headers": "Accept",
      "x-content-type-options": "nosniff",
      ...extraHeaders,
    },
  });
}

function rotateLeft(value, count) {
  return ((value << count) | (value >>> (32 - count))) >>> 0;
}

export function md5Hex(text) {
  const bytes = new TextEncoder().encode(text);
  const bitLength = BigInt(bytes.length) * 8n;
  const paddedLength = Math.ceil((bytes.length + 9) / 64) * 64;
  const padded = new Uint8Array(paddedLength);
  padded.set(bytes);
  padded[bytes.length] = 0x80;
  for (let i = 0; i < 8; i += 1) {
    padded[paddedLength - 8 + i] = Number((bitLength >> BigInt(i * 8)) & 0xffn);
  }

  const shifts = [
    7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
    5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20,
    4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
    6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
  ];
  const constants = Array.from({ length: 64 }, (_, i) =>
    Math.floor(Math.abs(Math.sin(i + 1)) * 0x100000000) >>> 0
  );
  let a0 = 0x67452301;
  let b0 = 0xefcdab89;
  let c0 = 0x98badcfe;
  let d0 = 0x10325476;

  for (let offset = 0; offset < padded.length; offset += 64) {
    const words = new Uint32Array(16);
    for (let i = 0; i < 16; i += 1) {
      const at = offset + i * 4;
      words[i] = (padded[at] | (padded[at + 1] << 8) |
        (padded[at + 2] << 16) | (padded[at + 3] << 24)) >>> 0;
    }
    let a = a0;
    let b = b0;
    let c = c0;
    let d = d0;
    for (let i = 0; i < 64; i += 1) {
      let f;
      let g;
      if (i < 16) {
        f = (b & c) | (~b & d);
        g = i;
      } else if (i < 32) {
        f = (d & b) | (~d & c);
        g = (5 * i + 1) % 16;
      } else if (i < 48) {
        f = b ^ c ^ d;
        g = (3 * i + 5) % 16;
      } else {
        f = c ^ (b | ~d);
        g = (7 * i) % 16;
      }
      const nextD = d;
      d = c;
      c = b;
      const sum = (a + f + constants[i] + words[g]) >>> 0;
      b = (b + rotateLeft(sum, shifts[i])) >>> 0;
      a = nextD;
    }
    a0 = (a0 + a) >>> 0;
    b0 = (b0 + b) >>> 0;
    c0 = (c0 + c) >>> 0;
    d0 = (d0 + d) >>> 0;
  }

  return [a0, b0, c0, d0].map((word) =>
    [0, 8, 16, 24].map((shift) => ((word >>> shift) & 0xff).toString(16).padStart(2, "0")).join("")
  ).join("").toUpperCase();
}

function chinaTimestamp(date = new Date()) {
  return new Date(date.getTime() + 8 * 60 * 60 * 1000)
    .toISOString().slice(0, 19).replace("T", " ");
}

function signTopRequest(params, secret) {
  const canonical = Object.keys(params).sort().map((key) => `${key}${params[key]}`).join("");
  return md5Hex(`${secret}${canonical}${secret}`);
}

function validPart(part) {
  return /^[A-Za-z0-9][A-Za-z0-9+._\/-]{3,63}$/.test(part);
}

export function exactPartMatch(title, part) {
  const haystack = String(title || "").toUpperCase();
  const needle = String(part || "").toUpperCase();
  if (!needle) return false;
  let index = haystack.indexOf(needle);
  while (index >= 0) {
    const before = index > 0 ? haystack[index - 1] : "";
    const after = haystack[index + needle.length] || "";
    if ((!before || !/[A-Z0-9]/.test(before)) && (!after || !/[A-Z0-9]/.test(after))) return true;
    index = haystack.indexOf(needle, index + 1);
  }
  return false;
}

export function isChipListing(title, part) {
  const normalized = String(title || "").toLowerCase();
  return exactPartMatch(title, part) && !BLOCKED_LISTING_TERMS.some((term) => normalized.includes(term));
}

function firstValue(item, paths) {
  for (const path of paths) {
    let value = item;
    for (const key of path.split(".")) value = value && value[key];
    if (value !== undefined && value !== null && value !== "") return value;
  }
  return "";
}

function safeProductUrl(value) {
  try {
    let raw = String(value || "").trim().replace(/^\/\//, "https://");
    if (/^[\w.-]+\//.test(raw)) raw = `https://${raw}`;
    const url = new URL(raw);
    return ["http:", "https:"].includes(url.protocol) ? url.href : "";
  } catch {
    return "";
  }
}

function resultItems(payload) {
  const response = payload.taobao_tbk_dg_material_optional_response || {};
  const candidates = response.result_list?.map_data ??
    response.result_list?.mapData ?? response.result_list ?? [];
  if (Array.isArray(candidates)) return candidates;
  return candidates && typeof candidates === "object" ? [candidates] : [];
}

export function pickQuotes(items, part, limit = 3) {
  const quotes = [];
  const shops = new Set();
  for (const item of items) {
    const title = String(firstValue(item, ["title", "item_basic_info.title", "item_basic_info.short_title"]));
    if (!isChipListing(title, part)) continue;
    const price = Number(firstValue(item, [
      "zk_final_price", "price_promotion_info.zk_final_price", "price_promotion_info.final_promotion_price",
      "item_basic_info.zk_final_price",
      "reserve_price", "item_basic_info.reserve_price",
    ]));
    if (!Number.isFinite(price) || price <= 0) continue;
    const shop = String(firstValue(item, ["shop_title", "shop_info.shop_title", "seller_nick"])).trim();
    const sellerId = String(firstValue(item, ["seller_id", "shop_info.seller_id", "user_id"])).trim();
    const shopKey = sellerId || shop.toLowerCase();
    if (!shopKey || shops.has(shopKey)) continue;
    const url = safeProductUrl(firstValue(item, [
      "click_url", "publish_info.click_url", "coupon_share_url", "item_url", "url",
    ]));
    shops.add(shopKey);
    quotes.push({
      shop: shop || `淘宝商家 ${quotes.length + 1}`,
      sellerId,
      title,
      price,
      itemId: String(firstValue(item, ["item_id", "item_basic_info.item_id"])),
      url,
    });
  }
  return quotes.sort((left, right) => left.price - right.price).slice(0, limit);
}

async function fetchTaobaoQuotes(part, env) {
  const params = {
    method: "taobao.tbk.dg.material.optional",
    app_key: env.TAOBAO_APP_KEY,
    timestamp: chinaTimestamp(),
    format: "json",
    v: "2.0",
    sign_method: "md5",
    adzone_id: env.TAOBAO_ADZONE_ID,
    q: part,
    platform: "2",
    page_no: "1",
    page_size: "100",
    material_id: "2836",
    biz_scene_id: "2",
    sort: "price_asc",
    npx_level: "2",
  };
  params.sign = signTopRequest(params, env.TAOBAO_APP_SECRET);
  const response = await fetch(TAOBAO_API, {
    method: "POST",
    headers: { "content-type": "application/x-www-form-urlencoded;charset=UTF-8" },
    body: new URLSearchParams(params),
  });
  if (!response.ok) throw new Error(`Taobao HTTP ${response.status}`);
  const payload = await response.json();
  if (payload.error_response) {
    const error = new Error(payload.error_response.sub_msg || payload.error_response.msg || "Taobao API error");
    error.taobaoCode = payload.error_response.sub_code || payload.error_response.code;
    throw error;
  }
  return pickQuotes(resultItems(payload), part);
}

async function quoteResponse(context, env, url) {
  const part = (url.searchParams.get("part") || "").trim().toUpperCase();
  if (!validPart(part)) return json({ code: "invalid_part", message: "订货号格式无效。" }, 400);
  if (!env.TAOBAO_APP_KEY || !env.TAOBAO_APP_SECRET || !env.TAOBAO_ADZONE_ID) {
    return json({ code: "not_configured", message: "淘宝开放平台参数尚未配置。" }, 503);
  }

  const cache = caches.default;
  const cacheKey = new Request(`${url.origin}/api/quotes?part=${encodeURIComponent(part)}`);
  const cached = await cache.match(cacheKey);
  if (cached) return cached;
  try {
    const quotes = await fetchTaobaoQuotes(part, env);
    const response = json({
      part,
      quotes,
      count: quotes.length,
      strict: true,
      updatedAt: new Date().toISOString(),
    }, 200, { "cache-control": `public, max-age=${CACHE_SECONDS}` });
    context.waitUntil(cache.put(cacheKey, response.clone()));
    return response;
  } catch (error) {
    console.error("Taobao quote error", error.taobaoCode || "unknown", error.message);
    return json({ code: "taobao_api_error", message: "淘宝接口暂时不可用，请稍后重试。" }, 502);
  }
}

export default {
  async fetch(request, env, context) {
    const url = new URL(request.url);
    if (request.method === "OPTIONS" && url.pathname === "/api/quotes") {
      return new Response(null, { status: 204, headers: {
        "access-control-allow-origin": "*",
        "access-control-allow-methods": "GET, OPTIONS",
        "access-control-allow-headers": "Accept",
      } });
    }
    if (url.pathname === "/health") {
      return json({ service: "MCUS", status: "ok", quotes: false, quotesStatus: "disabled" });
    }
    if (url.pathname === "/api/quotes") {
      if (!QUOTES_ENABLED) return json({ code: "feature_disabled", message: "1.0.2 暂不开放询价。" }, 404);
      if (request.method !== "GET") return json({ code: "method_not_allowed" }, 405);
      return quoteResponse(context, env, url);
    }
    return env.ASSETS.fetch(request);
  },
};

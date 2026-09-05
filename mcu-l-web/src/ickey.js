const TOKEN_SAFETY_SECONDS = 60;

let tokenCache = null;

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
  ).join("");
}

export function ickeySign(params, appKey) {
  const canonical = Object.keys(params)
    .filter((key) => key !== "sign" && params[key] !== undefined && params[key] !== null)
    .sort()
    .map((key) => `${key}=${params[key]}`)
    .join("&");
  return md5Hex(`${canonical}|${appKey}`);
}

export class IckeyError extends Error {
  constructor(code, message, status = 502) {
    super(message);
    this.name = "IckeyError";
    this.code = code;
    this.status = status;
  }
}

export function isIckeyConfigured(env) {
  return Boolean(env.ICKEY_API_BASE && env.ICKEY_APP_ID && env.ICKEY_APP_KEY);
}

function apiBase(env) {
  let url;
  try {
    url = new URL(env.ICKEY_API_BASE);
  } catch {
    throw new IckeyError("ickey_invalid_config", "云汉接口域名无效。", 503);
  }
  if (url.protocol !== "https:" || url.username || url.password) {
    throw new IckeyError("ickey_invalid_config", "云汉接口必须使用不含凭据的 HTTPS 域名。", 503);
  }
  return url.href.replace(/\/$/, "");
}

function unixSeconds(now = Date.now()) {
  return Math.floor(now / 1000);
}

async function postForm(env, path, params, fetchImpl) {
  const response = await fetchImpl(`${apiBase(env)}${path}`, {
    method: "POST",
    headers: {
      accept: "application/json",
      "content-type": "application/x-www-form-urlencoded;charset=UTF-8",
    },
    body: new URLSearchParams(Object.entries(params).map(([key, value]) => [key, String(value)])),
  });
  if (!response.ok) {
    throw new IckeyError("ickey_api_error", `云汉接口返回 HTTP ${response.status}。`);
  }
  try {
    return await response.json();
  } catch {
    throw new IckeyError("ickey_api_error", "云汉接口返回了无法识别的数据。");
  }
}

function providerMessage(payload, fallback) {
  return String(payload?.message || payload?.resultMessage || fallback);
}

function authFailure(payload) {
  const message = providerMessage(payload, "").toLowerCase();
  return payload?.errorCode === 401 || payload?.errorCode === 403 ||
    /token|登录|鉴权|认证|过期/.test(message);
}

async function getToken(env, fetchImpl, force = false) {
  const now = unixSeconds();
  const cacheKey = `${apiBase(env)}|${env.ICKEY_APP_ID}|${md5Hex(env.ICKEY_APP_KEY)}`;
  if (!force && tokenCache?.key === cacheKey && tokenCache.expiresAt > now + TOKEN_SAFETY_SECONDS) {
    return tokenCache.value;
  }

  const params = { _t: now, appid: env.ICKEY_APP_ID };
  params.sign = ickeySign(params, env.ICKEY_APP_KEY);
  const payload = await postForm(env, "/v2/new-token/create", params, fetchImpl);
  const token = String(payload?.result?.token || "").trim();
  if (!payload?.success || !token) {
    throw new IckeyError("ickey_auth_error", providerMessage(payload, "云汉接口鉴权失败。"), 502);
  }
  const expiresIn = Math.max(120, Number(payload.result.expireTime) || 7200);
  tokenCache = { key: cacheKey, value: token, expiresAt: now + expiresIn };
  return token;
}

function safeUrl(value) {
  try {
    const url = new URL(String(value || ""));
    return url.protocol === "https:" ? url.href : "";
  } catch {
    return "";
  }
}

function positiveNumber(value, fallback = 0) {
  const number = Number(value);
  return Number.isFinite(number) && number > 0 ? number : fallback;
}

function priceTiers(item) {
  const quantities = Array.isArray(item.nums) ? item.nums : [];
  const prices = Array.isArray(item.rmb) ? item.rmb : [];
  return quantities.map((quantity, index) => ({
    quantity: positiveNumber(quantity),
    price: positiveNumber(prices[index]),
  })).filter((tier) => tier.quantity && tier.price)
    .sort((left, right) => left.quantity - right.quantity);
}

function selectedPrice(tiers, item, quantity) {
  const moq = positiveNumber(item.moq, 1);
  const effectiveQuantity = Math.max(quantity, moq, tiers[0]?.quantity || 1);
  let selected = tiers[0];
  for (const tier of tiers) {
    if (tier.quantity <= effectiveQuantity) selected = tier;
  }
  return selected?.price || positiveNumber(item.unit_price);
}

export function normalizeIckeyOffers(items, part, quantity = 1, limit = 3) {
  const target = String(part || "").trim().toUpperCase();
  const offers = [];
  const seen = new Set();
  for (const item of Array.isArray(items) ? items : []) {
    const mpn = String(item.pro_name || item.pro_sno || item.mpn || "").trim().toUpperCase();
    if (mpn !== target) continue;
    const sku = String(item.sku || `${mpn}|${item.supplier || ""}|${item.pro_maf || ""}`);
    if (seen.has(sku)) continue;
    const tiers = priceTiers(item);
    const price = selectedPrice(tiers, item, quantity);
    if (!price) continue;
    seen.add(sku);
    const manufacturer = String(item.pro_maf || item.mfr_name || item.manufacturer || "").trim();
    offers.push({
      shop: String(item.supplier || "云汉芯城").trim(),
      sku,
      title: manufacturer ? `${target} · ${manufacturer}` : target,
      manufacturer,
      price,
      currency: "CNY",
      stock: Math.max(0, Number(item.stock) || 0),
      moq: positiveNumber(item.moq, 1),
      spq: positiveNumber(item.spq),
      mpq: positiveNumber(item.mpq),
      package: String(item.package || item.footprint || "").trim(),
      dateCode: String(item.date_code || item.dc || "").trim(),
      leadTime: String(item.lead_time_cn || item.lead_time || "").trim(),
      priceTiers: tiers.slice(0, 6),
      url: safeUrl(item.detail_url),
    });
  }
  return offers.sort((left, right) => left.price - right.price || right.stock - left.stock).slice(0, limit);
}

async function search(env, token, part, quantity, fetchImpl) {
  return postForm(env, "/search-v1/products/get-single-goods-new", {
    token,
    _t: unixSeconds(),
    keyword: part,
    is_exact_match: 1,
    pro_num: quantity,
    delivery: 1,
  }, fetchImpl);
}

export async function fetchIckeyQuotes(part, quantity, env, fetchImpl = fetch) {
  let token = await getToken(env, fetchImpl);
  let payload = await search(env, token, part, quantity, fetchImpl);
  if (!payload?.success && authFailure(payload)) {
    token = await getToken(env, fetchImpl, true);
    payload = await search(env, token, part, quantity, fetchImpl);
  }
  if (!payload?.success) {
    throw new IckeyError("ickey_api_error", providerMessage(payload, "云汉型号搜索失败。"));
  }
  return normalizeIckeyOffers(payload.result, part, quantity);
}

export function resetIckeyTokenCache() {
  tokenCache = null;
}

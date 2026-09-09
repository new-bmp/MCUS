/**
 * LCSC quote adapter.
 *
 * LCSC access is intentionally kept behind an authorized API gateway. The
 * public storefront is not treated as an API: scraping it would be brittle,
 * could violate its terms, and cannot guarantee that a result is the exact
 * chip rather than a board or a related variant. The gateway contract is a
 * small GET endpoint configured with LCSC_API_BASE/LCSC_API_PATH and may
 * return either {quotes: []}, {data: []}, {results: []}, or a bare array.
 */

export class LcscError extends Error {
  constructor(code, message, status = 502) {
    super(message);
    this.name = "LcscError";
    this.code = code;
    this.status = status;
  }
}
export function isLcscConfigured(env) {
  return Boolean(env.LCSC_API_BASE && env.LCSC_API_KEY);
}

function apiUrl(env, part, quantity) {
  let base;
  try {
    base = new URL(env.LCSC_API_BASE);
  } catch {
    throw new LcscError("lcsc_invalid_config", "立创接口地址无效。", 503);
  }
  if (base.protocol !== "https:" || base.username || base.password) {
    throw new LcscError("lcsc_invalid_config", "立创接口必须使用不含凭据的 HTTPS 地址。", 503);
  }
  const path = String(env.LCSC_API_PATH || "/v1/quotes").trim();
  const endpoint = new URL(path.replace(/^\/+/, ""), `${base.href.replace(/\/$/, "")}/`);
  endpoint.searchParams.set("part", part);
  endpoint.searchParams.set("quantity", String(quantity));
  endpoint.searchParams.set("exact", "1");
  return endpoint;
}

function positiveNumber(value, fallback = 0) {
  const number = Number(value);
  return Number.isFinite(number) && number > 0 ? number : fallback;
}

function safeUrl(value) {
  try {
    const url = new URL(String(value || ""));
    return url.protocol === "https:" ? url.href : "";
  } catch {
    return "";
  }
}

function text(value) {
  return String(value ?? "").trim();
}

function tiersFor(item) {
  const tiers = item.priceTiers || item.price_tiers || item.tiers;
  if (Array.isArray(tiers)) {
    return tiers.map((tier) => ({
      quantity: positiveNumber(tier.quantity ?? tier.qty ?? tier.minQuantity),
      price: positiveNumber(tier.price ?? tier.unitPrice ?? tier.unit_price),
    })).filter((tier) => tier.quantity && tier.price)
      .sort((left, right) => left.quantity - right.quantity);
  }
  const quantities = Array.isArray(item.quantities) ? item.quantities :
    Array.isArray(item.nums) ? item.nums : [];
  const prices = Array.isArray(item.prices) ? item.prices :
    Array.isArray(item.rmb) ? item.rmb : [];
  return quantities.map((quantity, index) => ({
    quantity: positiveNumber(quantity),
    price: positiveNumber(prices[index]),
  })).filter((tier) => tier.quantity && tier.price)
    .sort((left, right) => left.quantity - right.quantity);
}

function selectedPrice(tiers, item, quantity) {
  const moq = positiveNumber(item.moq ?? item.minimumOrderQuantity, 1);
  const effectiveQuantity = Math.max(quantity, moq, tiers[0]?.quantity || 1);
  let selected = tiers[0];
  for (const tier of tiers) {
    if (tier.quantity <= effectiveQuantity) selected = tier;
  }
  return selected?.price || positiveNumber(item.price ?? item.unitPrice ?? item.unit_price);
}

function responseItems(payload) {
  if (Array.isArray(payload)) return payload;
  for (const key of ["quotes", "data", "results", "items", "products"]) {
    if (Array.isArray(payload?.[key])) return payload[key];
  }
  return [];
}

/** Normalize only exact MPN matches; related variants and boards are dropped. */
export function normalizeLcscOffers(items, part, quantity = 1, limit = 3) {
  const target = text(part).toUpperCase();
  const offers = [];
  const seen = new Set();
  for (const item of Array.isArray(items) ? items : []) {
    if (!item || typeof item !== "object") continue;
    const mpn = text(item.mpn ?? item.partNumber ?? item.part_number ?? item.model ?? item.productCode).toUpperCase();
    if (mpn !== target) continue;
    const tiers = tiersFor(item);
    const price = selectedPrice(tiers, item, quantity);
    if (!price) continue;
    const sku = text(item.sku ?? item.lcscPartNumber ?? item.lcsc_part_number ?? item.productId ?? `${target}|${item.seller || item.shop || "LCSC"}`);
    if (seen.has(sku)) continue;
    seen.add(sku);
    offers.push({
      shop: text(item.shop ?? item.seller ?? item.supplier ?? "立创商城") || "立创商城",
      sku,
      title: text(item.title ?? item.name ?? target) || target,
      manufacturer: text(item.manufacturer ?? item.mfr ?? ""),
      price,
      currency: text(item.currency ?? "CNY") || "CNY",
      stock: Math.max(0, Number(item.stock ?? item.inventory ?? item.quantityInStock) || 0),
      moq: positiveNumber(item.moq ?? item.minimumOrderQuantity, 1),
      spq: positiveNumber(item.spq ?? item.standardPack),
      mpq: positiveNumber(item.mpq),
      package: text(item.package ?? item.packageType ?? item.footprint),
      dateCode: text(item.dateCode ?? item.date_code),
      leadTime: text(item.leadTime ?? item.lead_time),
      priceTiers: tiers.slice(0, 6),
      url: safeUrl(item.url ?? item.detailUrl ?? item.detail_url),
    });
  }
  return offers.sort((left, right) => left.price - right.price || right.stock - left.stock).slice(0, limit);
}

export async function fetchLcscQuotes(part, quantity, env, fetchImpl = fetch) {
  if (!isLcscConfigured(env)) {
    throw new LcscError("not_configured", "立创商城开放接口尚未配置。", 503);
  }
  const endpoint = apiUrl(env, part, quantity);
  let response;
  try {
    response = await fetchImpl(endpoint.href, {
      headers: {
        accept: "application/json",
        authorization: `Bearer ${env.LCSC_API_KEY}`,
      },
    });
  } catch {
    throw new LcscError("lcsc_api_error", "立创商城接口暂时不可用，请稍后重试。", 502);
  }
  if (!response.ok) {
    if (response.status === 401 || response.status === 403) {
      throw new LcscError("lcsc_auth_error", "立创商城接口鉴权失败，请检查开放平台配置。", 502);
    }
    throw new LcscError("lcsc_api_error", `立创商城接口返回 HTTP ${response.status}。`, 502);
  }
  let payload;
  try {
    payload = await response.json();
  } catch {
    throw new LcscError("lcsc_api_error", "立创商城接口返回了无法识别的数据。", 502);
  }
  const quotes = normalizeLcscOffers(responseItems(payload), part, quantity);
  return quotes;
}

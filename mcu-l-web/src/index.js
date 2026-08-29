import {
  IckeyError,
  fetchIckeyQuotes,
  isIckeyConfigured,
} from "./ickey.js";

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

function validPart(part) {
  return /^[A-Za-z0-9][A-Za-z0-9+._\/-]{3,63}$/.test(part);
}

function requestedQuantity(url) {
  const raw = url.searchParams.get("quantity") || "1";
  if (!/^\d{1,7}$/.test(raw)) return 0;
  const quantity = Number(raw);
  return quantity >= 1 && quantity <= 1_000_000 ? quantity : 0;
}

function quotesEnabled(env) {
  return String(env.MCUS_QUOTES_ENABLED || "").toLowerCase() === "true";
}

async function quoteResponse(context, env, url) {
  const part = (url.searchParams.get("part") || "").trim().toUpperCase();
  if (!validPart(part)) return json({ code: "invalid_part", message: "订货号格式无效。" }, 400);
  const quantity = requestedQuantity(url);
  if (!quantity) return json({ code: "invalid_quantity", message: "询价数量必须为 1 至 1000000。" }, 400);
  if (!isIckeyConfigured(env)) {
    return json({ code: "not_configured", message: "云汉芯城开放平台参数尚未配置。" }, 503);
  }

  try {
    const quotes = await fetchIckeyQuotes(part, quantity, env);
    const response = json({
      provider: "ickey",
      providerName: "云汉芯城",
      part,
      quantity,
      quotes,
      count: quotes.length,
      strict: true,
      updatedAt: new Date().toISOString(),
    }, 200, { "cache-control": "no-store" });
    return response;
  } catch (error) {
    const known = error instanceof IckeyError;
    console.error("Ickey quote error", known ? error.code : "unknown", error.message);
    return json({
      code: known ? error.code : "ickey_api_error",
      message: known ? error.message : "云汉芯城接口暂时不可用，请稍后重试。",
    }, known ? error.status : 502);
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
      return json({
        service: "MCUS",
        status: "ok",
        quotes: quotesEnabled(env),
        quotesConfigured: isIckeyConfigured(env),
        quotesProvider: "ickey",
      });
    }
    if (url.pathname === "/api/quotes") {
      if (!quotesEnabled(env)) return json({ code: "feature_disabled", message: "实时询价暂未开放。" }, 404);
      if (request.method !== "GET") return json({ code: "method_not_allowed" }, 405);
      return quoteResponse(context, env, url);
    }
    return env.ASSETS.fetch(request);
  },
};

export {
  fetchIckeyQuotes,
  ickeySign,
  isIckeyConfigured,
  md5Hex,
  normalizeIckeyOffers,
  resetIckeyTokenCache,
} from "./ickey.js";

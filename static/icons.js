(() => {
  const icons = {
    "play": '<polygon points="8 5 19 12 8 19 8 5"/>',
    "pause": '<rect x="6" y="5" width="4" height="14" rx="1"/><rect x="14" y="5" width="4" height="14" rx="1"/>',
    "x": '<path d="M18 6 6 18M6 6l12 12"/>',
    "check": '<path d="m5 12 4 4L19 6"/>',
    "circle-check": '<circle cx="12" cy="12" r="9"/><path d="m8 12 3 3 5-6"/>',
    "shield-check": '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10Z"/><path d="m9 12 2 2 4-4"/>',
    "search": '<circle cx="11" cy="11" r="7"/><path d="m20 20-4-4"/>',
    "zoom-in": '<circle cx="10" cy="10" r="6"/><path d="m15 15 5 5M10 7v6M7 10h6"/>',
    "zoom-out": '<circle cx="10" cy="10" r="6"/><path d="m15 15 5 5M7 10h6"/>',
    "folder-open": '<path d="M3 7h6l2 2h10l-3 9H4L3 7Z"/><path d="M3 7V5h7l2 2"/>',
    "folder": '<path d="M3 6h7l2 2h9v11H3V6Z"/>',
    "folder-kanban": '<path d="M3 6h7l2 2h9v11H3V6Z"/><path d="M8 12v4M12 11v5M16 13v3"/>',
    "folder-input": '<path d="M3 6h7l2 2h9v11H3V6Z"/><path d="m12 11-3 3 3 3M9 14h7"/>',
    "folder-output": '<path d="M3 6h7l2 2h9v11H3V6Z"/><path d="m13 11 3 3-3 3M8 14h8"/>',
    "download": '<path d="M12 3v12m-4-4 4 4 4-4M4 20h16"/>',
    "upload-cloud": '<path d="M7 18H5a3 3 0 0 1 0-6 7 7 0 0 1 13-2 4 4 0 0 1 1 8h-2M12 12v9m-3-6 3-3 3 3"/>',
    "file-up": '<path d="M6 3h8l4 4v14H6V3Z"/><path d="M14 3v5h5M12 18v-6m-3 3 3-3 3 3"/>',
    "file-archive": '<path d="M6 3h9l3 3v15H6V3Z"/><path d="M9 3v3m0 2v3m0 2v3M8 18h3"/>',
    "file-minus": '<path d="M6 3h8l4 4v14H6V3Z"/><path d="M14 3v5h5M9 14h6"/>',
    "database": '<ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v7c0 1.7 3.6 3 8 3s8-1.3 8-3V5M4 12v7c0 1.7 3.6 3 8 3s8-1.3 8-3v-7"/>',
    "box": '<path d="m4 7 8-4 8 4-8 4-8-4Z"/><path d="M4 7v10l8 4 8-4V7M12 11v10"/>',
    "hard-drive": '<rect x="3" y="6" width="18" height="12" rx="2"/><path d="M7 14h.01M11 14h6"/>',
    "step-back": '<path d="M18 5 9 12l9 7V5ZM6 5v14"/>',
    "step-forward": '<path d="m6 5 9 7-9 7V5Zm12 0v14"/>',
    "flag": '<path d="M5 21V4m0 1h12l-2 4 2 4H5"/>',
    "hand": '<path d="M7 11V7a2 2 0 0 1 4 0v3-5a2 2 0 0 1 4 0v5-3a2 2 0 0 1 4 0v7c0 5-3 8-7 8-3 0-5-2-7-5l-2-3a2 2 0 0 1 3-2l1 1"/>',
    "sparkles": '<path d="m12 3 1.3 3.7L17 8l-3.7 1.3L12 13l-1.3-3.7L7 8l3.7-1.3L12 3ZM5 15l.8 2.2L8 18l-2.2.8L5 21l-.8-2.2L2 18l2.2-.8L5 15Zm14-3 .8 2.2L22 15l-2.2.8L19 18l-.8-2.2L16 15l2.2-.8L19 12Z"/>',
    "scan": '<path d="M4 8V4h4m8 0h4v4m0 8v4h-4M8 20H4v-4"/><circle cx="12" cy="12" r="3"/>',
    "scan-line": '<path d="M4 8V4h4m8 0h4v4m0 8v4h-4M8 20H4v-4M3 12h18"/>',
    "scan-search": '<path d="M4 8V4h4m8 0h4v4M8 20H4v-4"/><circle cx="13" cy="13" r="4"/><path d="m16 16 4 4"/>',
    "plug-zap": '<path d="m13 2-3 7h4l-3 7 7-9h-4l2-5M6 13l-3 3 5 5 3-3M5 15l-2-2m7 7 2 2"/>',
    "waypoints": '<circle cx="5" cy="5" r="2"/><circle cx="19" cy="7" r="2"/><circle cx="8" cy="19" r="2"/><path d="m7 6 10 1M17 9 10 17M7 17l-1-10"/>'
  };

  const fallback = '<circle cx="12" cy="12" r="8"/><path d="M8 12h8M12 8v8"/>';

  function createIcons(options = {}) {
    const attrs = options.attrs || {};
    document.querySelectorAll("i[data-lucide]").forEach((element) => {
      const name = element.getAttribute("data-lucide");
      const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
      svg.setAttribute("viewBox", "0 0 24 24");
      svg.setAttribute("fill", "none");
      svg.setAttribute("stroke", "currentColor");
      svg.setAttribute("stroke-width", attrs["stroke-width"] || "1.8");
      svg.setAttribute("stroke-linecap", "round");
      svg.setAttribute("stroke-linejoin", "round");
      svg.setAttribute("aria-hidden", "true");
      svg.setAttribute("data-lucide", name || "icon");
      if (element.className) svg.setAttribute("class", element.className);
      svg.innerHTML = icons[name] || fallback;
      element.replaceWith(svg);
    });
  }

  window.lucide = { createIcons };
})();

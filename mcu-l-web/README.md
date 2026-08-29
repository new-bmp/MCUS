# MCUS Web

这是 MCUS 的 Web 版，静态页面与 Android 应用共用同一份离线目录快照、厂商头像和手机本地自然语言槽位模型（v0.7.0）。内置离线选型助手，可用口语化中文描述应用场景、主频、存储、核心、软偏好和外设约束；支持“跑得快、吃电少、两三路串口、够用就行、别带无线”“M33 或 M4、不要 M7”等表达，近似推荐仍锁定在明确的核心 / 厂商 / 型号范围内。

目录包含雅特力（Artery / ArteryTek）AT32F、AT32A、AT32L、AT32M、AT32WB 26 条产品线、222 个官方型号变体，支持“雅特力 / Artery / AT32”搜索和助手筛选。

1.1 正式页面仍不显示询价入口。Worker 已加入云汉芯城开放平台试接实现，但必须先取得云汉分配的接口域名、`appid` 与 `appkey`，完成真实接口验证后才能开启。

器件详情会显示来源中的封装名称和引脚数，并生成 QFP、QFN、BGA、LGA、SOP、DIP 等封装示意图；同时列出厂商页面、数据手册和 CMSIS Pack 文档。在线 HTTPS 文档可以直接打开，Pack 内相对路径会保留原路径并提供来源 Pack 链接，精确焊盘尺寸和引脚定义仍以对应手册为准。

可直接发布的成品位于 `staticfiles` 文件夹。把该文件夹内的全部文件上传到 Cloudflare Pages、Workers Static Assets 或任意静态服务器即可，不需要后端接口。目录已经拆为多个小型 JavaScript 分片，避免单个大文件在上传时被遗漏或拒绝。

## 本地预览

在 `mcu-l-web` 目录执行：

```powershell
.\scripts\sync_assets.ps1
```

该命令会同时更新 `public`、`staticfiles` 和 `MCUS-staticfiles.zip`。部署时必须上传 `staticfiles` 内的全部文件；如果平台支持 ZIP 上传，优先使用 `MCUS-staticfiles.zip`，可避免遗漏分片。更新部署后建议清理 Cloudflare 缓存。

## Cloudflare 部署

需要先登录 Wrangler：

```powershell
npx wrangler login
.\scripts\sync_assets.ps1
npx wrangler deploy
```

`wrangler.toml` 使用 Workers Static Assets，Worker 提供 `/health` 健康检查，其余页面请求交给 `staticfiles` 静态资源，并启用 SPA 回退。目录与参数数据均为随包离线快照。每次 Android 目录更新后重新运行 `sync_assets.ps1`。

## 云汉芯城询价试接

先在 [云汉芯城数据对接申请页](https://www.ickey.cn/api) 申请接口。云汉会提供实际请求域名、`appid` 和 `appkey`；文档中的 `{domain_name}` 不是可直接调用的地址。接口协议只允许为询价、购买目的使用数据，并限制擅自存储、展示和传播，因此还必须取得云汉对“在 MCUS 中向终端用户展示实时报价”的明确书面许可。

取得参数后，把它们保存为 Worker Secret：

```powershell
npx wrangler secret put ICKEY_API_BASE
npx wrangler secret put ICKEY_APP_ID
npx wrangler secret put ICKEY_APP_KEY
npx wrangler secret put MCUS_QUOTES_ENABLED
```

获得展示授权后，才能把 `MCUS_QUOTES_ENABLED` 设为 `true`。Worker 的 `/api/quotes?part=STM32F103C8T6&quantity=20` 会使用完整订货号精确匹配，并返回最多三条云汉货源，包含人民币阶梯价、库存、MOQ、包装、批次、交期和详情链接。报价响应使用 `Cache-Control: no-store`，不建立价格数据库。正式启用页面入口前，还需把 `quote-config.js` 中的 `MCUS_QUOTES_ENABLED` 改为 `true` 并重新生成静态包。

# MCUS Web

这是 MCUS 的 Web 版，静态页面与 Android 应用共用同一份离线目录快照、厂商头像和手机本地自然语言槽位模型（v0.6.0）。内置离线选型助手，可用口语化中文描述应用场景、主频、存储、核心、软偏好和外设约束；支持“跑得快、吃电少、两三路串口、够用就行、别带无线”等表达，近似推荐仍锁定在明确的核心 / 厂商 / 型号范围内。

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

`wrangler.toml` 使用 Workers Static Assets，Worker 只提供 `/health` 健康检查，其余请求交给 `staticfiles` 静态资源，并启用 SPA 回退。目录与参数数据均为随包离线快照，不依赖价格或商品接口。

如果不使用 Wrangler，只需把 `staticfiles` 文件夹上传即可。每次 Android 目录更新后重新运行 `sync_assets.ps1`。

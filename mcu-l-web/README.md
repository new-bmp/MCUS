# MCUS Web

这是 MCUS 的 Web 版，静态页面与 Android 应用共用同一份离线目录快照和厂商头像。

可直接发布的成品位于 `staticfiles` 文件夹。把该文件夹内的全部文件上传到 Cloudflare Pages、Workers Static Assets 或任意静态服务器即可，不需要后端接口。

## 本地预览

在 `mcu-l-web` 目录执行：

```powershell
.\scripts\sync_assets.ps1
```

该命令会同时更新 `public` 和 `staticfiles`。本地预览时可直接用静态服务器打开 `staticfiles/index.html`。

## Cloudflare 部署

需要先登录 Wrangler：

```powershell
npx wrangler login
.\scripts\sync_assets.ps1
npx wrangler deploy
```

`wrangler.toml` 使用 Workers Static Assets，Worker 负责 `/health`，其余请求交给 `staticfiles` 静态资源，并启用 SPA 回退。目录数据不会在运行时请求第三方服务，适合工程现场查选型。

如果不使用 Wrangler，只需把 `staticfiles` 文件夹上传即可。每次 Android 目录更新后重新运行 `sync_assets.ps1`。

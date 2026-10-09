# 3DGRT Viewer — WebGL2

当前主页使用 **WebGL2 二维 EWA Gaussian Splatting**，在访问者本地显卡渲染，无需 WebGPU、CUDA 服务或视频串流。当前默认模型直接使用针孔投影，标签页名称仍为 `3DGRT Test`。

原始 `fc-room1` 来自 `runs/fisheye/room1-1008_224810` 的 30000 步鱼眼 3DGS 模型：236450 个高斯。其本地文件仍保留；因为目前远端仅上线针孔微调模型，它暂不列入页面模型列表。

当前唯一且默认的模型是 **room1 · 针孔微调试验**（`fc-room1-pinhole-trial`），下载入口为 `https://models.hhds-us01.xyz/models/fc-room1-pinhole-trial/manifest.json`。它在上述 checkpoint 副本上额外优化 2000 步，加入原始鱼眼像素的针孔辅助监督与角度分裂，共 240034 个高斯、三阶 SH，保留 FP32 几何与 FP16 SH；分片经 gzip 和字节重排压缩后约 27.2 MiB，在 Worker 中解码并校验 SHA256。该条目保留 `output_projection: "pinhole"`，没有中间鱼眼图或显示重采样。校正已在核对 1118 个相机位姿一致后迁移。目录项只支持 `pinhole` 和 `fisheye-perspective`，缺省为后者。

这次有界试验的 76 张鱼眼验证图平均有效区 PSNR 为 31.25 → 31.27 dB，12 张验证图的四个直接针孔方向共 48 个结果为 26.16 → 30.23 dB。相同步数普通鱼眼微调对照为 31.38 / 26.27 dB。针孔误差是在相同原始鱼眼像素上比较预测颜色，未经真值图预重投影；自由移动视角仅作定性检查。部分走廊仍有遮挡，尚不能保证任意新视角或手机持续帧率。模型及训练源文件均未覆盖，完整报告位于 FullCircle 的 `runs/fisheye-pinhole-trial/evaluation-2000/report.json`。

`models.json` 目前只列出可由这条光栅化管线正确显示的模型。旧 3DGRT 四次核光追模型不会被静默解释为 3DGS。原模型文件保留，旧模型列表备份为 `models-webgpu.json`，原 WebGPU 查看器可从 `legacy-webgpu/` 打开（例如 `http://localhost:8080/legacy-webgpu/`）。旧 lounge 模型仍使用该归档入口。

## 本地查看与发布

在此目录执行：

```bash
python -m http.server 8080 --bind 127.0.0.1
```

在同一台机器打开 `http://localhost:8080/`。正式发布保持 GitHub Pages 的 main 分支、根目录配置，并使用其 HTTPS URL。WebGL2 本身不依赖 WebGPU；本页面需要 HTTPS/localhost 才能使用模型完整性校验。部署更新后强制刷新，避免混用缓存中的旧 JS 和新模型。

鼠标左键转向、右键平移，滚轮或 WASD 前后移动，Q/E 升降。手机单指转向，双指拖动平移，张开/捏合前后移动。手机默认静止渲染长边 720，不按高 DPR 放大像素数，移动时自适应分辨率；控制栏默认折叠，鱼眼对比窗默认关闭。30 FPS 是交互目标，不是所有中端手机的性能保证。

右下角可选预览是同一位置/朝向的 **180° 等距鱼眼**，不等同于 X4 的实测四系数畸变。它与主画面共享 GPU 模型，移动中暂停更新。页面的“安全退出”会停止下载和排序 Worker，等待已提交绘制后清理纹理/缓冲区；浏览器拒绝自动关页时可手动关闭。

## 模型下载加速：页面留在 GitHub Pages

当前加载器已经支持跨域模型地址，无需更换渲染器。可以让 HTML/JS/Worker 继续由 GitHub Pages 提供，只把一个模型目录内的 `manifest.json`、全部分片和可选 `alignment.json` 放到对象存储/CDN。这样改善的是模型下载；若 GitHub 页面或 JS 本身也慢，仍需另外处理页面托管。

| 访问范围 | 可优先试用的方案 | 边界 |
| --- | --- | --- |
| 海外为主、或成本优先的候选 | Cloudflare R2 + 自定义 HTTPS 域名 | 自定义域名可接入缓存；`r2.dev` 是有限流的开发入口。R2 免出站流量费不等于所有存储/请求免费，也不保证中国大陆访问速度。见 [R2 公共访问](https://developers.cloudflare.com/r2/buckets/public-buckets/)、[费用](https://developers.cloudflare.com/r2/pricing/)。 |
| 中国大陆与海外都有访客 | 阿里云 OSS + 全球 CDN，或腾讯云 COS + 全球 CDN | 优先评估包含中国大陆节点的全球加速；加速域名须符合平台备案要求，存储和 CDN 另行计费。见 [阿里 CDN 加速区域](https://www.alibabacloud.com/help/en/cdn/user-guide/change-the-accelerated-region)、[腾讯 CDN 域名要求](https://intl.cloud.tencent.com/document/product/228/36178?lang=en)。 |

当前需要同时服务中国大陆与海外，建议先比较 OSS/COS 的全球 CDN 方案。暂未完成域名备案时，可先小范围测试香港或新加坡对象存储源站 + 中国大陆以外 CDN；它不提供中国大陆节点，大陆访问仍经过跨境链路，不能保证比 GitHub/R2 更快或稳定。请分别从大陆和海外网络下载同一模型分片，比较首次加载和缓存命中后的耗时，再决定迁移。[加速区域与备案要求](https://www.alibabacloud.com/help/en/cdn/user-guide/change-the-accelerated-region)、[OSS 地域与跨境访问边界](https://www.alibabacloud.com/help/en/oss/user-guide/regions-and-endpoints)

建议按版本上传，例如 `models/room1-20261009/`；确认新目录全部可访问后，再将 `models.json` 对应项的 `url` 改为完整 HTTPS 地址，其他模型条目保持原样：

```json
{
  "id": "fc-room1",
  "name": "room1",
  "url": "https://models.example.com/models/room1-20261009/manifest.json"
}
```

分片路径相对 **manifest 地址** 解析；`alignment.json` 也从同一远端目录读取，不能仅放在 GitHub 页面目录。复制目录时保持导出文件名和 manifest 中的长度、SHA256 不变。跨域校正保存会下载 `alignment.json`，需要手动上传回该模型目录，不会从浏览器直接写入对象存储。

**CORS：** 在对象存储和 CDN 最终响应上允许页面来源。下面是 R2 控制台 CORS JSON 示例；把 `USER` 换成实际 Pages 用户名，使用自定义页面域名时也要加入那个来源。来源只写 `https://域名[:端口]`，不要带仓库路径或末尾 `/`。OSS/COS 控制台填写同样的 GET/HEAD 与来源规则，JSON 格式以各自工具为准。见 [R2 CORS](https://developers.cloudflare.com/r2/buckets/cors/)、[COS 跨域设置](https://cloud.tencent.com/document/product/436/13318)。

```json
[
  {
    "AllowedOrigins": ["https://USER.github.io"],
    "AllowedMethods": ["GET", "HEAD"],
    "AllowedHeaders": ["*"],
    "ExposeHeaders": ["ETag", "Content-Length"],
    "MaxAgeSeconds": 3600
  }
]
```

本项目使用 `credentials: 'omit'`，此方案需要无需 Cookie 登录即可读取的模型 URL；CORS 不等于私有访问控制。manifest、分片、校正文件以及校正文件缺失时的 404 都应返回正确 CORS 响应头。若按不同 Origin 返回不同允许来源，要配置 `Vary: Origin` 及相应 CDN 缓存规则；修改 CORS 后清除已有 CDN 缓存，避免命中旧响应头。[R2 缓存与 CORS](https://developers.cloudflare.com/r2/buckets/cors/#use-cors-with-a-custom-domain)、[OSS 跨域配置](https://www.alibabacloud.com/help/en/oss/user-guide/configure-cross-origin-resource-sharing)

**文件响应头与缓存：**

| 文件 | Content-Type | 建议 Cache-Control |
| --- | --- | --- |
| 固定版本目录中的 `.bin` / `.bin.shuf.gz` | `application/octet-stream` | `public, max-age=31536000, immutable` |
| `manifest.json` | `application/json` | `no-cache`；若完全不覆盖该版本，也可长缓存 |
| 允许后续编辑的 `alignment.json` | `application/json` | `no-store`，CDN 规则也绕过缓存 |

压缩分片**不要设置 `Content-Encoding: gzip`**，关闭这些模型分片的 CDN 自动重压缩/内容改写；这里的 gzip 是模型传输格式，由 Worker 解压，再逆字节重排和 SHA256 校验。当前加载器虽然兼容浏览器自动解压的响应，仍建议保持导出字节原样，避免多层编码配置混淆。服务器的 `Content-Length` 对应实际存储/传输的压缩文件长度，而 manifest 的 `byteLength` 是解码后的长度。对象元数据可设置这些响应头，见 [OSS 元数据说明](https://www.alibabacloud.com/help/en/oss/user-guide/manage-object-metadata-10)。

Cloudflare 默认可缓存 `.gz`、`.bin`，不会默认缓存 JSON；如另加 Cache Rules，只针对不可变分片，不要将 `alignment.json` 一并长期缓存。更新模型使用新版本目录和新 manifest URL，避免旧分片与新 manifest 混用。[Cloudflare 默认缓存行为](https://developers.cloudflare.com/cache/concepts/default-cache-behavior/)

迁移前可先检查实际 CDN 响应，再在浏览器 Network 中确认分片请求转到新域名：

```bash
curl -I -H 'Origin: https://USER.github.io' \
  'https://models.example.com/models/room1-20261009/manifest.json'
```

已用两个不同端口的本地 HTTP 来源验证当前发布代码：跨域 manifest、gzip + shuffle4 geometry、gzip + shuffle2 FP16 SH 和相对路径校正均正确加载，解码后字节/hash 一致；缺少 CORS 时正确失败，404 校正被视为不存在，下载不带 Cookie/Authorization。复现命令为 FullCircle 根目录下 `node WebViewer-WebGL2/tests/cdn_loading_smoke.mjs`。这验证兼容性，不代表已上传模型或测过某家 CDN 的真实速度。

## 复用 Bandwagon VPS：Ubuntu 24 模型源站

已有洛杉矶 VPS 可作为模型源站，不需要为了存模型另开 OSS。先建立 HTTPS 下载并从大陆、海外测速，再决定是否接入 CDN。单台 VPS 是源站，多地缓存由 CDN 服务商提供；海外 CDN 不保证比当前 VPS 直连更快。以下配置仅提供公开模型文件，不运行训练或服务端渲染。

预备配置在 [`deploy/nginx-models.conf`](deploy/nginx-models.conf)，域名为 `models.hhds-us01.xyz`，根目录为 `/var/www/3dgrt-assets`。保留模型的原始 gzip 文件，成功的二进制响应缓存 30 天，JSON、校正和错误响应不缓存；公开无凭据下载使用 `Access-Control-Allow-Origin: *`。配置是签发证书之前的 HTTP 版本，Certbot 会补上 HTTPS。

该配置已用独立的 Ubuntu 24.04 nginx 1.24.0 通过语法和实际 HTTP 检查：JSON/校正、原始 gzip 字节、CORS、HEAD、Range 206、ETag 304、404 不缓存、POST 拒绝均符合预期。此验证没有安装系统服务，也未连接远端 VPS；公网 DNS 与 HTTPS 仍需在部署时确认。

1. 在当前 DNS 托管平台添加 `models` 的 A 记录，指向 VPS 的公网 IPv4。只为确认可用的地址添加记录；签发证书前需解析生效且外网可访问 80/443。

2. 在 **VPS** 上检查端口；有监听时先按已有服务调整。以下安装流程按 80/443 空闲编写：

   ```bash
   sudo ss -ltnp '( sport = :80 or sport = :443 )'
   sudo apt update
   sudo apt install nginx certbot python3-certbot-nginx rsync
   sudo install -d -m 755 /var/www/3dgrt-assets/models
   sudo chown "$(id -un):$(id -gn)" /var/www/3dgrt-assets/models
   sudo ufw status
   ```

   若 UFW 已启用，添加 `sudo ufw allow 80/tcp` 和 `sudo ufw allow 443/tcp`；VPS 面板中若另有防火墙也需允许这两个端口。此流程不要求新启用 UFW，也不改 SSH 端口。

3. 在 **本地代码仓库** 上传模型与站点配置。把 `VPS_USER`、`VPS_IP` 换成实际 SSH 用户和地址：

   ```bash
   cd /home/harrison/droneGS/fullcircle/3DGRT-WebViewer
   rsync -av --chmod=D755,F644 \
     models/fc-room1-webgl2 models/fc-room1-pinhole-trial \
     VPS_USER@VPS_IP:/var/www/3dgrt-assets/models/
   scp deploy/nginx-models.conf VPS_USER@VPS_IP:/tmp/3dgrt-models.conf
   ```

   若 SSH 不是 22 端口，给 `scp` 加 `-P 端口`，给 `rsync` 加 `-e 'ssh -p 端口'`。不要将服务器密码或 SSH 私钥写入网页代码。

4. 在 **VPS** 上首次启用该站点。只新增自己的站点文件，保留其他站点：

   ```bash
   sudo install -m 644 /tmp/3dgrt-models.conf /etc/nginx/sites-available/3dgrt-models
   sudo ln -s /etc/nginx/sites-available/3dgrt-models /etc/nginx/sites-enabled/3dgrt-models
   sudo nginx -t
   sudo systemctl reload nginx
   sudo certbot --nginx -d models.hhds-us01.xyz --redirect
   sudo certbot renew --dry-run
   ```

   `nginx -t` 成功后才 reload；Certbot 按提示完成邮箱和证书申请。Ubuntu 24.04 官方仓库提供这些组件。若 nginx 已是其他安装方式或已有证书，先结合现有配置调整。签发成功后不要再直接用 HTTP 模板覆盖 Certbot 已修改的站点。

5. 验证实际 HTTPS 文件，而不是只有域名根路径（根路径按设计返回 404）：

   ```bash
   curl -I -H 'Origin: https://USER.github.io' \
     'https://models.hhds-us01.xyz/models/fc-room1-pinhole-trial/manifest.json'
   curl -I \
     'https://models.hhds-us01.xyz/models/fc-room1-pinhole-trial/sh.000.bin.shuf.gz'
   curl -L -o /dev/null \
     -w '\nseconds=%{time_total} speed_bytes_per_second=%{speed_download}\n' \
     'https://models.hhds-us01.xyz/models/fc-room1-pinhole-trial/sh.000.bin.shuf.gz'
   ```

   应返回 200、CORS 允许来源，JSON 为 `no-store`，二进制为长缓存；gzip 文件不应有 `Content-Encoding: gzip`。再将 `models.json` 的两个 URL 分别改成此域名下的对应 manifest 地址，保留 `id`、`default` 和试验条目的 `output_projection`。只有实际源站验证通过后再发布列表更新。

6. 如直连需要进一步加速，可以把此 VPS 接入阿里云等支持自有源站的 CDN。添加 `models.hhds-us01.xyz`，源站类型选 **IP** 并填写 VPS IP，回源 Host 和 HTTPS SNI 使用 `models.hhds-us01.xyz`。配置缓存/CORS/客户端证书后，将 `models` 的 DNS A 记录改为 CDN 分配的 CNAME；网页模型 URL 不用再次更改。尚未备案时选择 **全球（不包含中国内地）**；接入中国内地节点仍需备案，海外源站不会免除该条件。大陆和海外分别比较首次回源与缓存命中的速度，保留测量结果再决定长期线路。

相关官方说明：[Ubuntu nginx 安装](https://ubuntu.com/server/docs/how-to-install-nginx/)、[阿里 CDN 加速区域](https://help.aliyun.com/zh/cdn/user-guide/change-the-accelerated-region/)、[CDN 备案要求](https://help.aliyun.com/zh/icp-filing/basic-icp-service/product-overview/use-alibaba-cloud-cdn)。这些文件是本地部署准备，不代表已连接或配置你的 VPS/DNS。

## 移动视角时的雾状遮挡

鱼眼训练得到的部分高斯在视线侧方仍有较大尺度。直接套用针孔的一阶协方差投影时，它们可能变成覆盖屏幕的大斑块；原生 CUDA 针孔渲染也能复现，并非只发生在 WebGL2。仅对比 WebGL 与 CUDA 针孔输出，不能证明这个模型在所有转向和位移后都没有伪影。

为原始模型保留的 `fisheye-perspective` 模式先用等距鱼眼 EWA 投影，再按每个透视像素的方向采样浮点累积结果。只渲染包含当前视野的鱼眼矩形，中心采样密度与输出匹配，无需生成完整的高分辨率 180° 圆图。此模式保留全部模型数据、径向排序和 SH；不需要重新训练或导出模型，也不改变训练输入图像。最后的双线性重采样可能使宽视角边缘略软，仍不保证所有新视角都没有模型自身的伪影。

直接针孔投影通过渲染器 API `render({width, height, projection: 'pinhole'})` 调用，当前默认微调模型已使用此模式。右下角仍是独立的完整鱼眼预览。

## 导出后续 3DGS 模型

使用 `fisheye` 环境，在 FullCircle 根目录执行：

```bash
python 3DGRT-WebViewer/export_model.py \
  --checkpoint runs/fisheye/room1-1008_224810/ours_30000/ckpt_30000.pt \
  --out-dir 3DGRT-WebViewer/models/room1-next \
  --sh-dtype float16 --compression gzip-shuffle
```

输出目录默认不能已存在；更新同一 EWA 模型可显式加 `--overwrite`。脚本只接受 `render.method=3dgs`、径向排序和当前支持的二维 EWA 配置。它生成 `fullcircle-webgl2-ewa-v1` 格式，包含 48 字节/高斯的协方差记录和 SH 分片，不再需要 BVH。将新 manifest 地址加入 `models.json` 即可从页面切换。`compress_model.py` 可对这种新格式重新压缩，旧光追模型应使用 `WebViewer-WebGPU/` 的原导出工具。

## 地平面校正

模型目录中的 `alignment.json` 仍会自动加载，并验证 source、数量和坐标范围。此次 room1 迁移前已核对新旧模型的 1118 个相机位置与旋转完全一致，因此保留原校正的 origin/rotation/enabled，并将身份字段绑定到新模型。

页面继续支持可视化地面、三轴拖动和旋转。在 FullCircle 根目录使用共享保存服务，可直接写入当前模型文件夹：

```bash
python WebViewer-WebGPU/serve.py \
  --directory 3DGRT-WebViewer --bind 127.0.0.1 --port 8080
```

GitHub Pages 或普通静态服务会下载校正文件，由用户放回当前模型目录后推送。重新训练导致模型身份变化时，需核对坐标系再迁移校正，不能只改身份字段绕过检查。

## 验证范围

完整训练参数、优化器状态与 PLY 数量/数值检查通过。76 张原始鱼眼验证图的有效区域平均 PSNR 为 31.25 dB，局部走廊仍存在模糊，未裁掉模型中的大尺度高斯。

原针孔参考分支在桌面 Chrome/RTX4090 上，三个实际模型视角在 120×96 或 320×240 下与同一导出数据的 CUDA 针孔参考对照：PSNR 63.36–63.91 dB，RGB8 最大差值 2–3。这只验证针孔实现的一致性，不是当前默认鱼眼透视路径的画质保证。合成 180° 鱼眼和模型切换、SH、校正、保存 PNG、安全退出也通过。390×844、DPR3 的触屏模拟实际渲染 333×720；这不是手机 GPU 性能测试。

当前默认路径另外检查了完整模型在找平、转向 +60°/−30°、平移后转向四个位姿的 960×503 输出，以及移动位姿的 333×720 竖屏输出。与同一鱼眼裁切参数的 CUDA 浮点渲染加独立重采样对照，PSNR 为 63.38–63.75 dB，最大通道差值 3/255。测试视角中，旧针孔路径的走廊遮挡明显减轻；这是实现对照，不是对真实图像的重建 PSNR。模型切换、触屏与窗口变化、安全退出也通过；退出后测试追踪的 GL 资源为零。

累积目标优先使用 RGBA32F，其次 RGBA16F；缺少浮点目标的设备回退 RGBA8，页面会提示兼容色彩精度。16F 单视角相对 32F 的最大差值为 2/255，8bit 回退尚无同等画质保证。固定功能混合不能复现 CUDA 的逐像素提前终止，近乎相等的深度也可能因浮点舍入产生不同顺序，因此不承诺逐像素完全相等。

本机冷 headless Chrome 在 GPU 初始化期间会丢失上下文，空白 WebGL 页面也能复现；自动化 UI 测试显式等待浏览器初始化后通过。真实中端手机的持续 30 FPS 和热稳定性仍需实机验证。

代码来源与许可：基础界面和工具保留 NVIDIA/FullCircle 的 `LICENSE`；二维 EWA/SH 参考实现与 DirectFisheye-GS 来源说明见 `renderer.js` 和 `LICENSE-3DGS.md`。

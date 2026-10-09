# 3DGRT Viewer — WebGL2

当前主页使用 **WebGL2 二维 EWA Gaussian Splatting**，在访问者本地显卡渲染，无需 WebGPU、CUDA 服务或视频串流。主画面先在等距鱼眼投影中渲染当前视野需要的区域，再映射为透视画面；标签页名称仍为 `3DGRT Test`。

默认 `fc-room1` 是 `runs/fisheye/room1-1008_224810` 的 30000 步原始鱼眼 3DGS 模型：236450 个高斯、全部三阶 SH、FP32 位置/协方差与 FP16 SH。导出不剪枝；geometry/SH 经 gzip 和字节重排压缩后约 26.76 MiB，在 Worker 中解压并校验 SHA256，再直接上传 GPU。

`models.json` 目前只列出可由这条光栅化管线正确显示的模型。旧 3DGRT 四次核光追模型不会被静默解释为 3DGS。原模型文件保留，旧模型列表备份为 `models-webgpu.json`，原 WebGPU 查看器可从 `legacy-webgpu/` 打开（例如 `http://localhost:8080/legacy-webgpu/`）。旧 lounge 模型仍使用该归档入口。

## 本地查看与发布

在此目录执行：

```bash
python -m http.server 8080 --bind 127.0.0.1
```

在同一台机器打开 `http://localhost:8080/`。正式发布保持 GitHub Pages 的 main 分支、根目录配置，并使用其 HTTPS URL。WebGL2 本身不依赖 WebGPU；本页面需要 HTTPS/localhost 才能使用模型完整性校验。部署更新后强制刷新，避免混用缓存中的旧 JS 和新模型。

鼠标左键转向、右键平移，滚轮或 WASD 前后移动，Q/E 升降。手机单指转向，双指拖动平移，张开/捏合前后移动。手机默认静止渲染长边 720，不按高 DPR 放大像素数，移动时自适应分辨率；控制栏默认折叠，鱼眼对比窗默认关闭。30 FPS 是交互目标，不是所有中端手机的性能保证。

右下角可选预览是同一位置/朝向的 **180° 等距鱼眼**，不等同于 X4 的实测四系数畸变。它与主画面共享 GPU 模型，移动中暂停更新。页面的“安全退出”会停止下载和排序 Worker，等待已提交绘制后清理纹理/缓冲区；浏览器拒绝自动关页时可手动关闭。

## 移动视角时的雾状遮挡

鱼眼训练得到的部分高斯在视线侧方仍有较大尺度。直接套用针孔的一阶协方差投影时，它们可能变成覆盖屏幕的大斑块；原生 CUDA 针孔渲染也能复现，并非只发生在 WebGL2。仅对比 WebGL 与 CUDA 针孔输出，不能证明这个模型在所有转向和位移后都没有伪影。

当前默认先用等距鱼眼 EWA 投影，再按每个透视像素的方向采样浮点累积结果。只渲染包含当前视野的鱼眼矩形，中心采样密度与输出匹配，无需生成完整的高分辨率 180° 圆图。此改动保留全部模型数据、径向排序和 SH；不需要重新训练或导出模型，也不改变训练输入图像。最后的双线性重采样可能使宽视角边缘略软，仍不保证所有新视角都没有模型自身的伪影。

用于诊断的原始针孔投影仍可通过渲染器 API `render({width, height, projection: 'pinhole'})` 调用。页面默认使用 `fisheye-perspective`，右下角仍是独立的完整鱼眼预览。

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

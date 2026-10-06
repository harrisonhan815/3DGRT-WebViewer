# FullCircle WebGPU viewer

Static browser renderer. Models are FP32 inference data with BVHs; no Python server is required.

Publish this directory as a separate GitHub repository. In Settings > Pages, select Deploy from a branch, main, /(root). Open the resulting HTTPS URL in a WebGPU-capable browser.

Local preview: `python -m http.server 8080 --bind 127.0.0.1`, then open http://localhost:8080. Remote HTTP is not a secure context for WebGPU.

Controls: drag to look, right-drag to pan, wheel or WASD to move, Q/E vertically. Use the resolution selector to match the client GPU.

The renderer ports the native 3DGRT reference equations to WGSL and uses software BVH traversal. It is a prototype; GPU/OptiX numerical agreement and performance depend on the scene and hardware.

Equations adapted from NVIDIA 3DGRT, Copyright 2025 NVIDIA CORPORATION & AFFILIATES, Apache-2.0; see LICENSE.

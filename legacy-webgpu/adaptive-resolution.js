// Adjust only interaction image resolution. Gaussian parameters, SH, hit
// ordering and opacity thresholds are never changed by this controller.
export class AdaptiveResolution {
  constructor({minWidth = 160, settleMs = 200, targetFps = 30} = {}) {
    this.minWidth = minWidth;
    this.settleMs = settleMs;
    this.targetFps = targetFps;
    this.reset();
  }

  reset() {
    this.lastMotion = -Infinity;
    this.baseWidth = 0;
    this.movingWidth = 0;
    this.fastFrames = 0;
    this.lastSample = null;
  }

  markMotion(now) {
    this.lastMotion = now;
  }

  setTarget(fps) {
    if (!Number.isFinite(fps) || fps <= 0) throw new Error('Target FPS must be positive');
    this.targetFps = fps;
    this.fastFrames = 0;
  }

  clampWidth(width, baseWidth = this.baseWidth) {
    const minimum = Math.min(this.minWidth, baseWidth);
    return Math.max(minimum, Math.min(baseWidth, Math.floor(width / 16) * 16));
  }

  select(baseWidth, now, enabled = true) {
    if (!Number.isInteger(baseWidth) || baseWidth < 1) throw new Error('Render width must be a positive integer');
    if (baseWidth !== this.baseWidth) {
      this.baseWidth = baseWidth;
      this.movingWidth = 0;
      this.fastFrames = 0;
    }
    const moving = enabled && now - this.lastMotion < this.settleMs;
    if (!moving) return {width: baseWidth, moving: false};
    if (!this.movingWidth) {
      // Start conservatively, then use measured completed-frame cost. A slow
      // full-resolution first frame can inform the first interaction frame.
      let initial = baseWidth * 0.6;
      if (this.lastSample) {
        const budget = 0.85 * 1000 / this.targetFps;
        initial = Math.min(initial, this.lastSample.width * Math.sqrt(budget / this.lastSample.ms));
      }
      this.movingWidth = this.clampWidth(initial);
    }
    return {width: this.movingWidth, moving: true};
  }

  record({width, renderMs}, moving) {
    if (!Number.isFinite(renderMs) || renderMs <= 0 || !Number.isFinite(width) || width < 1) return;
    this.lastSample = {width, ms: renderMs};
    if (!moving || !this.baseWidth) return;
    const budget = 0.85 * 1000 / this.targetFps;
    if (renderMs > budget * 1.15) {
      this.fastFrames = 0;
      const ratio = Math.max(0.6, Math.min(0.9, Math.sqrt(budget / renderMs)));
      this.movingWidth = this.clampWidth(width * ratio);
    } else if (renderMs < budget * 0.7) {
      // Hysteresis avoids alternating dimensions and GPU reallocations after
      // each small timing fluctuation. Recover resolution more slowly.
      this.fastFrames++;
      if (this.fastFrames >= 4) {
        this.movingWidth = this.clampWidth(Math.max(width + 16, width * 1.15));
        this.fastFrames = 0;
      }
    } else {
      this.fastFrames = 0;
    }
  }
}

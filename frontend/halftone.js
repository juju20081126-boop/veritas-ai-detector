/**
 * Veritas — halftone field.
 * Renders a dithered, photo-like ink mass as diamond dots on a 45° lattice.
 * The mass comes from fractal value noise, so it has clumps and holes rather
 * than smooth bands. After each analysis the field re-seeds and settles, and
 * a higher AI share pushes the ink further across the panel.
 * Listens for "veritas:result" ({ detail: { aiShare: 0..1 } }).
 */
(function () {
  const canvas = document.getElementById("halftoneCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  const CELL = 8;          // lattice spacing in CSS px
  const SCALE = 1 / 230;   // noise features per px

  let width = 0;
  let height = 0;
  let aiShare = 0.55;
  let offsetX = 0;
  let offsetY = 0;
  let raf = 0;

  // Integer hash -> [0, 1)
  function hash(ix, iy) {
    let h = Math.imul(ix, 374761393) + Math.imul(iy, 668265263);
    h = Math.imul(h ^ (h >>> 13), 1274126177);
    h ^= h >>> 16;
    return (h >>> 0) / 4294967296;
  }

  function valueNoise(x, y) {
    const ix = Math.floor(x);
    const iy = Math.floor(y);
    const fx = x - ix;
    const fy = y - iy;
    const sx = fx * fx * (3 - 2 * fx);
    const sy = fy * fy * (3 - 2 * fy);
    const a = hash(ix, iy);
    const b = hash(ix + 1, iy);
    const c = hash(ix, iy + 1);
    const d = hash(ix + 1, iy + 1);
    return a + (b - a) * sx + (c - a) * sy + (a - b - c + d) * sx * sy;
  }

  function fbm(x, y) {
    let sum = 0;
    let amp = 0.5;
    let freq = 1;
    for (let i = 0; i < 5; i++) {
      sum += amp * valueNoise(x * freq, y * freq);
      freq *= 2.03;
      amp *= 0.5;
    }
    return sum / 0.97;
  }

  function smoothstep(e0, e1, x) {
    const t = Math.min(1, Math.max(0, (x - e0) / (e1 - e0)));
    return t * t * (3 - 2 * t);
  }

  function resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const rect = canvas.getBoundingClientRect();
    width = Math.max(1, Math.round(rect.width));
    height = Math.max(1, Math.round(rect.height));
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    draw();
  }

  function draw() {
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#0a0a0a";
    const half = CELL / 2;
    // Low share keeps the ink to the right edge; high share floods the panel
    const edgeBase = 0.42 - aiShare * 0.34;

    for (let cy = 0, row = 0; cy < height + CELL; cy += half, row++) {
      const v = cy * SCALE;
      // Ragged left boundary that wanders down the panel
      const edge = edgeBase + (fbm(v * 1.4 + 17.3 + offsetY, 4.1) - 0.5) * 0.55;

      for (let cx = (row % 2) * half; cx < width + CELL; cx += CELL) {
        const u = cx / width;
        const mask = smoothstep(edge, edge + 0.2, u);
        if (mask <= 0) continue;

        const n = fbm(cx * SCALE + offsetX, v + offsetY);
        const tone = smoothstep(0.26, 0.76, n) * mask;
        // Capped just under touching, so the darkest areas stay a crisp checker
        const r = tone * half * 0.96;
        if (r < 0.55) continue;

        ctx.beginPath();
        ctx.moveTo(cx, cy - r);
        ctx.lineTo(cx + r, cy);
        ctx.lineTo(cx, cy + r);
        ctx.lineTo(cx - r, cy);
        ctx.closePath();
        ctx.fill();
      }
    }
  }

  function settle(targetShare) {
    cancelAnimationFrame(raf);
    const toX = offsetX + 0.9;
    const toY = offsetY + 0.4;
    if (reduceMotion) {
      aiShare = targetShare;
      offsetX = toX;
      offsetY = toY;
      draw();
      return;
    }
    const start = performance.now();
    const from = { share: aiShare, x: offsetX, y: offsetY };
    const duration = 1200;

    function step(now) {
      const t = Math.min(1, (now - start) / duration);
      const e = 1 - Math.pow(1 - t, 3);
      aiShare = from.share + (targetShare - from.share) * e;
      offsetX = from.x + (toX - from.x) * e;
      offsetY = from.y + (toY - from.y) * e;
      draw();
      if (t < 1) raf = requestAnimationFrame(step);
    }
    raf = requestAnimationFrame(step);
  }

  window.addEventListener("veritas:result", (e) => {
    const share = e.detail && typeof e.detail.aiShare === "number" ? e.detail.aiShare : 0.5;
    settle(Math.max(0, Math.min(1, share)));
  });

  if ("ResizeObserver" in window) {
    new ResizeObserver(resize).observe(canvas);
  } else {
    window.addEventListener("resize", resize);
  }
  resize();
})();

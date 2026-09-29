/**
 * Veritas — halftone field.
 * Ordered dither on a 45° diamond lattice, like a newsprint photo: every dot is
 * the same size, and tone comes from which dots are switched on (Bayer matrix).
 * The underlying image is fractal noise with a ragged left edge.
 * Events:
 *   "veritas:busy"   ({ detail: { busy } })  field drifts while analysis runs
 *   "veritas:result" ({ detail: { aiShare } }) field re-forms; more AI = more ink
 */
(function () {
  const canvas = document.getElementById("halftoneCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // Square grid of diamonds touching only at their corners: full tone reads
  // as a black/white checker, never a solid slab.
  const CELL = 7;
  const DOT = CELL / 2;
  const SCALE = 1 / 240;

  // 4x4 Bayer thresholds
  const BAYER = [
    [0, 8, 2, 10],
    [12, 4, 14, 6],
    [3, 11, 1, 9],
    [15, 7, 13, 5],
  ].map((row) => row.map((v) => (v + 0.5) / 16));

  let width = 0;
  let height = 0;
  let aiShare = 0.55;
  let offsetX = 0;
  let offsetY = 0;
  let raf = 0;
  let busy = false;

  function hash(ix, iy) {
    let h = Math.imul(ix, 374761393) + Math.imul(iy, 668265263);
    h = Math.imul(h ^ (h >>> 13), 1274126177);
    h ^= h >>> 16;
    return (h >>> 0) / 4294967296;
  }

  function valueNoise(x, y) {
    const ix = Math.floor(x), iy = Math.floor(y);
    const fx = x - ix, fy = y - iy;
    const sx = fx * fx * (3 - 2 * fx), sy = fy * fy * (3 - 2 * fy);
    const a = hash(ix, iy), b = hash(ix + 1, iy);
    const c = hash(ix, iy + 1), d = hash(ix + 1, iy + 1);
    return a + (b - a) * sx + (c - a) * sy + (a - b - c + d) * sx * sy;
  }

  function fbm(x, y) {
    let sum = 0, amp = 0.5, f = 1;
    for (let i = 0; i < 5; i++) {
      sum += amp * valueNoise(x * f, y * f);
      f *= 2.03;
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
    ctx.beginPath();

    const edgeBase = 0.44 - aiShare * 0.36;

    for (let row = 0, cy = DOT; cy < height + CELL; row++, cy += CELL) {
      const v = cy * SCALE;
      const edge = edgeBase + (fbm(v * 1.5 + 17.3 + offsetY, 4.1) - 0.5) * 0.6;

      for (let col = 0, cx = DOT; cx < width + CELL; col++, cx += CELL) {
        const u = cx / width;
        const mask = smoothstep(edge, edge + 0.16, u);
        if (mask <= 0) continue;

        let tone = smoothstep(0.36, 0.66, fbm(cx * SCALE + offsetX, v + offsetY)) * mask;
        // Stray dots along the boundary, like ink breaking up at a photo's edge
        if (mask < 1) tone *= 0.55 + hash(col + 91, row + 7) * 0.9;

        const threshold = BAYER[row & 3][col & 3];
        if (tone <= threshold) continue;

        // In the deepest shadows the diamonds swell until only small white holes remain
        const r = tone > 0.82 ? DOT * (1 + (tone - 0.82) * 1.6) : DOT;
        ctx.moveTo(cx, cy - r);
        ctx.lineTo(cx + r, cy);
        ctx.lineTo(cx, cy + r);
        ctx.lineTo(cx - r, cy);
        ctx.closePath();
      }
    }
    ctx.fill();
  }

  function animateTo(targetShare, dx, dy, duration, onDone) {
    cancelAnimationFrame(raf);
    const toX = offsetX + dx;
    const toY = offsetY + dy;
    if (reduceMotion) {
      aiShare = targetShare;
      offsetX = toX;
      offsetY = toY;
      draw();
      if (onDone) onDone();
      return;
    }
    const start = performance.now();
    const from = { share: aiShare, x: offsetX, y: offsetY };

    function step(now) {
      const t = Math.min(1, (now - start) / duration);
      const e = 1 - Math.pow(1 - t, 3);
      aiShare = from.share + (targetShare - from.share) * e;
      offsetX = from.x + (toX - from.x) * e;
      offsetY = from.y + (toY - from.y) * e;
      draw();
      if (t < 1) raf = requestAnimationFrame(step);
      else if (onDone) onDone();
    }
    raf = requestAnimationFrame(step);
  }

  // Slow drift while the model runs, so the page shows it is working
  function drift() {
    if (!busy || reduceMotion) return;
    cancelAnimationFrame(raf);
    let last = performance.now();
    function step(now) {
      if (!busy) return;
      offsetX += (now - last) * 0.0004;
      last = now;
      draw();
      raf = requestAnimationFrame(step);
    }
    raf = requestAnimationFrame(step);
  }

  window.addEventListener("veritas:busy", (e) => {
    busy = Boolean(e.detail && e.detail.busy);
    if (busy) drift();
  });

  window.addEventListener("veritas:result", (e) => {
    busy = false;
    const share = e.detail && typeof e.detail.aiShare === "number" ? e.detail.aiShare : 0.5;
    animateTo(Math.max(0, Math.min(1, share)), 0.9, 0.4, 1300);
  });

  if ("ResizeObserver" in window) {
    new ResizeObserver(resize).observe(canvas);
  } else {
    window.addEventListener("resize", resize);
  }
  resize();
})();

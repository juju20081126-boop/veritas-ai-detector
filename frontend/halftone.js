/**
 * Veritas AI — halftone field.
 * Draws a diamond-dot dither beside the report column. Dot density follows a smooth noise field,
 * and after each analysis the field re-settles: denser ink means more AI.
 * Listens for "veritas:result" ({ detail: { aiShare: 0..1 } }).
 */
(function () {
  const canvas = document.getElementById("halftoneCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  const CELL = 9;
  let width = 0;
  let height = 0;
  let aiShare = 0.35;      // current (animated) value
  let targetShare = 0.35;  // value the field is settling toward
  let phase = 0;
  let raf = 0;

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

  // An organic ink mass anchored to the right edge with a ragged left boundary.
  // More AI pushes the boundary left, so the mass grows across the panel.
  function field(x, y) {
    const u = x / width;
    const v = y / height;
    const edge =
      0.34 - aiShare * 0.24 +
      Math.sin(v * 4.2 + phase * 0.6) * 0.13 +
      Math.sin(v * 11.0 - phase) * 0.05;
    const mass = Math.min(1, Math.max(0, (u - edge) / 0.22));
    const tone =
      Math.sin(u * 7.0 + v * 3.0 + phase) * 0.5 +
      Math.sin(v * 9.0 - u * 4.0 - phase * 0.7) * 0.3 +
      Math.sin((u - v) * 14.0 + phase * 1.2) * 0.2;
    // High-frequency grain gives the mass holes and clumps, like a dithered photo
    const grain =
      Math.sin(u * 26.0 - v * 19.0 + phase * 2.0) * 0.6 +
      Math.sin(u * 37.0 + v * 29.0) * 0.4;
    return mass * (0.5 + tone * 0.45 + grain * 0.22);
  }

  function draw() {
    ctx.clearRect(0, 0, width, height);
    const styles = getComputedStyle(document.documentElement);
    const rgb = styles.getPropertyValue("--halftone").trim() || "6, 9, 18";
    const alpha = parseFloat(styles.getPropertyValue("--halftone-alpha")) || 0.9;
    ctx.fillStyle = `rgba(${rgb}, ${alpha})`;

    const gain = 1.0 + aiShare * 0.5;
    const half = CELL / 2;

    for (let cy = 0, row = 0; cy < height + CELL; cy += CELL, row++) {
      for (let cx = (row % 2) * half; cx < width + CELL; cx += CELL) {
        // Contrast curve so the dots read as a bold dither, not a haze
        const level = Math.min(1, Math.max(0, (field(cx, cy) * gain - 0.12) * 1.45));
        const r = level * half * 1.05;
        if (r < 0.45) continue;
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

  function settle() {
    cancelAnimationFrame(raf);
    if (reduceMotion) {
      aiShare = targetShare;
      draw();
      return;
    }
    const start = performance.now();
    const fromShare = aiShare;
    const fromPhase = phase;
    const duration = 1100;

    function step(now) {
      const t = Math.min(1, (now - start) / duration);
      const eased = 1 - Math.pow(1 - t, 3);
      aiShare = fromShare + (targetShare - fromShare) * eased;
      phase = fromPhase + eased * 1.6;
      draw();
      if (t < 1) raf = requestAnimationFrame(step);
    }
    raf = requestAnimationFrame(step);
  }

  window.addEventListener("veritas:result", (e) => {
    const share = e.detail && typeof e.detail.aiShare === "number" ? e.detail.aiShare : 0.35;
    targetShare = Math.max(0, Math.min(1, share));
    settle();
  });

  // Theme switch changes the dot color, so repaint.
  new MutationObserver(draw).observe(document.documentElement, {
    attributes: true,
    attributeFilter: ["data-theme"],
  });

  // The report panel grows when results appear, so track the canvas box itself.
  if ("ResizeObserver" in window) {
    new ResizeObserver(resize).observe(canvas);
  } else {
    window.addEventListener("resize", resize);
  }
  resize();
})();

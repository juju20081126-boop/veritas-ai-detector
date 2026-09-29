/**
 * Veritas AI — halftone field.
 * Draws a diamond-dot dither in the hero. Dot density follows a smooth noise field,
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

  // Cheap smooth field: layered sines, plus a falloff that hugs the top-right corner.
  function field(x, y) {
    const u = x / width;
    const v = y / height;
    const waves =
      Math.sin(u * 5.2 + phase) * 0.5 +
      Math.sin(v * 6.1 - phase * 0.8 + u * 2.4) * 0.35 +
      Math.sin((u + v) * 9.0 + phase * 1.3) * 0.15;
    const falloff = Math.pow(u, 0.7) * (1 - v * 0.55);
    return (waves * 0.5 + 0.5) * falloff;
  }

  function draw() {
    ctx.clearRect(0, 0, width, height);
    const styles = getComputedStyle(document.documentElement);
    const rgb = styles.getPropertyValue("--halftone").trim() || "6, 9, 18";
    const alpha = parseFloat(styles.getPropertyValue("--halftone-alpha")) || 0.9;
    ctx.fillStyle = `rgba(${rgb}, ${alpha})`;

    const gain = 1.5 + aiShare * 1.4;
    const half = CELL / 2;

    for (let cy = 0, row = 0; cy < height + CELL; cy += CELL, row++) {
      for (let cx = (row % 2) * half; cx < width + CELL; cx += CELL) {
        // Contrast curve so the dots read as a bold dither, not a haze
        const level = Math.min(1, Math.max(0, (field(cx, cy) * gain - 0.28) * 1.5));
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

  window.addEventListener("resize", resize);
  resize();
})();

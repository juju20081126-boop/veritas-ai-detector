/**
 * Veritas — misty forest backdrop.
 * Paints a grayscale landscape once per resize: overcast sky, drifting cloud
 * banks, layered pine ridges in the lower left, fog between them and film grain.
 * Static by design so it never competes with the card.
 */
(function () {
  const canvas = document.getElementById("backdrop");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");

  function hash(ix, iy, seed) {
    let h = Math.imul(ix, 374761393) + Math.imul(iy, 668265263) + Math.imul(seed, 1442695041);
    h = Math.imul(h ^ (h >>> 13), 1274126177);
    h ^= h >>> 16;
    return (h >>> 0) / 4294967296;
  }

  function noise(x, y, seed) {
    const ix = Math.floor(x), iy = Math.floor(y);
    const fx = x - ix, fy = y - iy;
    const sx = fx * fx * (3 - 2 * fx), sy = fy * fy * (3 - 2 * fy);
    const a = hash(ix, iy, seed), b = hash(ix + 1, iy, seed);
    const c = hash(ix, iy + 1, seed), d = hash(ix + 1, iy + 1, seed);
    return a + (b - a) * sx + (c - a) * sy + (a - b - c + d) * sx * sy;
  }

  function fbm(x, y, seed) {
    let sum = 0, amp = 0.5, f = 1;
    for (let i = 0; i < 5; i++) {
      sum += amp * noise(x * f, y * f, seed + i);
      f *= 2.02;
      amp *= 0.5;
    }
    return sum / 0.97;
  }

  // Seeded random for tree placement, so the scene is identical every load
  let rs = 20260929;
  const rand = () => {
    rs = (Math.imul(rs, 1664525) + 1013904223) >>> 0;
    return rs / 4294967296;
  };

  // Cloud layer rendered small, then scaled up smooth
  function clouds(w, h, seed, scale, gain, bias) {
    const cw = Math.ceil(w / 6), ch = Math.ceil(h / 6);
    const off = document.createElement("canvas");
    off.width = cw;
    off.height = ch;
    const octx = off.getContext("2d");
    const img = octx.createImageData(cw, ch);
    for (let y = 0; y < ch; y++) {
      for (let x = 0; x < cw; x++) {
        const n = fbm((x / cw) * scale, (y / ch) * scale * 0.6, seed);
        const a = Math.max(0, Math.min(1, (n - bias) * gain));
        const i = (y * cw + x) * 4;
        img.data[i] = img.data[i + 1] = img.data[i + 2] = 255;
        img.data[i + 3] = a * 255;
      }
    }
    octx.putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(off, 0, 0, w, h);
  }

  // A spruce: narrow, many ragged branch tiers, slightly leaning, no clean symmetry
  function pine(x, baseY, height, color) {
    const tiers = 16 + Math.floor(rand() * 8);
    const maxHalf = height * (0.12 + rand() * 0.05);
    const lean = (rand() - 0.5) * height * 0.04;
    const top = baseY - height;
    const tierY = (t) => top + (height * 0.96 * t) / tiers;
    // Width swells slightly mid-crown, like a real spruce silhouette
    const widthAt = (t) => maxHalf * Math.pow(t / tiers, 0.85) * (0.6 + rand() * 0.55);
    const xAt = (t) => x + lean * (1 - t / tiers);

    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.moveTo(x + lean, top - height * 0.02);
    for (let t = 1; t <= tiers; t++) {
      const y = tierY(t);
      const w = widthAt(t);
      ctx.lineTo(xAt(t) + w, y + height * 0.012);            // drooping tip
      ctx.lineTo(xAt(t) + w * (0.2 + rand() * 0.25), y - height * 0.008);
    }
    ctx.lineTo(x + height * 0.012, baseY);
    ctx.lineTo(x - height * 0.012, baseY);
    for (let t = tiers; t >= 1; t--) {
      const y = tierY(t);
      const w = widthAt(t);
      ctx.lineTo(xAt(t) - w * (0.2 + rand() * 0.25), y - height * 0.008);
      ctx.lineTo(xAt(t) - w, y + height * 0.012);
    }
    ctx.closePath();
    ctx.fill();
  }

  function ridge(w, h, opts) {
    // Trees along a slope falling from the left edge toward the bottom
    const { startY, endX, size, color, density, fog } = opts;
    const slopeAt = (x) => startY + (h * 1.02 - startY) * Math.pow(x / endX, 1.25);
    const count = Math.floor((endX / size) * density);
    const trees = [];
    for (let i = 0; i < count; i++) trees.push(rand() * endX);
    trees.sort((a, b) => a - b);
    trees.forEach((x) => {
      const base = slopeAt(x) + size * (0.4 + rand() * 1.8);
      pine(x, base, size * (0.8 + rand() * 0.6), color);
    });
    // Solid hillside under the trees
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.moveTo(0, h);
    for (let x = 0; x <= endX; x += 8) ctx.lineTo(x, slopeAt(x) + size * 1.4);
    ctx.lineTo(endX, h);
    ctx.closePath();
    ctx.fill();
    if (fog) {
      const g = ctx.createLinearGradient(0, startY, 0, h);
      g.addColorStop(0, "rgba(220,221,222,0)");
      g.addColorStop(1, `rgba(220,221,222,${fog})`);
      ctx.fillStyle = g;
      ctx.fillRect(0, startY - size * 2, endX + size, h);
    }
  }

  function grain(w, h) {
    const img = ctx.getImageData(0, 0, w, h);
    const d = img.data;
    for (let i = 0; i < d.length; i += 4) {
      const n = (Math.random() - 0.5) * 14;
      d[i] += n;
      d[i + 1] += n;
      d[i + 2] += n;
    }
    ctx.putImageData(img, 0, 0);
  }

  function draw() {
    const w = window.innerWidth;
    const h = window.innerHeight;
    canvas.width = w;
    canvas.height = h;
    rs = 20260929;

    const sky = ctx.createLinearGradient(0, 0, 0, h);
    sky.addColorStop(0, "#bcbdbe");
    sky.addColorStop(0.55, "#a4a6a8");
    sky.addColorStop(1, "#8f9194");
    ctx.fillStyle = sky;
    ctx.fillRect(0, 0, w, h);

    // Out of focus behind the glass: far layers soft, near layer slightly sharper
    ctx.filter = "blur(3px)";
    clouds(w, h, 3, 4.2, 2.6, 0.42);
    ridge(w, h, { startY: h * 0.48, endX: w * 0.5, size: h * 0.08, color: "#8a8c8e", density: 4.5, fog: 0.55 });
    ctx.filter = "blur(2px)";
    clouds(w, h, 9, 3.0, 1.8, 0.55);
    ridge(w, h, { startY: h * 0.58, endX: w * 0.42, size: h * 0.12, color: "#505254", density: 4, fog: 0.4 });
    ctx.filter = "blur(1.2px)";
    ridge(w, h, { startY: h * 0.7, endX: w * 0.34, size: h * 0.17, color: "#1e1f20", density: 3.2, fog: 0 });
    ctx.filter = "blur(4px)";
    clouds(w, h, 21, 2.2, 1.4, 0.62);
    ctx.filter = "none";
    grain(w, h);
  }

  let t = 0;
  window.addEventListener("resize", () => {
    clearTimeout(t);
    t = setTimeout(draw, 150);
  });
  draw();
})();

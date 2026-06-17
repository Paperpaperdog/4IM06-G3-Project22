const pptxgen = require("/opt/homebrew/lib/node_modules/pptxgenjs");

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.title = "G3 Project22 - Weekly Progress";

const C = {
  navyDark:  "0A1628",
  navy:      "0D2137",
  teal:      "0E7490",
  tealLight: "06B6D4",
  ice:       "CFFAFE",
  white:     "FFFFFF",
  gray:      "94A3B8",
  cardBg:    "162032",
  good:      "10B981",
  warn:      "F59E0B",
  bad:       "EF4444",
  lightBg:   "F8FAFC",
  textDark:  "1E293B",
  textMid:   "475569",
};

const makeShadow = () => ({ type: "outer", blur: 8, offset: 3, angle: 135, color: "000000", opacity: 0.18 });

// ══════════════════════════════════════════════════════════════════════════════
// SLIDE 1 — Title
// ══════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.navyDark };

  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.06, fill: { color: C.tealLight }, line: { color: C.tealLight } });

  s.addShape(pres.shapes.OVAL, { x: 7.0, y: -0.5, w: 4.5, h: 4.5, fill: { color: C.teal, transparency: 80 }, line: { color: C.teal, transparency: 70 } });
  s.addShape(pres.shapes.OVAL, { x: 7.8, y: 2.5, w: 3.2, h: 3.2, fill: { color: C.tealLight, transparency: 88 }, line: { color: C.tealLight, transparency: 80 } });

  s.addText("4IM06 · G3 · Project 22  |  Weekly Progress Update", {
    x: 0.55, y: 0.35, w: 8, h: 0.35, fontSize: 11, color: C.tealLight, italic: true, margin: 0
  });

  s.addText("This Week's Progress:\nThree New Methods", {
    x: 0.5, y: 0.9, w: 7.5, h: 1.7, fontSize: 34, color: C.white, bold: true, align: "left", valign: "middle", margin: 0
  });

  s.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 2.75, w: 5.5, h: 0.025, fill: { color: C.teal }, line: { color: C.teal } });

  s.addText("June 2026  ·  Supervised by Quentin Bammey", {
    x: 0.55, y: 2.9, w: 7, h: 0.35, fontSize: 12, color: C.gray, margin: 0
  });

  const pills = [
    { label: "① JPEG vs Resample\nClassical Detector", x: 0.5 },
    { label: "② Learnable\nSpectral Mask", x: 3.5 },
    { label: "③ Spectral\nCNN", x: 6.5 },
  ];
  pills.forEach(p => {
    s.addShape(pres.shapes.RECTANGLE, { x: p.x, y: 3.7, w: 2.7, h: 0.85, fill: { color: C.cardBg }, line: { color: C.teal, width: 1 }, shadow: makeShadow() });
    s.addText(p.label, { x: p.x, y: 3.7, w: 2.7, h: 0.85, fontSize: 11, color: C.tealLight, bold: true, align: "center", valign: "middle", margin: 0 });
  });
}

// ══════════════════════════════════════════════════════════════════════════════
// SLIDE 2 — Method A: jpeg_resample_detector.py — Method
// ══════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.navy };

  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 0.08, h: 5.625, fill: { color: C.tealLight }, line: { color: C.tealLight } });

  s.addText("① Classical Detector: JPEG vs Block Resampling", {
    x: 0.35, y: 0.18, w: 9.3, h: 0.52, fontSize: 22, color: C.white, bold: true, margin: 0
  });
  s.addText("jpeg_resample_detector.py  ·  DCT / FFT features  ·  a contrario NFA decision", {
    x: 0.35, y: 0.68, w: 9.3, h: 0.3, fontSize: 11.5, color: C.tealLight, italic: true, margin: 0
  });

  // Goal
  s.addShape(pres.shapes.RECTANGLE, { x: 0.35, y: 1.08, w: 9.3, h: 0.52, fill: { color: C.cardBg }, line: { color: C.teal, width: 1 } });
  s.addText("Goal: distinguish JPEG compression (8×8 DCT artifacts) from block-level ×8 resampling — both create period-8 spectral structures.", {
    x: 0.5, y: 1.08, w: 9.1, h: 0.52, fontSize: 11.5, color: C.ice, valign: "middle", margin: 0
  });

  // 3-script pipeline
  s.addText("3-Script Pipeline", { x: 0.35, y: 1.72, w: 9, h: 0.3, fontSize: 13, color: C.tealLight, bold: true, margin: 0 });

  const scripts = [
    {
      name: "create_forensic_postprocess_dataset.py",
      role: "Build dataset",
      detail: "Generates 4 categories from raw images:\noriginal  ·  jpeg  ·  resample_×8  ·  mix",
      col: "1E3A5F",
    },
    {
      name: "jpeg_resample_detector.py",
      role: "Detect & classify",
      detail: "Residual → DCT block energy + FFT periodicity\n→ NFA test → Label output",
      col: "0E4D70",
    },
    {
      name: "evaluate_detector_on_dataset.py",
      role: "Batch evaluation",
      detail: "Runs detector over full dataset split\n→ accuracy + confusion matrix",
      col: "065A82",
    },
  ];
  scripts.forEach((sc, i) => {
    const x = 0.35 + i * 3.2;
    s.addShape(pres.shapes.RECTANGLE, { x, y: 2.1, w: 3.05, h: 2.2, fill: { color: sc.col }, line: { color: C.tealLight, width: 1 }, shadow: makeShadow() });
    s.addShape(pres.shapes.RECTANGLE, { x, y: 2.1, w: 3.05, h: 0.32, fill: { color: C.tealLight, transparency: 20 }, line: { color: C.tealLight } });
    s.addText(`Step ${i + 1}: ${sc.role}`, { x, y: 2.1, w: 3.05, h: 0.32, fontSize: 10.5, color: C.white, bold: true, align: "center", valign: "middle", margin: 0 });
    s.addText(sc.name, { x: x + 0.1, y: 2.46, w: 2.85, h: 0.38, fontSize: 9, color: C.tealLight, italic: true, margin: 0 });
    s.addText(sc.detail, { x: x + 0.1, y: 2.87, w: 2.85, h: 1.3, fontSize: 10, color: C.ice, valign: "top", margin: 0 });
    if (i < 2) s.addText("→", { x: x + 3.05, y: 2.1, w: 0.15, h: 2.2, fontSize: 18, color: C.teal, align: "center", valign: "middle", margin: 0 });
  });

  // Feature detail
  s.addText("Detector features:", { x: 0.35, y: 4.42, w: 2.5, h: 0.28, fontSize: 11, color: C.tealLight, bold: true, margin: 0 });
  const feats = [
    "DCT block energy at 8×8 boundaries  →  detects JPEG quantization grid",
    "FFT periodicity score  →  detects resampling period",
    "Null distribution estimated from original patches  →  NFA thresholding",
    "Output labels:  jpeg  /  8×8_resampling  /  mixed  /  uncertain",
  ];
  feats.forEach((f, i) => {
    s.addText("·  " + f, { x: 0.45, y: 4.72 + i * 0.23, w: 9.1, h: 0.22, fontSize: 10, color: C.ice, margin: 0 });
  });
}

// ══════════════════════════════════════════════════════════════════════════════
// SLIDE 3 — Method A: Results / Status
// ══════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.navy };

  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 0.08, h: 5.625, fill: { color: C.tealLight }, line: { color: C.tealLight } });

  s.addText("① Classical Detector: Experimental Results", {
    x: 0.35, y: 0.18, w: 9.3, h: 0.52, fontSize: 22, color: C.white, bold: true, margin: 0
  });

  // Two columns: prior NFA work (context) vs new pipeline
  // Left: prior NFA finding (brief context)
  s.addShape(pres.shapes.RECTANGLE, { x: 0.35, y: 0.82, w: 4.2, h: 3.55, fill: { color: C.cardBg }, line: { color: "334155", width: 1 } });
  s.addShape(pres.shapes.RECTANGLE, { x: 0.35, y: 0.82, w: 4.2, h: 0.38, fill: { color: "334155" }, line: { color: "334155" } });
  s.addText("Background: NFA Spectrum Correlation (Prior Work)", { x: 0.45, y: 0.82, w: 4.0, h: 0.38, fontSize: 10, color: C.gray, bold: true, valign: "middle", margin: 0 });

  const nfaRows = [
    { label: "Dataset", val: "RAISE-1K real images" },
    { label: "Method", val: "Spectral patch correlation + NFA" },
    { label: "Significant peak detection", val: "91%  (synthetic)" },
    { label: "Correct size estimation (top-1)", val: "11.1%  (synthetic)" },
    { label: "Real RAISE PNG resampling", val: "0% detectable" },
  ];
  nfaRows.forEach((r, i) => {
    const y = 1.3 + i * 0.42;
    s.addText(r.label, { x: 0.48, y, w: 2.0, h: 0.38, fontSize: 10, color: C.gray, margin: 0 });
    s.addText(r.val, { x: 2.5, y, w: 1.9, h: 0.38, fontSize: 10, color: C.ice, bold: true, margin: 0 });
  });
  s.addShape(pres.shapes.RECTANGLE, { x: 0.35, y: 3.42, w: 4.2, h: 0.52, fill: { color: "1C2A3A" }, line: { color: "334155" } });
  s.addText("Bottleneck: peaks detected but source size not localizable → motivated new feature approach.", {
    x: 0.48, y: 3.42, w: 3.95, h: 0.52, fontSize: 9.5, color: C.gray, italic: true, valign: "middle", margin: 0
  });

  // Right: new pipeline results
  s.addShape(pres.shapes.RECTANGLE, { x: 4.75, y: 0.82, w: 4.9, h: 3.55, fill: { color: C.cardBg }, line: { color: C.teal, width: 1.5 }, shadow: makeShadow() });
  s.addShape(pres.shapes.RECTANGLE, { x: 4.75, y: 0.82, w: 4.9, h: 0.38, fill: { color: C.teal }, line: { color: C.teal } });
  s.addText("New This Week: DCT + FFT Feature Pipeline", { x: 4.85, y: 0.82, w: 4.7, h: 0.38, fontSize: 10.5, color: C.white, bold: true, valign: "middle", margin: 0 });

  s.addText("Dataset generated by create_forensic_postprocess_dataset.py:", {
    x: 4.88, y: 1.3, w: 4.6, h: 0.28, fontSize: 10, color: C.tealLight, bold: true, margin: 0
  });

  const cats = [
    { label: "original", col: C.good },
    { label: "jpeg (Q80)", col: C.warn },
    { label: "resample_×8", col: C.bad },
    { label: "mix", col: "8B5CF6" },
  ];
  cats.forEach((c, i) => {
    const x = 4.88 + i * 1.18;
    s.addShape(pres.shapes.RECTANGLE, { x, y: 1.63, w: 1.1, h: 0.36, fill: { color: c.col, transparency: 20 }, line: { color: c.col, width: 1 } });
    s.addText(c.label, { x, y: 1.63, w: 1.1, h: 0.36, fontSize: 9, color: C.white, bold: true, align: "center", valign: "middle", margin: 0 });
  });

  s.addText("Detector outputs:", { x: 4.88, y: 2.1, w: 4.6, h: 0.28, fontSize: 10, color: C.tealLight, bold: true, margin: 0 });
  const outs = [
    { label: "jpeg", desc: "JPEG block artifact dominant" },
    { label: "8×8_resampling", desc: "Geometric ×8 period dominant" },
    { label: "mixed", desc: "Both signals present" },
    { label: "uncertain", desc: "Neither signal significant" },
  ];
  outs.forEach((o, i) => {
    const y = 2.42 + i * 0.36;
    s.addShape(pres.shapes.RECTANGLE, { x: 4.88, y: y + 0.03, w: 1.2, h: 0.28, fill: { color: C.teal, transparency: 30 }, line: { color: C.teal } });
    s.addText(o.label, { x: 4.88, y: y + 0.03, w: 1.2, h: 0.28, fontSize: 9.5, color: C.white, bold: true, align: "center", valign: "middle", margin: 0 });
    s.addText(o.desc, { x: 6.15, y, w: 3.4, h: 0.34, fontSize: 10, color: C.ice, valign: "middle", margin: 0 });
  });

  s.addShape(pres.shapes.RECTANGLE, { x: 4.75, y: 3.88, w: 4.9, h: 0.42, fill: { color: "1C3D5A" }, line: { color: C.tealLight } });
  s.addText("Status: pipeline complete & integrated. Large-scale quantitative results pending (run evaluate_detector_on_dataset.py).", {
    x: 4.85, y: 3.88, w: 4.7, h: 0.42, fontSize: 9.5, color: C.ice, italic: true, valign: "middle", margin: 0
  });

  // Takeaway
  s.addShape(pres.shapes.RECTANGLE, { x: 0.35, y: 4.5, w: 9.3, h: 0.75, fill: { color: "111D2B" }, line: { color: C.teal, width: 1 } });
  s.addText("Key improvement over prior NFA approach: instead of estimating resampling distance d from spectral peaks (unreliable), directly classify the dominant artifact type (JPEG block vs. geometric period) using separate DCT and FFT features.", {
    x: 0.5, y: 4.5, w: 9.1, h: 0.75, fontSize: 10.5, color: C.ice, italic: true, valign: "middle", margin: 0
  });
}

// ══════════════════════════════════════════════════════════════════════════════
// SLIDE 4 — Method B: Spectral Mask — Method
// ══════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.lightBg };

  s.addText("② Learnable Spectral Mask", {
    x: 0.4, y: 0.18, w: 9.2, h: 0.52, fontSize: 24, color: C.textDark, bold: true, margin: 0
  });
  s.addText("spectral-mask-resampling  ·  4 classes  ·  5 observed patch sizes (32 – 128 px)", {
    x: 0.4, y: 0.68, w: 9, h: 0.28, fontSize: 11.5, color: C.teal, italic: true, margin: 0
  });
  s.addShape(pres.shapes.RECTANGLE, { x: 0.4, y: 0.98, w: 9.2, h: 0.025, fill: { color: C.teal }, line: { color: C.teal } });

  s.addShape(pres.shapes.RECTANGLE, { x: 0.4, y: 1.08, w: 9.2, h: 0.48, fill: { color: "EFF6FF" }, line: { color: C.teal, width: 1 } });
  s.addText("Core idea: patches of different sizes have different native FFT dimensions. Remap all log-spectra to a unified 512×257 normalized frequency grid (cycles/pixel), then learn one mask per class.", {
    x: 0.5, y: 1.08, w: 9.0, h: 0.48, fontSize: 11, color: C.textDark, valign: "middle", margin: 0
  });

  // Pipeline
  const steps = [
    { t: "Crop patch\n(size o)", sub: "o ∈ {32,48,64,96,128}" },
    { t: "Y channel\n+ TV residual", sub: "suppress content" },
    { t: "rFFT\nlog|F|", sub: "raw spectrum" },
    { t: "Remap to\n512×257 grid", sub: "cycles/pixel" },
    { t: "Mask Mₖ\n(sigmoid)", sub: "per-class learnable" },
    { t: "Cosine sim.\n→ 4-class", sub: "vs reference Rₖ" },
  ];
  steps.forEach((st, i) => {
    const x = 0.3 + i * 1.6;
    s.addShape(pres.shapes.RECTANGLE, { x, y: 1.68, w: 1.45, h: 0.9, fill: { color: "EFF6FF" }, line: { color: C.teal, width: 1.5 } });
    s.addText(st.t, { x, y: 1.68, w: 1.45, h: 0.58, fontSize: 9.5, color: C.textDark, bold: true, align: "center", valign: "middle", margin: 0 });
    s.addText(st.sub, { x, y: 2.26, w: 1.45, h: 0.32, fontSize: 8.5, color: C.teal, align: "center", margin: 0 });
    if (i < 5) s.addText("→", { x: x + 1.45, y: 1.68, w: 0.15, h: 0.9, fontSize: 14, color: C.teal, align: "center", valign: "middle", margin: 0 });
  });

  // 4 class generation
  s.addText("How each class is generated (for every observed size o):", {
    x: 0.4, y: 2.72, w: 9, h: 0.3, fontSize: 12, color: C.textDark, bold: true, margin: 0
  });
  const tData = [
    [
      { text: "Class", options: { bold: true, fill: { color: C.teal }, color: C.white } },
      { text: "Source crop", options: { bold: true, fill: { color: C.teal }, color: C.white } },
      { text: "Processing applied", options: { bold: true, fill: { color: C.teal }, color: C.white } },
      { text: "Spectral artifact", options: { bold: true, fill: { color: C.teal }, color: C.white } },
    ],
    ["original", "o × o", "none", "natural image spectrum"],
    ["JPEG Q80", "o × o", "JPEG compress Q=80", "8×8 block quantization grid"],
    ["downsample ×8", "8o × 8o", "bicubic resize → o", "period-8 resampling peaks"],
    ["downsample ×16", "16o × 16o", "bicubic resize → o", "period-16 resampling peaks"],
  ];
  s.addTable(tData, {
    x: 0.4, y: 3.05, w: 9.2, h: 1.72,
    border: { pt: 0.5, color: "CBD5E1" },
    fill: { color: C.white },
    fontSize: 11,
    rowH: 0.34,
  });

  s.addShape(pres.shapes.RECTANGLE, { x: 0.4, y: 4.88, w: 9.2, h: 0.38, fill: { color: "EFF6FF" }, line: { color: C.teal, width: 1 } });
  s.addText("Training: 700 RAISE source images  ·  AdamW lr=1e-3  ·  30 epochs  ·  Test: 20,000 samples (4 classes × 5 sizes × 1,000)", {
    x: 0.5, y: 4.88, w: 9.0, h: 0.38, fontSize: 10.5, color: C.textMid, valign: "middle", margin: 0
  });
}

// ══════════════════════════════════════════════════════════════════════════════
// SLIDE 5 — Method B: Results
// ══════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.navy };

  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 0.08, h: 5.625, fill: { color: C.tealLight }, line: { color: C.tealLight } });

  s.addText("② Spectral Mask: Results", {
    x: 0.35, y: 0.18, w: 9.3, h: 0.52, fontSize: 22, color: C.white, bold: true, margin: 0
  });

  // Big stats
  s.addShape(pres.shapes.RECTANGLE, { x: 0.35, y: 0.82, w: 2.0, h: 1.25, fill: { color: C.cardBg }, line: { color: C.warn, width: 2 }, shadow: makeShadow() });
  s.addText("56.6%", { x: 0.35, y: 0.87, w: 2.0, h: 0.72, fontSize: 36, color: C.warn, bold: true, align: "center", margin: 0 });
  s.addText("Test Accuracy\n(random = 25%)", { x: 0.35, y: 1.59, w: 2.0, h: 0.44, fontSize: 9.5, color: C.gray, align: "center", margin: 0 });

  s.addShape(pres.shapes.RECTANGLE, { x: 2.55, y: 0.82, w: 1.8, h: 1.25, fill: { color: C.cardBg }, line: { color: C.teal, width: 1 }, shadow: makeShadow() });
  s.addText("0.561", { x: 2.55, y: 0.87, w: 1.8, h: 0.72, fontSize: 36, color: C.tealLight, bold: true, align: "center", margin: 0 });
  s.addText("Macro F1", { x: 2.55, y: 1.59, w: 1.8, h: 0.44, fontSize: 9.5, color: C.gray, align: "center", margin: 0 });

  s.addShape(pres.shapes.RECTANGLE, { x: 4.55, y: 0.82, w: 1.8, h: 1.25, fill: { color: C.cardBg }, line: { color: C.teal, width: 1 }, shadow: makeShadow() });
  s.addText("20,000", { x: 4.55, y: 0.87, w: 1.8, h: 0.72, fontSize: 28, color: C.white, bold: true, align: "center", margin: 0 });
  s.addText("Test samples", { x: 4.55, y: 1.59, w: 1.8, h: 0.44, fontSize: 9.5, color: C.gray, align: "center", margin: 0 });

  // Per-class F1 chart
  s.addChart(pres.charts.BAR, [{
    name: "F1 Score",
    labels: ["original", "JPEG Q80", "downsample ×8", "downsample ×16"],
    values: [0.59, 0.69, 0.45, 0.51],
  }], {
    x: 6.5, y: 0.75, w: 3.3, h: 1.45,
    barDir: "col",
    chartColors: [C.good, C.good, C.bad, C.warn],
    chartArea: { fill: { color: C.cardBg } },
    catAxisLabelColor: C.gray,
    valAxisLabelColor: C.gray,
    valGridLine: { color: "334155", size: 0.5 },
    catGridLine: { style: "none" },
    showValue: true,
    dataLabelColor: C.white,
    dataLabelFontSize: 9,
    showLegend: false,
    showTitle: true,
    title: "Per-class F1",
    titleColor: C.ice,
    titleFontSize: 10,
    valAxisMaxVal: 1.0,
  });

  // Accuracy by size chart
  s.addChart(pres.charts.LINE, [{
    name: "Accuracy",
    labels: ["128 px", "96 px", "64 px", "48 px", "32 px"],
    values: [0.633, 0.609, 0.569, 0.542, 0.476],
  }], {
    x: 0.35, y: 2.18, w: 4.5, h: 2.05,
    chartColors: [C.tealLight],
    chartArea: { fill: { color: C.cardBg } },
    catAxisLabelColor: C.gray,
    valAxisLabelColor: C.gray,
    valGridLine: { color: "334155", size: 0.5 },
    catGridLine: { style: "none" },
    lineSize: 3,
    showValue: true,
    dataLabelColor: C.white,
    dataLabelFontSize: 9,
    showLegend: false,
    showTitle: true,
    title: "Accuracy vs. observed patch size",
    titleColor: C.ice,
    titleFontSize: 10,
    valAxisMaxVal: 0.80,
    valAxisMinVal: 0.35,
  });

  // Confusion pairs
  s.addShape(pres.shapes.RECTANGLE, { x: 5.1, y: 2.18, w: 4.7, h: 2.05, fill: { color: C.cardBg }, line: { color: C.teal, width: 1 } });
  s.addText("Confusion pairs  (test set: 5,000 per class)", { x: 5.2, y: 2.22, w: 4.5, h: 0.32, fontSize: 11, color: C.ice, bold: true, margin: 0 });

  const confRows = [
    { pair: "×8  →  ×16", count: "1,902 / 5,000", sev: "SEVERE", col: C.bad },
    { pair: "×16  →  ×8", count: "1,762 / 5,000", sev: "SEVERE", col: C.bad },
    { pair: "original  →  JPEG", count: "1,515 / 5,000", sev: "HIGH", col: C.warn },
    { pair: "JPEG  →  original", count: "833 / 5,000", sev: "MED", col: C.warn },
    { pair: "JPEG  →  ×8", count: "298 / 5,000", sev: "LOW", col: C.good },
  ];
  confRows.forEach((r, i) => {
    const y = 2.62 + i * 0.31;
    s.addText(r.pair, { x: 5.2, y, w: 1.85, h: 0.28, fontSize: 10, color: C.ice, margin: 0 });
    s.addText(r.count, { x: 7.08, y, w: 1.45, h: 0.28, fontSize: 10, color: C.gray, align: "right", margin: 0 });
    s.addShape(pres.shapes.RECTANGLE, { x: 8.65, y: y + 0.03, w: 1.0, h: 0.22, fill: { color: r.col, transparency: 15 }, line: { color: r.col } });
    s.addText(r.sev, { x: 8.65, y: y + 0.03, w: 1.0, h: 0.22, fontSize: 8.5, color: C.white, bold: true, align: "center", valign: "middle", margin: 0 });
  });

  // Interpretability finding
  s.addShape(pres.shapes.RECTANGLE, { x: 0.35, y: 4.35, w: 9.3, h: 0.62, fill: { color: "1C2A3A" }, line: { color: C.bad, width: 1.5 } });
  s.addText("Interpretability: learned mask pairwise overlap = 0.936 — the four class masks are nearly identical, confirming the model cannot find class-specific frequency bands in the log-spectrum alone.", {
    x: 0.5, y: 4.35, w: 9.1, h: 0.62, fontSize: 10.5, color: C.ice, italic: true, valign: "middle", margin: 0
  });
}

// ══════════════════════════════════════════════════════════════════════════════
// SLIDE 6 — Method C: CNN — Method
// ══════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.lightBg };

  s.addText("③ Spectral Positional CNN", {
    x: 0.4, y: 0.18, w: 9.2, h: 0.52, fontSize: 24, color: C.textDark, bold: true, margin: 0
  });
  s.addText("spectral-history-cnn  ·  6 classes  ·  fixed 64×64 observation", {
    x: 0.4, y: 0.68, w: 9, h: 0.28, fontSize: 11.5, color: C.teal, italic: true, margin: 0
  });
  s.addShape(pres.shapes.RECTANGLE, { x: 0.4, y: 0.98, w: 9.2, h: 0.025, fill: { color: C.teal }, line: { color: C.teal } });

  // Difference from Mask
  s.addShape(pres.shapes.RECTANGLE, { x: 0.4, y: 1.08, w: 9.2, h: 0.48, fill: { color: "ECFDF5" }, line: { color: C.good, width: 1 } });
  s.addText("Design difference vs. Mask: single fixed size 64×64 (no multi-size), native 64×33 spectrum (no 512×257 remapping), positional encoding added to let the model learn frequency-location jointly.", {
    x: 0.5, y: 1.08, w: 9.0, h: 0.48, fontSize: 11, color: C.textDark, valign: "middle", margin: 0
  });

  // Pipeline
  const steps = [
    { t: "Random crop\n64×64", sub: "source size varies" },
    { t: "Y channel\n+ TV residual", sub: "suppress content" },
    { t: "rFFT\nlog|F| → 1×64×33", sub: "native spectrum" },
    { t: "+ 43-channel\nposition encoding", sub: "λ=1,2,4,8,16,32" },
    { t: "CNN\n44ch input", sub: "3×ConvBlock + GAP" },
    { t: "Linear\n→ 6 classes", sub: "softmax" },
  ];
  steps.forEach((st, i) => {
    const x = 0.3 + i * 1.6;
    s.addShape(pres.shapes.RECTANGLE, { x, y: 1.68, w: 1.45, h: 0.9, fill: { color: "EFF6FF" }, line: { color: C.teal, width: 1.5 } });
    s.addText(st.t, { x, y: 1.68, w: 1.45, h: 0.58, fontSize: 9.5, color: C.textDark, bold: true, align: "center", valign: "middle", margin: 0 });
    s.addText(st.sub, { x, y: 2.26, w: 1.45, h: 0.32, fontSize: 8.5, color: C.teal, align: "center", margin: 0 });
    if (i < 5) s.addText("→", { x: x + 1.45, y: 1.68, w: 0.15, h: 0.9, fontSize: 14, color: C.teal, align: "center", valign: "middle", margin: 0 });
  });

  // 6 classes
  s.addText("6 classes (each observed as a 64×64 patch):", {
    x: 0.4, y: 2.72, w: 9, h: 0.3, fontSize: 12, color: C.textDark, bold: true, margin: 0
  });
  const cls6 = [
    { label: "original", sub: "direct crop 64×64" },
    { label: "JPEG Q80", sub: "crop + JPEG" },
    { label: "downsample ×2", sub: "crop 128 → 64" },
    { label: "downsample ×4", sub: "crop 256 → 64" },
    { label: "downsample ×8", sub: "crop 512 → 64" },
    { label: "downsample ×16", sub: "crop 1024 → 64" },
  ];
  cls6.forEach((c, i) => {
    const x = 0.3 + i * 1.6;
    const col = i < 2 ? C.good : (i < 4 ? C.warn : C.bad);
    s.addShape(pres.shapes.RECTANGLE, { x, y: 3.08, w: 1.45, h: 0.72, fill: { color: C.white }, line: { color: col, width: 1.5 } });
    s.addShape(pres.shapes.RECTANGLE, { x, y: 3.08, w: 1.45, h: 0.24, fill: { color: col }, line: { color: col } });
    s.addText(c.label, { x, y: 3.08, w: 1.45, h: 0.24, fontSize: 9, color: C.white, bold: true, align: "center", valign: "middle", margin: 0 });
    s.addText(c.sub, { x, y: 3.32, w: 1.45, h: 0.48, fontSize: 9, color: C.textMid, align: "center", valign: "middle", margin: 0 });
  });

  // Config + design
  s.addShape(pres.shapes.RECTANGLE, { x: 0.4, y: 3.95, w: 4.5, h: 1.25, fill: { color: "F8FAFC" }, line: { color: "CBD5E1", width: 1 } });
  s.addText("Training", { x: 0.5, y: 3.98, w: 4.3, h: 0.28, fontSize: 11, color: C.textDark, bold: true, margin: 0 });
  ["AdamW lr=3e-4  ·  batch=256  ·  50 epochs", "AMP (mixed precision)  ·  GPU (HPC)", "Test set: 9,000 samples  (6 × 1,500)"].forEach((t, i) => {
    s.addText(t, { x: 0.5, y: 4.3 + i * 0.28, w: 4.3, h: 0.26, fontSize: 10, color: C.textMid, margin: 0 });
  });

  s.addShape(pres.shapes.RECTANGLE, { x: 5.1, y: 3.95, w: 4.6, h: 1.25, fill: { color: "EFF6FF" }, line: { color: C.teal, width: 1 } });
  s.addText("Why positional encoding?", { x: 5.2, y: 3.98, w: 4.4, h: 0.28, fontSize: 11, color: C.teal, bold: true, margin: 0 });
  ["Sinusoidal encoding at λ=1,2,4,8,16,32 gives the CNN explicit access to frequency bin location — critical for detecting period-8 and period-16 structures."].forEach((t, i) => {
    s.addText(t, { x: 5.2, y: 4.3 + i * 0.28, w: 4.4, h: 0.8, fontSize: 10, color: C.textMid, margin: 0 });
  });
}

// ══════════════════════════════════════════════════════════════════════════════
// SLIDE 7 — Method C: CNN Results
// ══════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.navy };

  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 0.08, h: 5.625, fill: { color: C.tealLight }, line: { color: C.tealLight } });

  s.addText("③ Spectral CNN: Results", {
    x: 0.35, y: 0.18, w: 9.3, h: 0.52, fontSize: 22, color: C.white, bold: true, margin: 0
  });

  const bigStats = [
    { val: "62.5%", label: "Test Accuracy", col: C.good },
    { val: "65.2%", label: "Best val (epoch 5)", col: C.tealLight },
    { val: "99.1%", label: "Train acc — overfit", col: C.bad },
  ];
  bigStats.forEach((st, i) => {
    const x = 0.35 + i * 3.15;
    s.addShape(pres.shapes.RECTANGLE, { x, y: 0.82, w: 2.9, h: 1.18, fill: { color: C.cardBg }, line: { color: st.col, width: 2 }, shadow: makeShadow() });
    s.addText(st.val, { x, y: 0.87, w: 2.9, h: 0.72, fontSize: 34, color: st.col, bold: true, align: "center", margin: 0 });
    s.addText(st.label, { x, y: 1.59, w: 2.9, h: 0.38, fontSize: 9.5, color: C.gray, align: "center", margin: 0 });
  });

  s.addChart(pres.charts.BAR, [{
    name: "F1",
    labels: ["original", "JPEG", "×2", "×4", "×8", "×16"],
    values: [0.91, 0.91, 0.73, 0.41, 0.23, 0.48],
  }], {
    x: 0.35, y: 2.1, w: 5.3, h: 2.2,
    barDir: "col",
    chartColors: [C.good, C.good, C.warn, C.warn, C.bad, C.warn],
    chartArea: { fill: { color: C.cardBg } },
    catAxisLabelColor: C.gray,
    valAxisLabelColor: C.gray,
    valGridLine: { color: "334155", size: 0.5 },
    catGridLine: { style: "none" },
    showValue: true,
    dataLabelColor: C.white,
    dataLabelFontSize: 9,
    showLegend: false,
    showTitle: true,
    title: "Per-class F1 Score (6 classes)",
    titleColor: C.ice,
    titleFontSize: 11,
    valAxisMaxVal: 1.0,
  });

  s.addShape(pres.shapes.RECTANGLE, { x: 5.9, y: 2.1, w: 3.8, h: 2.2, fill: { color: C.cardBg }, line: { color: C.teal, width: 1 } });
  s.addText("Confusion pairs  (N=1,500 per class)", { x: 6.0, y: 2.15, w: 3.6, h: 0.32, fontSize: 11, color: C.ice, bold: true, margin: 0 });

  const cnnConf = [
    { pair: "×8  →  ×16", count: "680", sev: "SEVERE", col: C.bad },
    { pair: "×4 ↔ ×8  (both dir.)", count: "553", sev: "HIGH", col: C.bad },
    { pair: "×16  →  ×8", count: "301", sev: "HIGH", col: C.warn },
    { pair: "original ↔ JPEG", count: "106", sev: "LOW", col: C.good },
    { pair: "JPEG  →  ×8", count: "15", sev: "MINIMAL", col: C.good },
  ];
  cnnConf.forEach((r, i) => {
    const y = 2.55 + i * 0.34;
    s.addText(r.pair, { x: 6.0, y, w: 1.7, h: 0.3, fontSize: 10, color: C.ice, margin: 0 });
    s.addText(r.count, { x: 7.75, y, w: 0.7, h: 0.3, fontSize: 10, color: C.gray, align: "right", margin: 0 });
    s.addShape(pres.shapes.RECTANGLE, { x: 8.55, y: y + 0.03, w: 1.0, h: 0.22, fill: { color: r.col, transparency: 15 }, line: { color: r.col } });
    s.addText(r.sev, { x: 8.55, y: y + 0.03, w: 1.0, h: 0.22, fontSize: 8, color: C.white, bold: true, align: "center", valign: "middle", margin: 0 });
  });

  // 4-class ablation note
  s.addShape(pres.shapes.RECTANGLE, { x: 0.35, y: 4.42, w: 9.3, h: 0.88, fill: { color: "1C2A3A" }, line: { color: C.tealLight, width: 1 } });
  s.addText("Post-hoc 4-class analysis (extract original/JPEG/×8/×16 from 6-class predictions, no retraining)", { x: 0.5, y: 4.44, w: 9.0, h: 0.3, fontSize: 11, color: C.tealLight, bold: true, margin: 0 });
  s.addText("Subset acc: 76.1% (inflated — ×2/×4 confusion removed)  ·  ×8 vs ×16 binary acc: 53.9%  ←  near random  ·  ×8→×4 bridge: 351 cases", {
    x: 0.5, y: 4.76, w: 9.0, h: 0.5, fontSize: 10.5, color: C.ice, margin: 0
  });
}

// ══════════════════════════════════════════════════════════════════════════════
// SLIDE 8 — Comparison & Next Steps
// ══════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.lightBg };

  s.addText("Comparison & Next Steps", {
    x: 0.4, y: 0.18, w: 9.2, h: 0.52, fontSize: 24, color: C.textDark, bold: true, margin: 0
  });
  s.addShape(pres.shapes.RECTANGLE, { x: 0.4, y: 0.72, w: 9.2, h: 0.025, fill: { color: C.teal }, line: { color: C.teal } });

  const tData = [
    [
      { text: "", options: { fill: { color: C.teal }, color: C.white } },
      { text: "① Classical Detector", options: { bold: true, fill: { color: C.teal }, color: C.white } },
      { text: "② Spectral Mask", options: { bold: true, fill: { color: C.teal }, color: C.white } },
      { text: "③ CNN", options: { bold: true, fill: { color: C.teal }, color: C.white } },
    ],
    ["Learns parameters?", "No — hand-crafted features", "Yes — mask + reference", "Yes — full CNN"],
    ["Task", "jpeg vs. ×8 (binary)", "4 classes, 5 sizes", "6 classes, fixed 64×64"],
    ["Overall accuracy", "Pending eval run", "56.6%", "62.5%"],
    ["original / JPEG", "—", "F1 = 0.59 / 0.69", "F1 = 0.91 / 0.91  ✓"],
    ["×8 / ×16 separation", "Spectrally ambiguous", "F1 = 0.45 / 0.51", "F1 = 0.23 / 0.48"],
    ["Key bottleneck", "Feature localisation", "×8 ↔ ×16 overlap", "Overfitting + ×8 recall"],
  ];
  s.addTable(tData, {
    x: 0.4, y: 0.82, w: 9.2, h: 2.52,
    border: { pt: 0.5, color: "CBD5E1" },
    fill: { color: C.white },
    fontSize: 10.5,
    rowH: 0.36,
  });

  s.addText("Pending Results for Final Defense", { x: 0.4, y: 3.45, w: 9.2, h: 0.34, fontSize: 15, color: C.textDark, bold: true, margin: 0 });

  const nexts = [
    { icon: "A", text: "Route A2 — Classical Detector: run evaluate_detector_on_dataset.py on held-out test split to obtain quantitative accuracy and confusion matrix.", col: C.teal },
    { icon: "C", text: "CNN 4-class retrain (original / JPEG / ×8 / ×16, matching Mask setup) — enables fair apples-to-apples comparison across all three routes.", col: C.teal },
  ];
  nexts.forEach((n, i) => {
    const y = 3.9 + i * 0.62;
    s.addShape(pres.shapes.RECTANGLE, { x: 0.4, y: y + 0.04, w: 0.3, h: 0.28, fill: { color: n.col, transparency: 15 }, line: { color: n.col } });
    s.addText(n.icon, { x: 0.4, y: y + 0.04, w: 0.3, h: 0.28, fontSize: 10, color: C.white, bold: true, align: "center", valign: "middle", margin: 0 });
    s.addText(n.text, { x: 0.8, y, w: 8.8, h: 0.55, fontSize: 11, color: C.textMid, valign: "middle", margin: 0 });
  });
}

// ══════════════════════════════════════════════════════════════════════════════
// SLIDE 9 — Report Narrative Structure
// ══════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.navyDark };

  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.06, fill: { color: C.tealLight }, line: { color: C.tealLight } });
  s.addShape(pres.shapes.OVAL, { x: 7.5, y: 2.5, w: 4.0, h: 4.0, fill: { color: C.teal, transparency: 88 }, line: { color: C.teal, transparency: 80 } });

  s.addText("Report Narrative Structure", {
    x: 0.45, y: 0.18, w: 9.1, h: 0.52, fontSize: 24, color: C.white, bold: true, margin: 0
  });
  s.addText("How we tell the story in the written report", {
    x: 0.45, y: 0.68, w: 9.1, h: 0.28, fontSize: 12, color: C.tealLight, italic: true, margin: 0
  });

  // Central narrative flow — 5 arrow boxes
  const flow = [
    { label: "Prior Method", sub: "NFA spectral\ncorrelation (paper)", col: C.gray },
    { label: "Reproduce\n& Test", sub: "resampling_core.py\nW3 + RAISE100", col: C.teal },
    { label: "Expose\nLimitations", sub: "NFA fails on real data\nFourier ambiguity", col: C.bad },
    { label: "Our 3 Routes", sub: "A2 DCT  ·  Mask\nCNN", col: C.good },
    { label: "Compare\n& Conclude", sub: "Quantify limits\nof Fourier-only", col: "8B5CF6" },
  ];
  flow.forEach((f, i) => {
    const x = 0.3 + i * 1.9;
    s.addShape(pres.shapes.RECTANGLE, { x, y: 1.1, w: 1.7, h: 1.05, fill: { color: f.col, transparency: i === 0 ? 60 : 20 }, line: { color: f.col, width: 1.5 } });
    s.addText(f.label, { x, y: 1.1, w: 1.7, h: 0.52, fontSize: 10.5, color: C.white, bold: true, align: "center", valign: "middle", margin: 0 });
    s.addText(f.sub, { x, y: 1.62, w: 1.7, h: 0.5, fontSize: 8.5, color: C.ice, align: "center", valign: "top", margin: 0 });
    if (i < 4) s.addText("→", { x: x + 1.7, y: 1.1, w: 0.2, h: 1.05, fontSize: 16, color: C.tealLight, align: "center", valign: "middle", margin: 0 });
  });

  // Key insight per chapter
  s.addText("Chapter-by-chapter logic:", {
    x: 0.45, y: 2.32, w: 9.1, h: 0.3, fontSize: 12.5, color: C.tealLight, bold: true, margin: 0
  });

  const chapters = [
    {
      ch: "§2  Prior method & reproduction",
      body: "Establish the baseline. Two reproduction experiments show that NFA detects peaks (91% on synthetic) but cannot localise them (top-1: 11%), and completely fails on real RAISE images (0%). This is not a failure to reproduce — it is the finding that motivates everything that follows.",
      col: C.teal,
    },
    {
      ch: "§3  Task redefinition",
      body: "Reframe from binary (resampled / not) to 4-class (original / JPEG / ×8 / ×16). Introduces the Fourier ambiguity hierarchy — the scientific problem all three routes attack.",
      col: C.teal,
    },
    {
      ch: "§4–6  Our three routes",
      body: "A2 (DCT+FFT) brings non-Fourier evidence for JPEG.  Mask quantifies the ceiling of Fourier-only representations (mask overlap 0.936).  CNN shows that original/JPEG can be solved (F1=0.91) while ×8/×16 remains the shared bottleneck.",
      col: C.good,
    },
    {
      ch: "§7  Cross-route comparison",
      body: "The three routes are complementary, not competing. Together they systematically answer: what can Fourier-only do, where does it break, and what does adding DCT evidence change?",
      col: "8B5CF6",
    },
  ];
  chapters.forEach((c, i) => {
    const y = 2.7 + i * 0.7;
    s.addShape(pres.shapes.RECTANGLE, { x: 0.4, y, w: 0.12, h: 0.55, fill: { color: c.col }, line: { color: c.col } });
    s.addText(c.ch, { x: 0.62, y, w: 2.5, h: 0.28, fontSize: 10, color: c.col, bold: true, margin: 0 });
    s.addText(c.body, { x: 0.62, y: y + 0.27, w: 9.0, h: 0.42, fontSize: 9.5, color: C.ice, margin: 0 });
  });
}

// ══════════════════════════════════════════════════════════════════════════════
// SLIDE 10 — Conclusions & Contributions
// ══════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.navy };

  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 0.08, h: 5.625, fill: { color: C.tealLight }, line: { color: C.tealLight } });

  s.addText("Conclusions & Contributions", {
    x: 0.35, y: 0.18, w: 9.3, h: 0.52, fontSize: 22, color: C.white, bold: true, margin: 0
  });
  s.addText("What we found — and why it matters", {
    x: 0.35, y: 0.68, w: 9.3, h: 0.28, fontSize: 12, color: C.tealLight, italic: true, margin: 0
  });

  // ── Section 8.1: What we found ───────────────────────────────────────────
  s.addText("What we found", {
    x: 0.35, y: 1.08, w: 9.3, h: 0.32, fontSize: 14, color: C.tealLight, bold: true, margin: 0
  });

  const findings = [
    {
      icon: "①",
      text: "Fourier ambiguity is real — Spectral Mask overlap = 0.936; ×8 ↔ ×16 confusion: 3,664 / 10,000 cases; CNN post-hoc binary accuracy: 53.9% (near random). No log-spectrum method alone can reliably separate ×8 from ×16.",
      col: C.bad,
    },
    {
      icon: "②",
      text: "CNN breaks the original / JPEG barrier — F1 = 0.91 for both classes, far above Spectral Mask (0.59 / 0.69). TV residual + positional encoding + convolution effectively captures JPEG 8×8 quantisation artifacts.",
      col: C.good,
    },
    {
      icon: "③",
      text: "NFA limits precisely characterised — real RAISE: 0% detectable; synthetic: 91% peak detection but only 11% top-1 size accuracy. Bottleneck is localisation, not sensitivity.",
      col: C.warn,
    },
    {
      icon: "④",
      text: "DCT+FFT pipeline (Route A2) provides non-Fourier evidence — DCT block energy and FFT periodicity are treated as independent channels, offering a complementary view beyond pure log-spectrum methods.",
      col: C.teal,
    },
  ];

  findings.forEach((f, i) => {
    const y = 1.48 + i * 0.82;
    s.addShape(pres.shapes.RECTANGLE, { x: 0.35, y, w: 9.3, h: 0.74, fill: { color: C.cardBg }, line: { color: f.col, width: 1.5 } });
    s.addShape(pres.shapes.RECTANGLE, { x: 0.35, y, w: 0.36, h: 0.74, fill: { color: f.col, transparency: 20 }, line: { color: f.col } });
    s.addText(f.icon, { x: 0.35, y, w: 0.36, h: 0.74, fontSize: 13, color: C.white, bold: true, align: "center", valign: "middle", margin: 0 });
    s.addText(f.text, { x: 0.8, y: y + 0.08, w: 8.75, h: 0.6, fontSize: 10, color: C.ice, valign: "top", margin: 0 });
  });

  // ── Section 8.2: Outlook ─────────────────────────────────────────────────
  s.addText("Outlook", {
    x: 0.35, y: 4.82, w: 1.5, h: 0.28, fontSize: 14, color: C.tealLight, bold: true, margin: 0
  });
  const outlook = [
    "4-class CNN retrain (original / JPEG / ×8 / ×16) — fair comparison with Spectral Mask on identical setup",
    "Route A2 quantitative evaluation — run batch evaluation on held-out test split",
    "Hybrid Fourier + DCT model — combine log-spectrum with DCT block evidence to break the ×8 / ×16 ambiguity",
  ];
  outlook.forEach((t, i) => {
    s.addShape(pres.shapes.RECTANGLE, { x: 1.95 + i * 2.7, y: 4.78, w: 0.28, h: 0.28, fill: { color: C.teal, transparency: 20 }, line: { color: C.teal } });
    s.addText(`${i + 1}`, { x: 1.95 + i * 2.7, y: 4.78, w: 0.28, h: 0.28, fontSize: 10, color: C.white, bold: true, align: "center", valign: "middle", margin: 0 });
    s.addText(t, { x: 2.28 + i * 2.7, y: 4.72, w: 2.35, h: 0.42, fontSize: 9.5, color: C.gray, valign: "middle", margin: 0 });
  });
}

// ══════════════════════════════════════════════════════════════════════════════
const outPath = "/Users/suzy/Desktop/Repository/Telecom/IM06/projet/ird-main/4IM06-G3-Project22/G3_Project22_Weekly_Slides.pptx";
pres.writeFile({ fileName: outPath }).then(() => console.log("Done:", outPath)).catch(e => { console.error(e); process.exit(1); });

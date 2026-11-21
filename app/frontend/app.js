"use strict";

/* All requests use relative URLs, so the site works on localhost, Docker,
   Hugging Face Spaces, ngrok - any host. Server values are always inserted with
   textContent / element properties (never innerHTML) to prevent XSS. */

const $ = (id) => document.getElementById(id);
const GRADE_COLORS = ["var(--g0)", "var(--g1)", "var(--g2)", "var(--g3)", "var(--g4)"];

const state = { config: null, file: null, result: null, refClass: null };

/* ------------------------------------------------------------------ */
/* Explanation texts (match what the code actually computes)          */
/* ------------------------------------------------------------------ */
const INFO = {
  original: {
    title: "Input retina",
    body: [
      "The uploaded image after cropping the black border. This is the image all overlays are drawn on.",
      "The model itself sees a Ben-Graham enhanced 300 × 300 version, prepared exactly like the training images.",
    ],
  },
  gradcam: {
    title: "Grad-CAM attention",
    body: [
      "Gradient-weighted Class Activation Mapping highlights the regions that pushed the model towards the predicted grade.",
    ],
    legend: [["#ef4444", "Red: strongest influence"], ["#facc15", "Yellow / green: moderate"], ["#3b82f6", "Blue: little or none"]],
    tip: "Trust increases when hot spots sit on haemorrhages, exudates or new vessels, not on the image border or optic disc reflections.",
  },
  evidence: {
    title: "Layered evidence map",
    body: [
      "Grad-CAM attention combined with lesion candidates found by classical image processing on the enhanced image. The optic disc (OD) is located and excluded.",
    ],
    legend: [["#facc15", "Yellow circle: bright lesion candidate (exudate-like)"], ["#ff4d4d", "Red circle: dark lesion candidate (haemorrhage / microaneurysm-like)"], ["#ffffff", "White circle: optic disc estimate"]],
    tip: "Candidates are a heuristic, not a trained lesion detector; small or faint lesions can be missed and reflections can be flagged.",
  },
  consistency: {
    title: "Focus consistency",
    body: [
      "Measures how much of the model's high-attention area (Grad-CAM ≥ 50 % of peak) lies close to a lesion candidate.",
    ],
    legend: [["#22c55e", "Green: attention on a lesion candidate (agreement)"], ["#0090ff", "Blue: attention without a candidate"], ["#ffc800", "Amber: candidate without attention"]],
    tip: "High (> 40 %), Partial (15–40 %), Low (< 15 %). Low agreement on a No DR image is expected; on a high grade it is a reason to look closer.",
  },
  saliency: {
    title: "Attention spread",
    body: [
      "Contours at 25 %, 50 % and 75 % of peak attention show how attention narrows from context to the decisive spots.",
      "The label is based on the smallest share of the retina that holds half of all attention: Focused (< 10 %), Moderate (10–25 %), Diffuse (> 25 %).",
    ],
    legend: [["#ff4d4d", "Red contour: ≥ 75 %"], ["#facc15", "Yellow contour: ≥ 50 %"], ["#0090ff", "Blue contour: ≥ 25 %"]],
  },
  regions: {
    title: "Region importance",
    body: [
      "Share of attention in three concentric zones scaled to the retina's size: central (macula / posterior pole, inner 25 % of the radius), mid (25–60 %) and peripheral (60–100 %).",
    ],
    tip: "Assumes a macula-centred photograph, the standard screening field.",
  },
  pseudo3d: {
    title: "Pseudo-3D layered view",
    body: [
      "Stacks four layers: dimmed retina, extracted blood vessels (white), lesion candidates, and Grad-CAM attention drawn as a shaded relief, where higher 'hills' mean stronger attention.",
    ],
    tip: "Useful for seeing whether attention follows vessels (e.g. new vessels in proliferative DR) or isolated lesions.",
  },
  quality: {
    title: "Image quality",
    body: [
      "Flags parts of the photo that are unreliable for grading and gives an overall 0–100 score (Good ≥ 75, Fair ≥ 50, Poor < 50).",
    ],
    legend: [["#808080", "Grey: blurred / low-detail area"], ["#facc15", "Yellow: over-exposed or glare"], ["#0050c8", "Blue: under-exposed (too dark)"]],
    tip: "If the macula or large areas are flagged, retake the photo before relying on the result.",
  },
};

/* ------------------------------------------------------------------ */
/* Helpers                                                             */
/* ------------------------------------------------------------------ */
function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === "class") node.className = v;
    else if (k === "style") node.setAttribute("style", v);
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else if (k in node) node[k] = v;
    else node.setAttribute(k, v);
  }
  for (const c of children) if (c != null) node.append(c);
  return node;
}

let toastTimer;
function toast(message, kind = "error") {
  const t = $("toast");
  t.textContent = message;
  t.className = "toast" + (kind === "success" ? " success" : "");
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (t.hidden = true), 5000);
}

async function api(url, options = {}) {
  let response;
  try {
    response = await fetch(url, options);
  } catch {
    throw new Error("Cannot reach the server. Check your connection and try again.");
  }
  let data = null;
  try { data = await response.json(); } catch { /* not JSON */ }
  if (!response.ok) {
    const detail = data && (typeof data.detail === "string" ? data.detail : null);
    throw new Error(detail || `Request failed (HTTP ${response.status}).`);
  }
  return data;
}

const pct = (v, digits = 1) => (v == null ? "n/a" : `${(v * 100).toFixed(digits)}%`);

function openLightbox(src, caption) {
  $("lightboxImg").src = src;
  $("lightboxImg").alt = caption;
  $("lightboxCaption").textContent = caption;
  $("lightbox").showModal();
}

function showInfo(key) {
  const info = INFO[key];
  $("infoTitle").textContent = info.title;
  const body = $("infoBody");
  body.replaceChildren(...info.body.map((p) => el("p", { textContent: p })));
  if (info.legend) {
    body.append(el("ul", {}, ...info.legend.map(([color, text]) =>
      el("li", {}, el("span", { class: "legend", style: `background:${color}` }), text))));
  }
  if (info.tip) body.append(el("p", { class: "muted" }, el("b", { textContent: "Tip: " }), info.tip));
  $("infoDialog").showModal();
}

/* ------------------------------------------------------------------ */
/* Init                                                                */
/* ------------------------------------------------------------------ */
async function init() {
  document.querySelectorAll("[data-open]").forEach((b) =>
    b.addEventListener("click", () => $(b.dataset.open).showModal()));
  $("lightbox").addEventListener("click", (e) => { if (e.target.id === "lightbox") $("lightbox").close(); });

  setupUpload();

  try {
    state.config = await api("/api/config");
    $("uploadHint").textContent = `JPG, PNG, BMP or TIFF · up to ${state.config.max_upload_mb} MB`;
    buildReferenceTabs();
  } catch (e) {
    toast(e.message);
  }

  try {
    const health = await api("/api/health");
    setStatus(health.model_loaded, health.model_loaded ? "Model ready" : "Model unavailable");
  } catch {
    setStatus(false, "Server offline");
  }
}

function setStatus(ok, text) {
  const pill = $("modelStatus");
  pill.textContent = text;
  pill.className = "status-pill " + (ok ? "ok" : "bad");
}

/* ------------------------------------------------------------------ */
/* Upload                                                              */
/* ------------------------------------------------------------------ */
function setupUpload() {
  const input = $("imageInput");
  const zone = $("dropZone");

  input.addEventListener("change", () => selectFile(input.files[0]));
  ["dragenter", "dragover"].forEach((ev) => zone.addEventListener(ev, (e) => { e.preventDefault(); zone.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((ev) => zone.addEventListener(ev, (e) => { e.preventDefault(); zone.classList.remove("drag"); }));
  zone.addEventListener("drop", (e) => selectFile(e.dataTransfer.files[0]));

  $("analyzeBtn").addEventListener("click", analyze);
  $("copyPwd").addEventListener("click", copyPassword);
  $("emailForm").addEventListener("submit", sendEmail);
}

function selectFile(file) {
  if (!file) return;
  const maxMb = state.config ? state.config.max_upload_mb : 15;
  if (!/\.(jpe?g|png|bmp|tiff?)$/i.test(file.name)) {
    toast("Please choose a JPG, PNG, BMP or TIFF image.");
    return;
  }
  if (file.size > maxMb * 1024 * 1024) {
    toast(`The file is larger than ${maxMb} MB.`);
    return;
  }
  state.file = file;
  const preview = $("previewImg");
  if (preview.src) URL.revokeObjectURL(preview.src);
  preview.src = URL.createObjectURL(file);
  preview.hidden = false;
  $("dropText").hidden = true;
  $("analyzeBtn").disabled = false;
}

const LOADING_STEPS = ["Preprocessing image", "Running EfficientNet-B3", "Computing Grad-CAM attention",
  "Detecting lesion candidates", "Assessing image quality", "Writing encrypted report"];

async function analyze() {
  if (!state.file) { toast("Please choose an image first."); return; }

  const btn = $("analyzeBtn");
  btn.disabled = true;
  $("emptyState").hidden = true;
  $("results").hidden = true;
  $("loading").hidden = false;

  let step = 0;
  $("loadingStep").textContent = LOADING_STEPS[0];
  const timer = setInterval(() => {
    step = Math.min(step + 1, LOADING_STEPS.length - 1);
    $("loadingStep").textContent = LOADING_STEPS[step];
  }, 900);

  const form = new FormData();
  form.append("file", state.file);
  form.append("preprocessed", $("preprocessed").value);

  try {
    const data = await api("/api/analyze", { method: "POST", body: form });
    state.result = data;
    renderResults(data);
  } catch (e) {
    toast(e.message);
    $("emptyState").hidden = state.result != null;
    if (state.result) $("results").hidden = false;
  } finally {
    clearInterval(timer);
    $("loading").hidden = true;
    btn.disabled = false;
  }
}

/* ------------------------------------------------------------------ */
/* Results                                                             */
/* ------------------------------------------------------------------ */
function renderResults(d) {
  $("results").hidden = false;

  // ---- summary
  $("predLabel").textContent = d.prediction;
  $("predLabel").style.color = GRADE_COLORS[d.class_index];
  $("confidence").textContent = pct(d.confidence);
  $("modelName").textContent = state.config ? state.config.model : "EfficientNet-B3";

  const q = d.quality;
  const qClass = q.label === "Good" ? "good" : q.label === "Fair" ? "mid" : "bad";
  const cClass = d.consistency_label.startsWith("High") ? "good" : d.consistency_label.startsWith("Partial") ? "mid" : "bad";
  $("chips").replaceChildren(
    chip("Image quality", `${q.label} (${Math.round(q.score)}/100)`, qClass),
    chip("Focus consistency", d.consistency_score == null ? d.consistency_label : `${d.consistency_label} · ${pct(d.consistency_score, 0)}`, cClass),
    chip("Attention", d.attention_label),
    chip("Lesion candidates", `${d.lesion_counts.bright} bright · ${d.lesion_counts.dark} dark`),
    chip("Input", d.already_preprocessed ? "Preprocessed image" : "Raw photograph"),
  );

  const warnings = [];
  if (d.confidence < 0.6) warnings.push("Low model confidence: the grade is uncertain between neighbouring levels.");
  if (q.label === "Poor") warnings.push("Poor image quality: consider retaking the photograph.");
  $("lowConfidence").textContent = warnings.join(" ");
  $("lowConfidence").hidden = warnings.length === 0;

  renderBars($("probBars"), d.classes.map((name, i) => ({
    label: name, value: d.probabilities[i], active: i === d.class_index,
    color: GRADE_COLORS[i],
  })));

  // ---- gallery
  $("gallery").replaceChildren(...Object.entries(d.images).map(([key, src]) => {
    const title = d.image_titles[key] || key;
    const img = el("img", { src, alt: title, loading: "lazy", onclick: () => openLightbox(src, title) });
    const info = el("button", { class: "info-btn ghost", type: "button", textContent: "ⓘ",
      "aria-label": `About ${title}`, onclick: () => showInfo(key) });
    return el("article", { class: "card" }, img, el("div", { class: "card-foot" }, el("span", { textContent: title }), info));
  }));

  // ---- regions
  const r = d.region_importance;
  renderBars($("regionBars"), [
    { label: "Central", value: r.central, color: "#3b82f6" },
    { label: "Mid retina", value: r.mid, color: "#22c55e" },
    { label: "Peripheral", value: r.peripheral, color: "#facc15" },
  ]);

  // ---- report
  if (d.report_url) {
    $("pdfPassword").textContent = d.report_password;
    $("previewLink").href = d.report_url;
    $("downloadLink").href = d.report_url + "?download=true";
  }
  const emailOn = state.config && state.config.email_enabled;
  $("emailForm").hidden = !emailOn;
  $("emailDisabled").hidden = !!emailOn;
  $("ttlNote").textContent = state.config ? `Results and report are deleted automatically after ${state.config.result_ttl_hours} hours.` : "";

  // ---- comparison
  renderComparison(d);

  $("results").scrollIntoView({ behavior: "smooth", block: "start" });
}

function chip(label, value, kind = "") {
  return el("span", { class: `chip ${kind}` }, `${label}: `, el("b", { textContent: value }));
}

function renderBars(container, rows) {
  container.replaceChildren(...rows.map((row) => {
    const fill = el("div", { class: "fill" });
    if (row.color) fill.style.background = row.color;
    const node = el("div", { class: "bar-row" + (row.active ? " active" : "") },
      el("span", { textContent: row.label }),
      el("div", { class: "track" }, fill),
      el("span", { class: "pct", textContent: pct(row.value) }));
    requestAnimationFrame(() => requestAnimationFrame(() => (fill.style.width = `${Math.max(0, Math.min(1, row.value)) * 100}%`)));
    return node;
  }));
}

function renderComparison(d) {
  const figure = (src, caption) => src
    ? el("figure", {}, el("img", { src, alt: caption, loading: "lazy", onclick: () => openLightbox(src, caption) }), el("figcaption", { textContent: caption }))
    : el("figure", {}, el("div", { class: "placeholder", textContent: "No reference image available. Add examples to data/reference_images/<grade>/" }), el("figcaption", { textContent: caption }));

  $("comparison").replaceChildren(
    figure(d.images.original, "Your image"),
    figure(d.references.same_class_example, `Reference: ${d.prediction}`),
    figure(d.references.severe_example, "Reference: Proliferative DR"),
  );
}

/* ------------------------------------------------------------------ */
/* Report actions                                                      */
/* ------------------------------------------------------------------ */
async function copyPassword() {
  const pwd = $("pdfPassword").textContent;
  try {
    await navigator.clipboard.writeText(pwd);
    toast("Password copied to clipboard.", "success");
  } catch {
    toast("Could not copy automatically. Please select the password and copy it.");
  }
}

async function sendEmail(e) {
  e.preventDefault();
  if (!state.result) return;
  const email = $("doctorEmail").value.trim();
  if (!$("doctorEmail").checkValidity()) { toast("Please enter a valid email address."); return; }

  const btn = $("sendBtn");
  btn.disabled = true;
  btn.textContent = "Sending…";
  try {
    await api(`/api/reports/${state.result.result_id}/email`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email }),
    });
    toast(`Report sent to ${email}. Share the password separately.`, "success");
  } catch (err) {
    toast(err.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "Email report";
  }
}

/* ------------------------------------------------------------------ */
/* Reference library                                                   */
/* ------------------------------------------------------------------ */
function buildReferenceTabs() {
  const classes = state.config.classes;
  $("refTabs").replaceChildren(...classes.map((name, i) => el("button", {
    type: "button", role: "tab", textContent: name, "aria-selected": "false",
    onclick: () => loadReference(i),
  })));
  const total = state.config.reference_counts.reduce((a, b) => a + b, 0);
  if (total === 0) {
    $("referenceImages").replaceChildren(el("p", { class: "muted small",
      textContent: "No reference images installed. Add labelled examples to data/reference_images/0 … 4 to enable this library and the comparison panel." }));
  } else {
    loadReference(0);
  }
}

async function loadReference(index) {
  state.refClass = index;
  [...$("refTabs").children].forEach((b, i) => b.setAttribute("aria-selected", String(i === index)));
  const box = $("referenceImages");
  try {
    const data = await api(`/api/reference/${index}`);
    if (state.refClass !== index) return;
    if (!data.images.length) {
      box.replaceChildren(el("p", { class: "muted small", textContent: `No reference images for ${data.class}.` }));
      return;
    }
    box.replaceChildren(...data.images.map((src) =>
      el("img", { src, alt: `${data.class} reference case`, loading: "lazy", onclick: () => openLightbox(src, `${data.class} reference case`) })));
  } catch (e) {
    box.replaceChildren(el("p", { class: "muted small", textContent: e.message }));
  }
}

init();

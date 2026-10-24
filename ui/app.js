/* docintel viewer: page image + grounded overlays, chunk inspector, schema extraction, graph, export. */
const $ = (s) => document.querySelector(s);
const state = { docs: [], doc: null, page: 1, active: null, poll: null };
const TYPES = ["text","title","section_header","list_item","table","figure","chart","map","infographic","form_field","caption","handwriting","formula","footnote","page_header","page_footer","page_number"];
const color = (t) => getComputedStyle(document.documentElement).getPropertyValue(`--${t}`).trim() || "#999";

// ---------------------------------------------------------------- api
const api = {
  list: () => fetch("/api/documents").then(r => r.json()),
  get: (id) => fetch(`/api/documents/${id}`).then(r => r.json()),
  upload: (file, schema, mode) => { const fd = new FormData(); fd.append("file", file); fd.append("mode", mode || "parse"); if (schema) fd.append("schema", schema); return fetch("/api/documents", { method: "POST", body: fd }).then(r => r.json()); },
  suggest: (id) => fetch(`/api/documents/${id}/suggest-schema`, { method: "POST" }).then(r => r.json()),
  extract: (id, schema, name) => fetch(`/api/documents/${id}/extract`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ schema, name }) }).then(r => r.json()),
  search: (id, q) => fetch(`/api/documents/${id}/search?q=${encodeURIComponent(q)}`).then(r => r.json()),
  health: () => fetch("/api/health").then(r => r.json()),
};

// ---------------------------------------------------------------- documents list
async function refreshDocs() {
  state.docs = await api.list();
  const el = $("#docs"); el.innerHTML = "";
  for (const d of state.docs) {
    const div = document.createElement("div");
    div.className = "doc" + (state.doc && state.doc.doc_id === d.doc_id ? " active" : "");
    const prog = d.status === "running" ? ` · ${esc(d.stage || "")} ${Math.round((d.progress || 0) * 100)}%` : d.status === "failed" ? ` · ✖ ${esc(d.error || "failed")}` : "";
    div.innerHTML = `<div class="name">${esc(d.filename)}</div><div class="meta">${d.n_pages || "?"} pages · ${d.n_chunks ?? "…"} chunks · ${esc((d.languages || []).join(",") || "–")} · <span class="badge">${esc(d.mode || "parse")}${d.n_fields ? ` · ${d.n_fields} fields` : ""}</span>${prog}</div>`;
    div.onclick = () => openDoc(d.doc_id);
    el.appendChild(div);
  }
  const busy = state.docs.some(d => d.status === "queued" || d.status === "running");
  clearTimeout(state.poll); if (busy) state.poll = setTimeout(refreshDocs, 1500);
}

async function openDoc(id) {
  const d = await api.get(id);
  if (d.status === "failed") { alert("Parsing failed: " + (d.error || "unknown error")); refreshDocs(); return; }
  if (d.status && d.status !== "parsed") { setTimeout(() => openDoc(id), 1500); return; }
  state.doc = d; state.page = 1; state.active = null;
  renderAll(); refreshDocs();
}

// ---------------------------------------------------------------- page viewer
function curPage() { return state.doc?.pages.find(p => p.number === state.page); }

function renderPage() {
  const p = curPage(); if (!p) return;
  $("#pageLabel").textContent = `page ${state.page} / ${state.doc.n_pages}`;
  const img = $("#pageImg");
  img.src = `/api/documents/${state.doc.doc_id}/pages/${state.page}`;
  const w = 8 * parseInt($("#zoom").value); $("#page").style.width = w + "px";
  const svg = $("#overlay"); svg.setAttribute("viewBox", `0 0 ${p.width} ${p.height}`); svg.innerHTML = "";
  if (!$("#showBoxes").checked) return;
  for (const c of p.chunks) {
    const r = rect(c.bbox, p, color(c.type), c.id);
    r.classList.toggle("active", state.active === c.id);
    r.onclick = () => setActive(c.id);
    const t = document.createElementNS("http://www.w3.org/2000/svg", "title"); t.textContent = `${c.type} · ${(c.confidence * 100).toFixed(0)}%\n${c.text.slice(0, 200)}`; r.appendChild(t);
    svg.appendChild(r);
    if ($("#showWords").checked) for (const wd of c.words || []) svg.appendChild(rect(wd.bbox, p, "#ff3d71", null, "word"));
  }
  // extraction groundings (word-level) on top
  for (const f of state.doc.extraction?.fields || []) for (const g of f.groundings || []) if (g.page === state.page) {
    const r = rect(g.bbox, p, "#ff3d71", null, "word"); r.setAttribute("stroke-width", "3");
    const t = document.createElementNS("http://www.w3.org/2000/svg", "title"); t.textContent = `${f.name} = ${f.value}`; r.appendChild(t); svg.appendChild(r);
  }
}
function rect(b, p, stroke, id, cls) {
  const r = document.createElementNS("http://www.w3.org/2000/svg", "rect");
  r.setAttribute("x", b.x0 * p.width); r.setAttribute("y", b.y0 * p.height);
  r.setAttribute("width", Math.max(2, (b.x1 - b.x0) * p.width)); r.setAttribute("height", Math.max(2, (b.y1 - b.y0) * p.height));
  r.setAttribute("stroke", stroke); if (id) r.dataset.id = id; if (cls) r.classList.add(cls);
  return r;
}
function setActive(id, jump = true) {
  state.active = id;
  const c = state.doc.pages.flatMap(p => p.chunks).find(c => c.id === id);
  if (c && c.page !== state.page && jump) { state.page = c.page; renderPage(); renderChunks(); }
  document.querySelectorAll("#overlay rect").forEach(r => r.classList.toggle("active", r.dataset.id === id));
  document.querySelectorAll(".chunk").forEach(e => e.classList.toggle("active", e.dataset.id === id));
  const el = document.querySelector(`.chunk[data-id="${id}"]`); if (el) el.scrollIntoView({ block: "nearest" });
}
function showPage(n) { if (!state.doc) return; state.page = Math.min(Math.max(1, n), state.doc.n_pages); renderPage(); renderChunks(); }

// ---------------------------------------------------------------- right panel
function renderChunks(list) {
  const p = curPage(); if (!p) return;
  const el = $("#chunks"); el.innerHTML = "";
  const chunks = list || [...p.chunks].sort((a, b) => a.reading_order - b.reading_order);
  for (const c of chunks) {
    const div = document.createElement("div"); div.className = "chunk" + (state.active === c.id ? " active" : "");
    div.dataset.id = c.id; div.style.borderLeftColor = color(c.type);
    const langs = (c.languages || []).map(l => `<span class="badge">${esc(l)}</span>`).join("");
    const conf = c.confidence < 0.75 ? `<span class="warn">${Math.round(c.confidence * 100)}%</span>` : `${Math.round(c.confidence * 100)}%`;
    div.innerHTML = `<div class="t"><span>${esc(c.type)}</span><span>p${c.page}</span><span>${conf}</span><span>${esc(c.source)}</span>${langs}</div><pre>${esc(c.text.slice(0, 600))}${c.text.length > 600 ? "…" : ""}</pre>`;
    div.onclick = () => setActive(c.id);
    el.appendChild(div);
  }
}
function renderTables() {
  const el = $("#tab-tables"); el.innerHTML = "";
  for (const p of state.doc.pages) for (const c of p.chunks) {
    if (!["table", "chart", "map", "infographic", "figure"].includes(c.type)) continue;
    const d = document.createElement("div"); d.className = "chunk"; d.style.borderLeftColor = color(c.type); d.dataset.id = c.id;
    let body = "";
    if (c.table) body = gridHtml(c.table);
    if (c.image_path) body += `<img src="/api/documents/${state.doc.doc_id}/crops/${c.id}" style="max-width:100%;margin-top:6px;border:1px solid var(--line)">`;
    if (c.chart) body += `<div class="muted" style="margin-top:4px">${esc(c.chart.chart_type)} · ${esc(c.chart.title || "")}</div>${c.chart.raw_table ? gridHtml(c.chart.raw_table) : ""}<div>${esc(c.chart.summary || "")}</div>`;
    if (!c.table && !c.chart) body += `<pre>${esc(c.text.slice(0, 400))}</pre>`;
    d.innerHTML = `<div class="t"><span>${esc(c.type)}</span><span>p${c.page}</span><span>${esc(c.attributes?.visual_kind || "")}</span></div>${body}`;
    d.onclick = () => setActive(c.id); el.appendChild(d);
  }
  if (!el.children.length) el.innerHTML = '<p class="muted">No tables, charts or figures detected.</p>';
}
function gridHtml(t) {
  const grid = []; for (let r = 0; r < t.n_rows; r++) grid.push(new Array(t.n_cols).fill(""));
  for (const c of t.cells) if (grid[c.row]) grid[c.row][c.col] = c.text;
  return `<table class="grid">${grid.map((row, i) => `<tr>${row.map(x => `<${i ? "td" : "th"}>${esc(x)}</${i ? "td" : "th"}>`).join("")}</tr>`).join("")}</table>`;
}
function renderExtraction(ex) {
  const el = $("#fields"); el.innerHTML = "";
  if (!ex) { $("#extractJson").textContent = ""; return; }
  for (const f of ex.fields) {
    const g = f.groundings?.[0];
    const d = document.createElement("div"); d.className = "field";
    const ctx = f.context || {}; const ctxTxt = [ctx.section && `§ ${ctx.section}`, ctx.label && `label: ${ctx.label}`, ctx.row_label && `row: ${ctx.row_label}`, ctx.column_header && `col: ${ctx.column_header}`].filter(Boolean).join(" · ");
    const val = f.value == null ? '<span class="muted">null</span>' : (typeof f.value === "object" ? `<pre class="json">${esc(JSON.stringify(f.value, null, 1))}</pre>` : esc(String(f.value)));
    const snip = g && !Array.isArray(f.value) ? `<img src="/api/documents/${state.doc.doc_id}/snippets/${encodeURIComponent(f.name)}?t=${Date.now()}" style="max-width:100%;max-height:70px;margin-top:4px;border:1px solid var(--line);background:#fff" onerror="this.remove()">` : "";
    d.innerHTML = `<b>${esc(f.name)}</b> <span class="${f.verified ? "ok" : "warn"}">${f.verified ? "✓ verified" : "⚠ unverified"}</span> ${f.type_ok === false ? '<span class="warn">type✗</span>' : ""} <span class="muted">${Math.round(f.confidence * 100)}%</span><div class="v">${val}</div>${g ? `<div class="muted">page ${g.page}${ctxTxt ? " · " + esc(ctxTxt) : ""}${f.notes ? " · " + esc(f.notes) : ""}</div>${snip}` : '<div class="muted">no grounding</div>'}`;
    if (g) d.onclick = () => { state.page = g.page; renderPage(); renderChunks(); if (g.chunk_id) setActive(g.chunk_id, false); };
    el.appendChild(d);
  }
  $("#extractJson").textContent = JSON.stringify(ex.data, null, 1);
}
function renderGraph() {
  const el = $("#tab-graph"); const t = state.doc.triples || [];
  el.innerHTML = `<p class="muted">${t.length} triples with provenance (also exported as Neo4j Cypher + ragdoc nodes/edges).</p>` +
    t.slice(0, 500).map(x => `<div class="chunk" data-id="${x.groundings?.[0]?.chunk_id || ""}" style="border-left-color:var(--list_item)"><span>${esc(x.subject)}</span> <span class="badge">${esc(x.predicate)}</span> <span>${esc(x.object)}</span> <span class="muted">p${x.groundings?.[0]?.page ?? "?"}</span></div>`).join("");
  el.querySelectorAll(".chunk").forEach(d => d.onclick = () => d.dataset.id && setActive(d.dataset.id));
}
function renderAgent() {
  const pl = state.doc.pipeline || {};
  const steps = (pl.agent_trace || []).map(s => `<div class="chunk" data-id="${esc(s.chunk_id || "")}" style="border-left-color:var(--chart)"><div class="t"><span>${esc(s.action)}</span></div><pre>${esc(JSON.stringify(s, null, 0).slice(0, 300))}</pre></div>`).join("");
  $("#tab-agent").innerHTML = `<pre class="json">${esc(JSON.stringify({ ocr_backend: pl.ocr_backend, layout_engine: pl.layout_engine, llm: pl.llm, page_sources: pl.page_sources, timings_s: pl.timings_s, total_s: pl.total_s, warnings: pl.warnings }, null, 1))}</pre><h4 style="margin:8px 0">Agent trace (${(pl.agent_trace || []).length} steps)</h4>${steps || '<p class="muted">No agent actions were needed.</p>'}`;
  $("#tab-agent").querySelectorAll(".chunk").forEach(d => d.onclick = () => d.dataset.id && setActive(d.dataset.id));
}
function renderAll() {
  renderPage(); renderChunks(); renderTables(); renderGraph(); renderAgent(); renderExtraction(state.doc.extraction);
  const sug = state.doc.metadata && state.doc.metadata.suggested_schema;
  if (sug) { const sc = { ...sug }; delete sc._source; delete sc._llm_error; $("#schema").value = JSON.stringify(sc, null, 2); }
  $("#md").textContent = state.doc.markdown; $("#json").textContent = JSON.stringify(state.doc, null, 1).slice(0, 200000);
}
const esc = (s) => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

// ---------------------------------------------------------------- wiring
$("#prev").onclick = () => showPage(state.page - 1); $("#next").onclick = () => showPage(state.page + 1);
$("#zoom").oninput = renderPage; $("#showBoxes").onchange = renderPage; $("#showWords").onchange = renderPage;
document.querySelectorAll(".tabs div").forEach(t => t.onclick = () => { document.querySelectorAll(".tabs div,.tab").forEach(e => e.classList.remove("active")); t.classList.add("active"); $(`#tab-${t.dataset.tab}`).classList.add("active"); });
$("#export").onclick = () => state.doc && (location.href = `/api/documents/${state.doc.doc_id}/export`);
$("#runExtract").onclick = async () => {
  if (!state.doc) return; let schema; try { schema = JSON.parse($("#schema").value); } catch { $("#extractStatus").textContent = "invalid JSON"; return; }
  $("#extractStatus").textContent = "extracting…"; $("#runExtract").disabled = true;
  try { const ex = await api.extract(state.doc.doc_id, schema, "ui"); state.doc.extraction = ex; renderExtraction(ex); renderPage(); $("#extractStatus").textContent = `${ex.fields.filter(f => f.value != null).length}/${ex.fields.length} fields found`; }
  catch (e) { $("#extractStatus").textContent = "error: " + e; } finally { $("#runExtract").disabled = false; }
};
let searchT; $("#search").oninput = () => { clearTimeout(searchT); searchT = setTimeout(async () => {
  if (!state.doc) return; const q = $("#search").value.trim(); if (!q) return renderChunks();
  const hits = await api.search(state.doc.doc_id, q); const all = state.doc.pages.flatMap(p => p.chunks);
  renderChunks(hits.map(h => all.find(c => c.id === h.chunk_id)).filter(Boolean));
}, 250); };
function uploadMode() { return document.querySelector("input[name=mode]:checked").value; }
document.querySelectorAll("input[name=mode]").forEach(r => r.onchange = () => { $("#uploadSchema").style.display = uploadMode() === "schema" ? "block" : "none"; });
async function doUpload(file) {
  const mode = uploadMode(); const schema = $("#uploadSchema").value.trim();
  if (mode === "schema" && !schema) { alert("Paste a schema first, or choose 'Parse only' / 'Auto-discover'."); return; }
  const r = await api.upload(file, mode === "schema" ? schema : null, mode);
  if (r.detail) alert(typeof r.detail === "string" ? r.detail : JSON.stringify(r.detail)); refreshDocs();
}
$("#suggestSchema").onclick = async () => {
  if (!state.doc) return; $("#extractStatus").textContent = "proposing schema…"; $("#suggestSchema").disabled = true;
  try { const sc = await api.suggest(state.doc.doc_id); delete sc._source; delete sc._llm_error; $("#schema").value = JSON.stringify(sc, null, 2); $("#extractStatus").textContent = `${Object.keys(sc.properties || {}).length} fields proposed — edit, then Extract`; }
  catch (e) { $("#extractStatus").textContent = "error: " + e; } finally { $("#suggestSchema").disabled = false; }
};
$("#file").onchange = (e) => e.target.files[0] && doUpload(e.target.files[0]);
const drop = $("#drop"); drop.ondragover = (e) => { e.preventDefault(); drop.classList.add("over"); }; drop.ondragleave = () => drop.classList.remove("over");
drop.ondrop = (e) => { e.preventDefault(); drop.classList.remove("over"); [...e.dataTransfer.files].forEach(doUpload); };
document.addEventListener("keydown", (e) => { if (e.target.tagName === "TEXTAREA" || e.target.tagName === "INPUT") return; if (e.key === "ArrowLeft") showPage(state.page - 1); if (e.key === "ArrowRight") showPage(state.page + 1); });
$("#legend").innerHTML = TYPES.map(t => `<span style="border-color:${color(t)};color:${color(t)}">${t}</span>`).join("");
api.health().then(h => $("#health").textContent = `ocr: ${h.ocr_backend} · layout: ${h.layout} · llm: ${h.llm || "offline"}`).catch(() => {});
refreshDocs();

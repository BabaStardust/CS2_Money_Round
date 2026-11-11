"use strict";

// Zustand der Seite. Rundenverlauf immer aus Sicht des Gegnerteams.
const state = {
  file: null,        // hochgeladene Datei
  example: null,     // Name eines Beispiel-Screenshots
  reading: null,     // Lesung des Vision-Modells
  derived: null,     // Vorschläge je Seite (Historie, Kills, Deaths)
  side: "T",
  store: null,       // Featurestore/Registry für die Inferenz: example | hopsworks | local
  rounds: [],        // [{won: bool, reason: "e"|"b"|"d"|"t"|"?"}]
};

const REASONS = { e: "alle tot", b: "Bombe", d: "entschärft", t: "Zeit", "?": "unbekannt" };
const FEATURE_LABELS = {
  round_in_half: "Runde in der Halbzeit",
  is_ct: "Gegner ist CT",
  score_team: "Runden Gegner",
  score_opp: "Runden eigenes Team",
  prev_round_won: "letzte Runde gewonnen",
  prev_reason_bomb: "letzte Runde: Bombe",
  prev_reason_defuse: "letzte Runde: entschärft",
  prev_reason_time: "letzte Runde: Zeit",
  loss_streak: "Niederlagen in Folge",
  loss_bonus_level: "Loss-Bonus-Stufe",
  wins_last_3: "Siege in den letzten 3",
  deaths_per_round: "Tode pro Runde",
  kill_death_diff: "Kills − Deaths",
  hist_full_buy_rate: "historische Quote (Featurestore)",
};

const $ = (id) => document.getElementById(id);
const el = (tag, props = {}, ...children) => {
  const node = Object.assign(document.createElement(tag), props);
  for (const c of children) node.append(c);
  return node;
};
const pct = (x) => `${Math.round(x * 100)} %`;

async function api(path, options) {
  const res = await fetch(path, options);
  let body = null;
  try { body = await res.json(); } catch { /* leer */ }
  if (!res.ok) {
    const detail = body && body.detail;
    const msg = Array.isArray(detail) ? detail.map((d) => d.msg).join("; ") : detail;
    throw new Error(msg || `Fehler ${res.status}`);
  }
  return body;
}

// ------------------------------------------------------------ Start
async function init() {
  bindTabs();
  bindUpload();
  bindValues();
  renderRounds();
  try {
    const s = await api("/api/status");
    renderStores(s.stores, s.default_store);
    const sel = $("provider");
    for (const [key, label] of Object.entries(s.providers)) sel.append(el("option", { value: key, textContent: label }));
    if (s.default_provider) sel.value = s.default_provider;
    if (!Object.keys(s.providers).length) {
      sel.append(el("option", { textContent: "kein Vision-Modell verfügbar", value: "" }));
      sel.disabled = true;
    }
  } catch (e) {
    $("status").textContent = `Server nicht erreichbar: ${e.message}`;
  }
  try {
    const list = await api("/api/examples");
    if (list.length) {
      $("examples-wrap").hidden = false;
      for (const ex of list) {
        const btn = el("button", { className: "example", type: "button", title: ex.note || ex.name },
          el("img", { src: `/examples/${encodeURIComponent(ex.name)}`, alt: "", loading: "lazy" }),
          el("span", { textContent: ex.note || ex.name }));
        btn.setAttribute("aria-pressed", "false");
        btn.addEventListener("click", () => chooseExample(ex.name, btn));
        $("examples").append(btn);
      }
    }
  } catch { /* Beispiele sind optional */ }
}

// ------------------------------------------------------------ Featurestore-Schalter
const STORE_HINTS = {
  example: "Mitgelieferter Schnappschuss aus Hopsworks im Repo (examples/store) – läuft ohne Zugang.",
  hopsworks: "Live aus Hopsworks: Feature View buy_rate_fv (online) und Model Registry – braucht HOPSWORKS_API_KEY.",
  local: "Eigener lokaler Store aus FEATURE_STORE=local (data/featurestore, models/registry).",
};
const STORE_KEY = "cs2.store";

function renderStores(stores, fallback) {
  const box = $("store");
  box.replaceChildren();
  const kinds = Object.keys(stores);
  if (!kinds.length) {
    $("store-info").textContent = "Kein Featurestore verfügbar – Beispiel-Store fehlt und kein Hopsworks-Key gesetzt.";
    return;
  }
  for (const kind of kinds) {
    const b = el("button", { type: "button", textContent: stores[kind] });
    b.dataset.store = kind;
    b.addEventListener("click", () => setStore(kind, true));
    box.append(b);
  }
  let saved = null;
  try { saved = localStorage.getItem(STORE_KEY); } catch { /* Speicher gesperrt */ }
  setStore(kinds.includes(saved) ? saved : fallback || kinds[0], false);
}

async function setStore(kind, remember) {
  state.store = kind;
  for (const b of $("store").querySelectorAll("button")) b.setAttribute("aria-pressed", String(b.dataset.store === kind));
  if (remember) { try { localStorage.setItem(STORE_KEY, kind); } catch { /* Speicher gesperrt */ } }
  $("store-info").textContent = `${STORE_HINTS[kind] || ""} Lade Modell …`;
  $("status").textContent = "Lade Modell …";
  try {
    const m = await api(`/api/model?store=${encodeURIComponent(kind)}`);
    if (state.store !== kind) return;  // inzwischen umgeschaltet
    $("status").innerHTML = `Modell <b>v${m.version}</b> · ROC-AUC ${m.roc_auc} · <b>${m.label}</b>`;
    $("store-info").textContent = STORE_HINTS[kind] || "";
  } catch (e) {
    if (state.store !== kind) return;
    $("status").textContent = "Modell nicht geladen";
    $("store-info").textContent = e.message;
  }
}

function bindTabs() {
  const tabs = [["tab-shot", "panel-shot"], ["tab-manual", "panel-manual"]];
  for (const [tab, panel] of tabs) {
    $(tab).addEventListener("click", () => {
      for (const [t, p] of tabs) {
        $(t).setAttribute("aria-selected", String(t === tab));
        $(p).hidden = p !== panel;
      }
    });
  }
}

// ------------------------------------------------------------ 1. Screenshot
function bindUpload() {
  const drop = $("drop");
  $("file").addEventListener("change", (e) => chooseFile(e.target.files[0]));
  drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("is-over"); });
  drop.addEventListener("dragleave", () => drop.classList.remove("is-over"));
  drop.addEventListener("drop", (e) => {
    e.preventDefault();
    drop.classList.remove("is-over");
    if (e.dataTransfer.files.length) chooseFile(e.dataTransfer.files[0]);
  });
  $("btn-read").addEventListener("click", readScreenshot);
}

function chooseFile(file) {
  if (!file) return;
  state.file = file;
  state.example = null;
  for (const b of document.querySelectorAll(".example")) b.setAttribute("aria-pressed", "false");
  $("drop").classList.add("has-file");
  $("drop").querySelector(".drop__title").textContent = file.name;
  $("preview").src = URL.createObjectURL(file);
  $("btn-read").disabled = !$("provider").value;
}

function chooseExample(name, btn) {
  state.example = name;
  state.file = null;
  for (const b of document.querySelectorAll(".example")) b.setAttribute("aria-pressed", String(b === btn));
  $("drop").classList.remove("has-file");
  $("drop").querySelector(".drop__title").textContent = "Screenshot wählen oder hierher ziehen";
  $("preview").src = `/examples/${encodeURIComponent(name)}`;
  $("btn-read").disabled = !$("provider").value;
}

async function readScreenshot() {
  const form = new FormData();
  form.append("provider", $("provider").value);
  if (state.file) form.append("file", state.file);
  else if (state.example) form.append("example", state.example);
  else return;

  const progress = $("read-progress");
  const label = $("provider").selectedOptions[0].textContent;
  const started = Date.now();
  const tick = () => { progress.textContent = `${label} liest das Bild … ${Math.round((Date.now() - started) / 1000)} s`; };
  tick();
  const timer = setInterval(tick, 1000);
  progress.hidden = false;
  $("btn-read").disabled = true;
  try {
    const res = await api("/api/read", { method: "POST", body: form });
    state.reading = res.reading;
    state.derived = res.derived;
    renderReading(res);
    if (res.default_side) setSide(res.default_side);
    else applyDerived();
  } catch (e) {
    $("reading").hidden = false;
    $("reading-meta").textContent = "";
    $("reading-teams").replaceChildren();
    $("reading-notes").replaceChildren(el("li", { textContent: e.message }));
  } finally {
    clearInterval(timer);
    progress.hidden = true;
    $("btn-read").disabled = false;
  }
}

function renderReading(res) {
  const r = res.reading;
  $("reading").hidden = false;
  $("reading-meta").textContent = `Gelesen mit ${$("provider").selectedOptions[0].textContent} in ${res.seconds} s`;
  const teams = [["CT", r.ct], ["T", r.t]].map(([side, team]) => {
    const head = el("h4", {}, `${side === "CT" ? "Counter-Terrorists" : "Terrorists"} `,
      el("span", { textContent: `Score ${team.score ?? "?"}` }));
    const thead = el("thead", {}, el("tr", {},
      el("th", { textContent: "Spieler" }), el("th", { className: "num", textContent: "Geld" }),
      el("th", { className: "num", textContent: "K" }), el("th", { className: "num", textContent: "D" }),
      el("th", { className: "num", textContent: "A" })));
    const rows = team.players.map((p) => {
      const tr = el("tr", { className: p.highlighted ? "is-own" : "" },
        el("td", { className: "name", textContent: p.name + (p.highlighted ? " (eigener Spieler)" : "") }),
        cell("Geld", p.money == null ? "–" : `$${p.money}`),
        cell("Kills", p.kills ?? "–"), cell("Deaths", p.deaths ?? "–"), cell("Assists", p.assists ?? "–"));
      return tr;
    });
    if (!rows.length) rows.push(el("tr", {}, el("td", { colSpan: 5, textContent: "keine Zeilen gelesen" })));
    return el("div", { className: "team" }, head, el("table", { className: "stack" }, thead, el("tbody", {}, ...rows)));
  });
  $("reading-teams").replaceChildren(...teams);

  const notes = [];
  if (!res.default_side) notes.push("Keine eindeutig hervorgehobene Zeile: Gegnerseite in Schritt 2 wählen.");
  for (const side of ["CT", "T"]) {
    if (state.derived[side].demo_check) notes.push(`Demo-Ansicht, Gegner ${side}: ${state.derived[side].demo_check}`);
  }
  const items = notes.map((n) => el("li", { textContent: n }));
  for (const f of r.uncertain_fields) {
    // Hinweise der Bildanalyse sind Information, der Rest ist unsicher gelesen
    const info = f.includes("Bildanalyse") || f.includes("Leiste gilt");
    items.push(el("li", { className: info ? "ok" : "", textContent: info ? f : `Unsicher: ${f}` }));
  }
  $("reading-notes").replaceChildren(...items);
}

function cell(label, value) {
  const td = el("td", { className: "num", textContent: String(value) });
  td.dataset.label = label;
  return td;
}

// ------------------------------------------------------------ 2. Werte
function bindValues() {
  for (const b of $("side").querySelectorAll("button")) b.addEventListener("click", () => setSide(b.dataset.side));
  $("round-add").addEventListener("click", () => { state.rounds.push({ won: false, reason: "e" }); renderRounds(); });
  $("round-remove").addEventListener("click", () => { state.rounds.pop(); renderRounds(); });
  $("history").addEventListener("input", (e) => {
    const parsed = parseHistory(e.target.value.trim());
    e.target.setCustomValidity(parsed ? "" : "Je Runde W/L und Grund e/b/d/t/?");
    if (parsed) { state.rounds = parsed; renderRounds(false); }
  });
  for (const id of ["kills", "deaths"]) {
    $(id).addEventListener("input", () => { $(id).classList.remove("is-uncertain"); renderSummary(); });
  }
  $("btn-predict").addEventListener("click", runPredict);
}

function setSide(side) {
  state.side = side;
  for (const b of $("side").querySelectorAll("button")) b.setAttribute("aria-pressed", String(b.dataset.side === side));
  applyDerived();
}

// Vorschlag aus dem Screenshot für die gewählte Gegnerseite übernehmen.
function applyDerived() {
  const d = state.derived && state.derived[state.side];
  if (d) {
    if (d.history !== null) state.rounds = parseHistory(d.history) || [];
    else if (d.score[0] !== null && d.score[1] !== null) {
      // Historie nicht lesbar: Rundenzahl aus dem Score vorgeben, Rest von Hand
      state.rounds = Array.from({ length: d.score[0] + d.score[1] }, () => ({ won: false, reason: "?" }));
    }
    if (d.kills !== null) $("kills").value = d.kills;
    if (d.deaths !== null) $("deaths").value = d.deaths;
  }
  // unsicher gelesene Werte des Gegnerteams gelb markieren (Plausibilitätsprüfung)
  const fields = (state.reading && state.reading.uncertain_fields) || [];
  const prefix = state.side.toLowerCase();
  for (const id of ["kills", "deaths"]) {
    const unsure = !!d && (d[id] === null || fields.some((f) => f.startsWith(`${prefix}.${id}`)));
    $(id).classList.toggle("is-uncertain", unsure);
    $(id).title = unsure ? "Unsicher gelesen – bitte mit dem Screenshot vergleichen" : "";
  }
  renderRounds();
}

function parseHistory(code) {
  if (code.length % 2) return null;
  const out = [];
  for (let i = 0; i < code.length; i += 2) {
    const wl = code[i].toUpperCase();
    const reason = code[i + 1].toLowerCase();
    if (!"WL".includes(wl) || !(reason in REASONS)) return null;
    out.push({ won: wl === "W", reason });
  }
  return out;
}

const historyCode = () => state.rounds.map((r) => (r.won ? "W" : "L") + r.reason).join("");

function renderRounds(syncCode = true) {
  const items = [];
  state.rounds.forEach((r, i) => {
    if (i === 12) items.push(el("li", { className: "half", textContent: "Halbzeit – Seitenwechsel" }));
    if (i === 24) items.push(el("li", { className: "half", textContent: "Overtime (keine Vorhersage)" }));
    const wl = el("button", {
      className: "round__wl", type: "button",
      textContent: r.won ? "Sieg" : "Niederlage",
      ariaLabel: `Runde ${i + 1}: ${r.won ? "Sieg" : "Niederlage"} des Gegners, umschalten`,
    });
    wl.addEventListener("click", () => { r.won = !r.won; renderRounds(); });
    const sel = el("select", { ariaLabel: `Runde ${i + 1}: Grund` },
      ...Object.entries(REASONS).map(([k, v]) => el("option", { value: k, textContent: v })));
    sel.value = r.reason;
    sel.addEventListener("change", () => { r.reason = sel.value; renderRounds(); });
    const li = el("li", { className: "round" }, el("span", { className: "round__no", textContent: `Runde ${i + 1}` }), wl, sel);
    li.dataset.won = r.won ? "1" : "0";
    items.push(li);
  });
  $("rounds").replaceChildren(...items);
  if (syncCode) { $("history").value = historyCode(); $("history").setCustomValidity(""); }
  renderSummary();
}

function renderSummary() {
  const n = state.rounds.length + 1;
  const wins = state.rounds.filter((r) => r.won).length;
  const parts = [`Runde ${n}`, `Gegner ${state.side}`, `Stand aus Sicht des Gegners ${wins}:${state.rounds.length - wins}`];
  if (n === 1 || n === 13) parts.push("Pistolenrunde: Full Buy unmöglich");
  if (n > 24) parts.push("Overtime: keine Vorhersage");
  $("summary").textContent = parts.join(" · ");
}

// ------------------------------------------------------------ 3. Prognose
async function runPredict() {
  const err = $("predict-error");
  err.hidden = true;
  const kills = $("kills").value, deaths = $("deaths").value;
  if (kills === "" || deaths === "") {
    err.textContent = "Bitte Kills und Deaths des Gegnerteams eintragen.";
    err.hidden = false;
    return;
  }
  $("btn-predict").disabled = true;
  try {
    const res = await api("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        side: state.side, history: historyCode(), kills: Number(kills), deaths: Number(deaths),
        reading: state.reading, store: state.store,
      }),
    });
    renderResult(res);
  } catch (e) {
    err.textContent = e.message;
    err.hidden = false;
  } finally {
    $("btn-predict").disabled = false;
  }
}

function renderResult(res) {
  $("result").hidden = false;
  const notes = res.notes.map((n) => el("li", { textContent: n }));
  if (res.skipped) {
    $("prob").textContent = "–";
    $("verdict-text").textContent = res.skipped;
    $("verdict-text").classList.remove("is-yes");
    $("verdict-sub").textContent = `Gegner ${res.side} · Runde ${res.round_number}`;
    $("meter").style.width = "0";
    $("rt").replaceChildren();
    $("agg").replaceChildren();
    $("result-meta").textContent = "";
  } else {
    const yes = res.possible;
    $("prob").textContent = pct(res.probability);
    $("verdict-text").textContent = yes ? "Full Buy möglich" : "kein Full Buy möglich";
    $("verdict-text").classList.toggle("is-yes", yes);
    $("verdict-sub").textContent =
      `Wahrscheinlichkeit, dass Gegner ${res.side} in Runde ${res.round_number} voll kaufen kann · Konfidenz ${pct(res.confidence)}`;
    $("meter").style.width = pct(res.probability).replace(" ", "");
    $("meter").parentElement.classList.toggle("is-yes", yes);
    $("rt").replaceChildren(...kvRows(res.rt));
    $("agg").replaceChildren(...kvRows(res.agg));
    const r = res.hist_rate;
    $("result-meta").textContent =
      `Modell full_buy_predictor v${res.model_version} (ROC-AUC Test ${res.model_auc}) · ` +
      `${res.store_label}: Quote für ${r.team_side}, Loss-Bonus-Stufe ${r.loss_bonus_level} aus ${r.n_rounds} Runden, ` +
      `Stand ${new Date(r.updated_at).toLocaleDateString("de-CH")}`;
  }
  $("result-notes").replaceChildren(...notes);
  $("result").scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
}

function kvRows(obj) {
  return Object.entries(obj).map(([k, v]) => el("tr", {},
    el("td", { textContent: k }),
    el("td", { textContent: FEATURE_LABELS[k] || "" }),
    el("td", { className: "num", textContent: Number.isInteger(v) ? String(v) : v.toFixed(2) })));
}

init();

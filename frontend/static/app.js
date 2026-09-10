/**
 * StartupGenie – Frontend Application
 * Handles form submission, API calls, and result rendering.
 */

const API_BASE = "/api/v1";

// ── DOM References ──────────────────────────────────────────
const form = document.getElementById("startup-form");
const generateBtn = document.getElementById("generate-btn");
const loadingState = document.getElementById("loading-state");
const emptyState = document.getElementById("empty-state");
const resultsArea = document.getElementById("results-area");
const errorState = document.getElementById("error-state");
const errorMsg = document.getElementById("error-msg");
const blueprintContent = document.getElementById("blueprint-content");
const sectionTabs = document.getElementById("section-tabs");
const ragContextList = document.getElementById("rag-context-list");
const ragCount = document.getElementById("rag-count");
const labelLegend = document.getElementById("label-legend");
const scoreOverall = document.getElementById("score-overall");

// ── Section Definitions ─────────────────────────────────────
const SECTIONS = [
  { key: "startup_idea_analysis",         label: "💡 Idea Analysis" },
  { key: "problem_statement",             label: "🎯 Problem" },
  { key: "proposed_solution",             label: "🔧 Solution" },
  { key: "target_customers",              label: "👥 Customers" },
  { key: "unique_value_proposition",      label: "✨ UVP" },
  { key: "business_model_canvas",         label: "🗺 BMC" },
  { key: "market_research",               label: "📊 Market" },
  { key: "competitor_analysis",           label: "🔍 Competitors" },
  { key: "revenue_model",                 label: "💰 Revenue" },
  { key: "estimated_startup_budget",      label: "📋 Budget" },
  { key: "go-to-market_strategy",         label: "🚀 GTM" },
  { key: "funding_opportunities",         label: "💼 Funding" },
  { key: "government_schemes",            label: "🏛 Gov Schemes" },
  { key: "legal_and_compliance_checklist",label: "⚖️ Legal" },
  { key: "incubators_and_accelerators",   label: "🏢 Incubators" },
  { key: "potential_investor_categories", label: "🤝 Investors" },
  { key: "30_60_90_day_action_plan",      label: "📅 Action Plan" },
  { key: "startup_readiness_score",       label: "⭐ Score" },
];

// State
let currentBlueprint = null;
let activeSection = null;

// ── Loading Steps ────────────────────────────────────────────
const LOADING_STEPS = [
  { id: "step-1", label: "Retrieving knowledge base context (RAG)..." },
  { id: "step-2", label: "Building prompt for IBM Granite..." },
  { id: "step-3", label: "Generating blueprint with IBM Granite..." },
  { id: "step-4", label: "Parsing and structuring results..." },
];

function renderLoadingSteps() {
  const container = document.getElementById("loading-steps");
  container.innerHTML = LOADING_STEPS.map((step, i) =>
    `<div class="loading-step" id="${step.id}">
      <div class="step-dot"></div>
      <span>${step.label}</span>
    </div>`
  ).join("");
}

function setLoadingStep(index) {
  LOADING_STEPS.forEach((step, i) => {
    const el = document.getElementById(step.id);
    if (!el) return;
    el.className = "loading-step";
    if (i < index) el.classList.add("done");
    else if (i === index) {
      el.classList.add("active");
      el.querySelector(".step-dot").classList.add("pulse");
    }
  });
}

// ── Label Highlighting ───────────────────────────────────────
function highlightLabels(text) {
  return text
    .replace(/\[RETRIEVED FACT\]/g, '<span class="lbl-retrieved">[RETRIEVED FACT]</span>')
    .replace(/\[ESTIMATE\]/g, '<span class="lbl-estimate">[ESTIMATE]</span>')
    .replace(/\[ASSUMPTION\]/g, '<span class="lbl-assumption">[ASSUMPTION]</span>')
    .replace(/\[RECOMMENDATION\]/g, '<span class="lbl-recommendation">[RECOMMENDATION]</span>')
    .replace(/^(#{1,3}\s.+)$/gm, '<span class="section-heading">$1</span>');
}

// ── Score Rendering ──────────────────────────────────────────
function renderScore(score) {
  scoreOverall.textContent = score.overall || 0;
  // colour the score
  const s = score.overall || 0;
  scoreOverall.style.color = s >= 70 ? "#3fb950" : s >= 50 ? "#d29922" : "#f85149";

  const dims = [
    { key: "idea_clarity",           label: "Idea Clarity",         max: 20 },
    { key: "market_opportunity",     label: "Market Opportunity",   max: 20 },
    { key: "technical_feasibility",  label: "Tech Feasibility",     max: 20 },
    { key: "business_model_strength",label: "Business Model",       max: 20 },
    { key: "team_resource_readiness",label: "Team Readiness",       max: 10 },
    { key: "regulatory_readiness",   label: "Regulatory",           max: 10 },
  ];

  const grid = document.getElementById("score-grid");
  grid.innerHTML = dims.map(dim => {
    const val = score[dim.key] || 0;
    const pct = Math.round((val / dim.max) * 100);
    const colour = pct >= 70 ? "#3fb950" : pct >= 50 ? "#d29922" : "#f85149";
    return `
      <div class="score-item">
        <div class="score-item-label">${dim.label}</div>
        <div class="score-bar-track">
          <div class="score-bar-fill" style="width:${pct}%;background:${colour}"></div>
        </div>
        <div class="score-value">${val}/${dim.max}</div>
      </div>`;
  }).join("");
}

// ── Label Legend ─────────────────────────────────────────────
function renderLabelLegend(counts) {
  labelLegend.innerHTML = `
    <div class="label-pill retrieved">🔎 Retrieved Facts: ${counts.retrieved_facts}</div>
    <div class="label-pill estimate">💲 Estimates: ${counts.estimates}</div>
    <div class="label-pill assumption">🤔 Assumptions: ${counts.assumptions}</div>
    <div class="label-pill recommendation">✅ Recommendations: ${counts.recommendations}</div>
  `;
}

// ── Section Tabs ─────────────────────────────────────────────
function renderSectionTabs(sections) {
  sectionTabs.innerHTML = SECTIONS.map((s, i) => {
    const hasContent = sections[s.key] && !sections[s.key].includes("not parsed");
    return `<button class="section-tab ${i === 0 ? "active" : ""}" 
              data-key="${s.key}" title="${s.label}"
              ${hasContent ? "" : 'style="opacity:0.5"'}>
              ${s.label}
            </button>`;
  }).join("");

  sectionTabs.querySelectorAll(".section-tab").forEach(btn => {
    btn.addEventListener("click", () => {
      sectionTabs.querySelectorAll(".section-tab").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      showSection(btn.dataset.key);
    });
  });
}

function showSection(key) {
  activeSection = key;
  const section = currentBlueprint.sections[key] || "(Section content not available)";
  blueprintContent.innerHTML = highlightLabels(section);
}

// ── RAG Context ──────────────────────────────────────────────
function renderRAGContext(docs) {
  ragCount.textContent = docs.length;
  if (!docs.length) {
    ragContextList.innerHTML = '<p style="color:var(--muted);font-size:0.85rem;">No context retrieved.</p>';
    return;
  }
  ragContextList.innerHTML = docs.map((doc, i) => {
    const sourceName = doc.source.replace(/_/g, " ").replace(".txt", "");
    const scoreColor = doc.score > 0.7 ? "#3fb950" : doc.score > 0.4 ? "#d29922" : "#8b949e";
    return `
      <div class="rag-chunk">
        <div class="rag-chunk-header">
          <span class="rag-source">${sourceName}</span>
          <span class="rag-score" style="color:${scoreColor}">score: ${doc.score.toFixed(3)}</span>
        </div>
        <div class="rag-chunk-text">${doc.content.substring(0, 200)}...</div>
      </div>`;
  }).join("");
}

// ── Full Blueprint View ──────────────────────────────────────
function showFullBlueprint() {
  blueprintContent.innerHTML = highlightLabels(currentBlueprint.blueprint_raw);
  sectionTabs.querySelectorAll(".section-tab").forEach(b => b.classList.remove("active"));
  blueprintContent.scrollIntoView({ behavior: "smooth" });
}

// ── Simulate loading progress ────────────────────────────────
function simulateLoadingProgress() {
  let step = 0;
  setLoadingStep(0);
  const interval = setInterval(() => {
    step++;
    if (step < LOADING_STEPS.length) {
      setLoadingStep(step);
    } else {
      clearInterval(interval);
    }
  }, 3000);
  return interval;
}

// ── Form Submission ──────────────────────────────────────────
form.addEventListener("submit", async (e) => {
  e.preventDefault();

  const formData = {
    idea: document.getElementById("idea").value.trim(),
    industry: document.getElementById("industry").value.trim(),
    target_customer: document.getElementById("target_customer").value.trim(),
    location: document.getElementById("location").value.trim() || "India",
    budget: document.getElementById("budget").value.trim(),
    stage: document.getElementById("stage").value,
  };

  // Validate
  if (!formData.idea || !formData.industry || !formData.target_customer || !formData.budget) {
    showError("Please fill in all required fields.");
    return;
  }

  // UI: loading state
  hideAll();
  loadingState.classList.add("active");
  generateBtn.disabled = true;
  generateBtn.querySelector(".btn-text").textContent = "Generating...";
  renderLoadingSteps();
  const progressInterval = simulateLoadingProgress();

  try {
    const response = await fetch(`${API_BASE}/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(formData),
    });

    clearInterval(progressInterval);
    setLoadingStep(LOADING_STEPS.length - 1);

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      const detail = err.detail || `HTTP error ${response.status}`;
      throw new Error(detail);
    }

    const data = await response.json();
    currentBlueprint = data;

    // Short delay for UX
    await new Promise(r => setTimeout(r, 500));

    // Render results
    hideAll();
    resultsArea.classList.add("active");
    renderScore(data.readiness_score);
    renderLabelLegend(data.label_counts);
    renderSectionTabs(data.sections);
    renderRAGContext(data.retrieved_context);

    // Show first section
    const firstKey = SECTIONS[0].key;
    showSection(firstKey);

  } catch (err) {
    clearInterval(progressInterval);
    hideAll();
    showError(err.message || "An unexpected error occurred. Please try again.");
  } finally {
    generateBtn.disabled = false;
    generateBtn.querySelector(".btn-text").textContent = "Generate Blueprint";
  }
});

// ── UI Helpers ───────────────────────────────────────────────
function hideAll() {
  loadingState.classList.remove("active");
  emptyState.style.display = "none";
  resultsArea.classList.remove("active");
  errorState.classList.remove("active");
}

// Patterns that indicate the user needs to configure their .env file
const CREDENTIAL_ERROR_PATTERNS = [
  "watsonx_api_key is not set",
  "watsonx_project_id is not set",
  "authentication failed",
  "iam token",
  "http 400",
  "api key is invalid",
  "placeholder",
];

function isCredentialError(message) {
  const lower = message.toLowerCase();
  return CREDENTIAL_ERROR_PATTERNS.some(p => lower.includes(p));
}

function showError(message) {
  errorState.classList.add("active");

  if (isCredentialError(message)) {
    errorMsg.innerHTML = `
      <strong>⚙️ IBM watsonx Credentials Not Configured</strong><br><br>
      ${escapeHtml(message)}<br><br>
      <strong>How to fix in 3 steps:</strong>
      <ol style="margin:8px 0 0 16px;line-height:2;">
        <li>Get your <strong>IBM Cloud API key</strong> →
          <a href="https://cloud.ibm.com/iam/apikeys" target="_blank" style="color:#60a5fa;">
            cloud.ibm.com/iam/apikeys
          </a>
        </li>
        <li>Get your <strong>watsonx.ai Project ID</strong> →
          open your project, go to <em>Manage → General</em>
        </li>
        <li>Edit your <code style="background:rgba(255,255,255,0.1);padding:1px 5px;border-radius:3px;">.env</code>
          file and set:
          <pre style="background:rgba(0,0,0,0.3);padding:8px 12px;border-radius:6px;margin-top:6px;font-size:0.82rem;line-height:1.6;">WATSONX_API_KEY=&lt;your-real-api-key&gt;
WATSONX_PROJECT_ID=&lt;your-project-id&gt;</pre>
          Then <strong>restart the server</strong>.
        </li>
      </ol>`;
  } else {
    errorMsg.textContent = "⚠️ " + message;
  }
}

function escapeHtml(str) {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

// Full blueprint button
document.getElementById("btn-full-blueprint")?.addEventListener("click", showFullBlueprint);

// ── Init ──────────────────────────────────────────────────────
renderLoadingSteps();

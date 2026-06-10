// =============================================================================
// AI Physio — app.js  (v2.0)
// Two-phase UX: Step 1 = detect fracture, Step 2 = clinical context + recovery
// =============================================================================

const API_BASE_URL = "http://127.0.0.1:8000";

// ---------------------------------------------------------------------------
// DOM refs — Auth & navigation
// ---------------------------------------------------------------------------
const apiStatus            = document.getElementById("apiStatus");
const authPanel            = document.getElementById("authPanel");
const casesPanel           = document.getElementById("casesPanel");
const casesList            = document.getElementById("casesList");
const clearAllCasesButton  = document.getElementById("clearAllCasesButton");
const userStatus           = document.getElementById("userStatus");
const userLabel            = document.getElementById("userLabel");
const authUsername         = document.getElementById("authUsername");
const authPassword         = document.getElementById("authPassword");
const authMessage          = document.getElementById("authMessage");
const loginButton          = document.getElementById("loginButton");
const registerButton       = document.getElementById("registerButton");
const logoutButton         = document.getElementById("logoutButton");
const themeToggle          = document.getElementById("themeToggle");
const navLinks             = document.querySelectorAll("[data-page-target]");
const pages                = document.querySelectorAll(".page");
const caseCountStat        = document.getElementById("caseCountStat");
const apiStat              = document.getElementById("apiStat");

// ---------------------------------------------------------------------------
// DOM refs — Upload & detection
// ---------------------------------------------------------------------------
const fileInput            = document.getElementById("xrayInput");
const dropzone             = document.getElementById("dropzone");
const previewWrap          = document.getElementById("previewWrap");
const imagePreview         = document.getElementById("imagePreview");
const originalStatus       = document.getElementById("originalStatus");
const detectButton         = document.getElementById("detectButton");
const message              = document.getElementById("message");
const detectPhase          = document.getElementById("detectPhase");

// ---------------------------------------------------------------------------
// DOM refs — Clinical intake
// ---------------------------------------------------------------------------
const clinicalIntakePanel  = document.getElementById("clinicalIntakePanel");
const bodyPart             = document.getElementById("bodyPart");
const treatmentStatus      = document.getElementById("treatmentStatus");
const recoveryStage        = document.getElementById("recoveryStage");
const swellingLevel        = document.getElementById("swellingLevel");
const mobilityStatus       = document.getElementById("mobilityStatus");
const doctorRestrictions   = document.getElementById("doctorRestrictions");
const exerciseApproval     = document.getElementById("exerciseApproval");
const painLevel            = document.getElementById("painLevel");
const painDisplay          = document.getElementById("painDisplay");
const patientAge           = document.getElementById("patientAge");
const patientSex           = document.getElementById("patientSex");
const symptomNotes         = document.getElementById("symptomNotes");
const analyzeButton        = document.getElementById("analyzeButton");
const clinicalMessage      = document.getElementById("clinicalMessage");
const recoveryBasis        = document.getElementById("recoveryBasis");
const recoveryBasisText    = document.getElementById("recoveryBasisText");
const additionalNotes      = document.getElementById("additionalNotes");

// ---------------------------------------------------------------------------
// DOM refs — Image viewer
// ---------------------------------------------------------------------------
const heatmapWrap          = document.getElementById("heatmapWrap");
const heatmapImage         = document.getElementById("heatmapImage");
const heatmapNote          = document.getElementById("heatmapNote");
const heatmapStatus        = document.getElementById("heatmapStatus");

// ---------------------------------------------------------------------------
// DOM refs — Result panel
// ---------------------------------------------------------------------------
const emptyState           = document.getElementById("emptyState");
const resultContent        = document.getElementById("resultContent");
const resultPanel          = document.getElementById("resultPanel");
const predictionLabel      = document.getElementById("predictionLabel");
const conditionLabel       = document.getElementById("conditionLabel");
const confidenceValue      = document.getElementById("confidenceValue");
const confidenceLevelValue = document.getElementById("confidenceLevelValue");
const probabilities        = document.getElementById("probabilities");
const predictionDisclaimer = document.getElementById("predictionDisclaimer");
const modelMode            = document.getElementById("modelMode");
const explainabilityMethod = document.getElementById("explainabilityMethod");
const decisionStatus       = document.getElementById("decisionStatus");
const safetyChecks         = document.getElementById("safetyChecks");
const qualityStatus        = document.getElementById("qualityStatus");
const decisionReasons      = document.getElementById("decisionReasons");
const riskBanner           = document.getElementById("riskBanner");
const riskLabel            = document.getElementById("riskLabel");
const doctorAdvice         = document.getElementById("doctorAdvice");
const downloadReportButton = document.getElementById("downloadReportButton");
const assistantOpenButton  = document.getElementById("assistantOpenButton");

// ---------------------------------------------------------------------------
// DOM refs — LLM panel
// ---------------------------------------------------------------------------
const llmPanel             = document.getElementById("llmPanel");
const llmStatus            = document.getElementById("llmStatus");
const llmSummary           = document.getElementById("llmSummary");
const llmConfidenceNote    = document.getElementById("llmConfidenceNote");
const llmConfidenceLevel   = document.getElementById("llmConfidenceLevel");
const llmHeatmapNote       = document.getElementById("llmHeatmapNote");
const llmNextSteps         = document.getElementById("llmNextSteps");
const llmSafetyNote        = document.getElementById("llmSafetyNote");

// ---------------------------------------------------------------------------
// DOM refs — Recovery guidance
// ---------------------------------------------------------------------------
const recommendations      = document.getElementById("recommendations");
const recommendationSummary = document.getElementById("recommendationSummary");
const severityBadge        = document.getElementById("severityBadge");
const exerciseGrid         = document.getElementById("exerciseGrid");
const exercisePlanNote     = document.getElementById("exercisePlanNote");
const dietaryTips          = document.getElementById("dietaryTips");
const nutritionPlanPanel   = document.getElementById("nutritionPlanPanel");
const nutritionSummary     = document.getElementById("nutritionSummary");
const nutritionTargets     = document.getElementById("nutritionTargets");
const mealGrid             = document.getElementById("mealGrid");
const nutritionAvoid       = document.getElementById("nutritionAvoid");
const doList               = document.getElementById("doList");
const avoidList            = document.getElementById("avoidList");
const followUpTimeline     = document.getElementById("followUpTimeline");
const careRedFlags         = document.getElementById("careRedFlags");
const eligibilityBanner    = document.getElementById("eligibilityBanner");
const eligibilityIcon      = document.getElementById("eligibilityIcon");
const eligibilityTitle     = document.getElementById("eligibilityTitle");
const eligibilityReason    = document.getElementById("eligibilityReason");
const redFlagAlert         = document.getElementById("redFlagAlert");
const redFlagList          = document.getElementById("redFlagList");
const treatmentNote        = document.getElementById("treatmentNote");
const treatmentNoteText    = document.getElementById("treatmentNoteText");
const recoveryJustification = document.getElementById("recoveryJustification");
const justificationText    = document.getElementById("justificationText");

// New sections
const recoveryRiskBanner   = document.getElementById("recoveryRiskBanner");
const riskLevelIcon        = document.getElementById("riskLevelIcon");
const riskLevelLabel       = document.getElementById("riskLevelLabel");
const riskLevelReasons     = document.getElementById("riskLevelReasons");
const clinicalContextUsed  = document.getElementById("clinicalContextUsed");
const clinicalContextGrid  = document.getElementById("clinicalContextGrid");
const recommendationReasoning = document.getElementById("recommendationReasoning");
const reasoningFactorList  = document.getElementById("reasoningFactorList");
const whenToSeekHelp       = document.getElementById("whenToSeekHelp");
const seekHelpList         = document.getElementById("seekHelpList");

// Dashboard stats
const fractureStat         = document.getElementById("fractureStat");
const normalStat           = document.getElementById("normalStat");
const avgConfidenceStat    = document.getElementById("avgConfidenceStat");
const highRiskStat         = document.getElementById("highRiskStat");
const lowRiskStatCard      = document.getElementById("lowRiskStatCard");
const moderateRiskStatCard = document.getElementById("moderateRiskStatCard");
const highRiskDistributionStatCard = document.getElementById("highRiskDistributionStatCard");
const lowRiskStat          = document.getElementById("lowRiskStat");
const moderateRiskStat     = document.getElementById("moderateRiskStat");
const highRiskDistributionStat = document.getElementById("highRiskDistributionStat");

// ---------------------------------------------------------------------------
// DOM refs — Assistant drawer
// ---------------------------------------------------------------------------
const assistantTopButton   = document.getElementById("assistantTopButton");
const assistantDrawer      = document.getElementById("assistantDrawer");
const assistantCloseButton = document.getElementById("assistantCloseButton");
const assistantThread      = document.getElementById("assistantThread");
const assistantQuestion    = document.getElementById("assistantQuestion");
const assistantSendButton  = document.getElementById("assistantSendButton");
const assistantStatus      = document.getElementById("assistantStatus");

// ---------------------------------------------------------------------------
// DOM refs — Metrics
// ---------------------------------------------------------------------------
const performanceCards     = document.getElementById("performanceCards");
const confusionMatrixImage = document.getElementById("confusionMatrixImage");
const matrixTabs           = document.querySelectorAll(".matrix-tab");

// ---------------------------------------------------------------------------
// Timeline steps
// ---------------------------------------------------------------------------
const timelineSteps = {
  upload:   document.getElementById("stepUpload"),
  predict:  document.getElementById("stepPredict"),
  context:  document.getElementById("stepContext"),
  recover:  document.getElementById("stepRecover"),
  done:     document.getElementById("stepDone"),
};

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
let selectedFile         = null;
let selectedImageUrl     = "";
let latestAnalysis       = null;
let latestPrediction     = null;   // Phase 1 result (fracture detection only)
let latestFileId         = null;
let latestPatientContext = {};
let authToken            = localStorage.getItem("ai-physio-token") || "";
let currentUser          = null;

// ---------------------------------------------------------------------------
// Utility helpers
// ---------------------------------------------------------------------------

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatPercent(value) {
  if (typeof value !== "number") return "-";
  return `${(value * 100).toFixed(2)}%`;
}

function getConfidenceLevel(value) {
  if (typeof value !== "number" || Number.isNaN(value)) return "-";
  const percent = value * 100;
  if (percent >= 95) return "Very High Confidence";
  if (percent >= 85) return "High Confidence";
  if (percent >= 70) return "Moderate Confidence";
  return "Low Confidence - Additional Clinical Review Recommended";
}

function titleCase(value) {
  return String(value || "-")
    .replaceAll("_", " ")
    .replace(/\b\w/g, (l) => l.toUpperCase());
}

function saferPredictionLabel(label) {
  const raw = String(label || "");
  if (raw.toLowerCase() === "fractured") return "Possible Fracture Pattern";
  return titleCase(raw);
}

function setMessage(text, isError = false) {
  message.textContent = text;
  message.classList.toggle("error", isError);
}

function setClinicalMessage(text, isError = false) {
  clinicalMessage.textContent = text;
  clinicalMessage.classList.toggle("error", isError);
}

function setTimeline(activeKeys) {
  const active = new Set(activeKeys);
  Object.entries(timelineSteps).forEach(([key, el]) => {
    if (el) el.classList.toggle("active", active.has(key));
  });
}

function showPage(pageId) {
  const target = document.getElementById(pageId) ? pageId : "dashboardPage";
  pages.forEach((page) => {
    page.hidden = page.id !== target;
    page.classList.toggle("active-page", page.id === target);
  });
  navLinks.forEach((link) => {
    link.classList.toggle("active", link.dataset.pageTarget === target);
  });
  if (window.location.hash !== `#${target}`) {
    history.replaceState(null, "", `#${target}`);
  }
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function pageFromHash() {
  return (window.location.hash || "#dashboardPage").replace("#", "");
}

function apiImageUrl(path) {
  if (!path) return "";
  if (/^https?:\/\//i.test(path) || path.startsWith("data:")) return path;
  return `${API_BASE_URL}${path.startsWith("/") ? path : `/${path}`}`;
}

// ---------------------------------------------------------------------------
// API fetch with auth
// ---------------------------------------------------------------------------

async function apiFetch(path, options = {}) {
  const headers = new Headers(options.headers || {});
  if (authToken && !headers.has("Authorization")) {
    headers.set("Authorization", `Bearer ${authToken}`);
  }
  try {
    return await fetch(`${API_BASE_URL}${path}`, { ...options, headers });
  } catch {
    throw new Error("Cannot reach the AI Physio API. Start the backend server, then try again.");
  }
}

// ---------------------------------------------------------------------------
// API status check
// ---------------------------------------------------------------------------

async function checkApiStatus() {
  try {
    const response = await fetch(`${API_BASE_URL}/health`);
    if (!response.ok) throw new Error("API unavailable");
    apiStatus.className = "api-status online";
    apiStatus.lastElementChild.textContent = "API online";
    if (apiStat) apiStat.textContent = "Online";
  } catch {
    apiStatus.className = "api-status offline";
    apiStatus.lastElementChild.textContent = "API offline";
    if (apiStat) apiStat.textContent = "Offline";
  }
}

// ---------------------------------------------------------------------------
// Theme
// ---------------------------------------------------------------------------

function applyTheme(theme) {
  document.body.dataset.theme = theme;
  themeToggle.textContent = theme === "dark" ? "Light Mode" : "Dark Mode";
  localStorage.setItem("ai-physio-theme", theme);
}

// ---------------------------------------------------------------------------
// File handling
// ---------------------------------------------------------------------------

function handleFile(file) {
  if (!file) return;
  selectedFile = file;
  detectButton.disabled = false;
  setMessage(`Selected: ${file.name}`);
  setTimeline(["upload"]);

  if (selectedImageUrl) URL.revokeObjectURL(selectedImageUrl);
  selectedImageUrl = URL.createObjectURL(file);
  imagePreview.src = selectedImageUrl;
  originalStatus.textContent = file.name;
  previewWrap.classList.add("has-image");

  // Reset phase 2 if user picks a new file
  clinicalIntakePanel.hidden = true;
  detectPhase.hidden = false;
  latestPrediction = null;
  latestFileId = null;

  // Clear assistant chat history
  if (assistantThread) {
    assistantThread.innerHTML = `
      <article class="assistant-message assistant-message-ai">
        <p>Ask about symptoms, safe exercise timing, report meaning, or recovery diet. I will use your latest analysis context including body part, treatment, stage, and exercise eligibility when answering.</p>
      </article>
    `;
  }
}

// ---------------------------------------------------------------------------
// Phase 1 — upload + predict
// ---------------------------------------------------------------------------

async function uploadImage(file) {
  const formData = new FormData();
  formData.append("file", file);
  const response = await apiFetch("/upload", { method: "POST", body: formData });
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || "Upload failed");
  }
  return response.json();
}

async function runDetection() {
  if (!selectedFile) return;

  detectButton.disabled = true;
  detectButton.textContent = "Uploading…";
  setMessage("Uploading X-ray…");
  setTimeline(["upload"]);

  try {
    // Step 1a: upload
    const uploadData = await uploadImage(selectedFile);
    latestFileId = uploadData.file_id;

    // Step 1b: predict
    detectButton.textContent = "Detecting…";
    setMessage("Running fracture detection…");
    setTimeline(["upload", "predict"]);

    const response = await apiFetch(`/predict/${latestFileId}`, { method: "POST" });
    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.detail || "Detection failed");
    }

    latestPrediction = await response.json();

    // Update image preview with the server URL
    imagePreview.src = apiImageUrl(`/uploads/${latestFileId}`);

    // Show detection result
    renderDetectionResult(latestPrediction);

    // Show clinical intake form
    detectPhase.hidden = true;
    clinicalIntakePanel.hidden = false;
    setTimeline(["upload", "predict", "context"]);
    setMessage("");

    // Scroll into view
    clinicalIntakePanel.scrollIntoView({ behavior: "smooth", block: "start" });

  } catch (err) {
    setMessage(err.message, true);
    detectButton.disabled = false;
    detectButton.textContent = "Step 1 — Detect Fracture";
  }
}

// ---------------------------------------------------------------------------
// Patient context collection
// ---------------------------------------------------------------------------

function numericFieldValue(input) {
  if (!input || !input.value.trim()) return null;
  const value = Number(input.value);
  return Number.isFinite(value) ? value : null;
}

function collectPatientContext() {
  const ctx = {
    body_part:            bodyPart.value || null,
    treatment_status:     treatmentStatus.value || null,
    recovery_stage:       recoveryStage.value || null,
    // BUG FIX: Ensure pain level is sent as an integer
    pain_level:           painLevel ? Math.round(Number(painLevel.value)) : null,
    swelling_level:       swellingLevel.value || null,
    mobility_status:      mobilityStatus.value || null,
    doctor_restrictions:  doctorRestrictions.value || null,
    exercise_approval:    exerciseApproval.value || null,
    // Optional
    age:                  numericFieldValue(patientAge),
    sex:                  patientSex.value || null,
    symptom_notes:        symptomNotes.value.trim() || null,
    additional_notes:     additionalNotes ? additionalNotes.value.trim() || null : null,
  };

  latestPatientContext = Object.fromEntries(
    Object.entries(ctx).filter(([, v]) => v !== null)
  );
  return latestPatientContext;
}

function validateClinicalContext() {
  const required = [
    bodyPart, treatmentStatus, recoveryStage, swellingLevel,
    mobilityStatus, doctorRestrictions, exerciseApproval
  ];
  return required.every((el) => el && el.value !== "");
}

function updateAnalyzeButton() {
  analyzeButton.disabled = !validateClinicalContext();
}

function buildRecoveryBasisText(ctx) {
  const parts = [];
  if (ctx.body_part)           parts.push(`Body Part: ${titleCase(ctx.body_part)}`);
  if (ctx.treatment_status)    parts.push(`Treatment: ${titleCase(ctx.treatment_status)}`);
  if (ctx.recovery_stage)      parts.push(`Stage: ${titleCase(ctx.recovery_stage)}`);
  if (ctx.pain_level != null)  parts.push(`Pain: ${ctx.pain_level}/10`);
  if (ctx.swelling_level)      parts.push(`Swelling: ${titleCase(ctx.swelling_level)}`);
  if (ctx.mobility_status)     parts.push(`Mobility: ${titleCase(ctx.mobility_status)}`);
  if (ctx.doctor_restrictions) parts.push(`Restrictions: ${titleCase(ctx.doctor_restrictions)}`);
  if (ctx.exercise_approval)   parts.push(`Exercise Approved: ${titleCase(ctx.exercise_approval)}`);
  return parts.join(" · ");
}

function updateRecoveryBasis() {
  const ctx = collectPatientContext();
  const text = buildRecoveryBasisText(ctx);
  if (text) {
    recoveryBasisText.textContent = text;
    recoveryBasis.hidden = false;
  } else {
    recoveryBasis.hidden = true;
  }
}

// ---------------------------------------------------------------------------
// Phase 2 — recovery
// ---------------------------------------------------------------------------

async function runRecovery() {
  if (!latestFileId) return;

  const patientContext = collectPatientContext();

  analyzeButton.disabled = true;
  analyzeButton.textContent = "Generating Recovery Plan…";
  setClinicalMessage("Running recovery engine…");
  setTimeline(["upload", "predict", "context", "recover"]);

  try {
    const response = await apiFetch(`/recovery/${latestFileId}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(patientContext),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.detail || "Recovery generation failed");
    }

    const data = await response.json();
    latestAnalysis = data;
    renderResults(data);
    setTimeline(["upload", "predict", "context", "recover", "done"]);
    setClinicalMessage("");

    // Persist to session storage
    persistLatestAnalysis(data);

    // Scroll to recommendations
    recommendations.scrollIntoView({ behavior: "smooth", block: "start" });

  } catch (err) {
    setClinicalMessage(err.message, true);
  } finally {
    analyzeButton.disabled = false;
    analyzeButton.textContent = "Step 2 — Generate Recovery Plan";
  }
}

// ---------------------------------------------------------------------------
// Session storage
// ---------------------------------------------------------------------------

function analysisForStorage(data, includeOverlay = true) {
  const copy = JSON.parse(JSON.stringify(data || {}));
  if (!includeOverlay && copy.prediction?.explainability?.overlay_image) {
    copy.prediction.explainability.overlay_image = "";
    copy.prediction.explainability.note =
      copy.prediction.explainability.note || "Heatmap was available in the original analysis.";
  }
  return copy;
}

function persistLatestAnalysis(data) {
  try {
    sessionStorage.setItem("ai-physio-latest-analysis", JSON.stringify(analysisForStorage(data)));
  } catch {
    try {
      sessionStorage.setItem("ai-physio-latest-analysis", JSON.stringify(analysisForStorage(data, false)));
    } catch {
      sessionStorage.removeItem("ai-physio-latest-analysis");
    }
  }
}

function restoreLatestAnalysis() {
  try {
    const saved = sessionStorage.getItem("ai-physio-latest-analysis");
    if (!saved) return;
    renderResults(JSON.parse(saved), false);
    setMessage("Restored the latest analysis from this browser session.");
  } catch {
    sessionStorage.removeItem("ai-physio-latest-analysis");
  }
}

// ---------------------------------------------------------------------------
// Render helpers
// ---------------------------------------------------------------------------

function renderProbabilities(classScores) {
  probabilities.innerHTML = "";
  Object.entries(classScores || {}).forEach(([label, score]) => {
    const item = document.createElement("div");
    item.innerHTML = `
      <div class="probability-label">
        <span>${titleCase(label)}</span>
        <strong>${formatPercent(score)}</strong>
      </div>
      <div class="bar"><span style="width: ${Math.max(0, Math.min(100, score * 100))}%"></span></div>
    `;
    probabilities.appendChild(item);
  });
}

function renderDetectionResult(prediction) {
  emptyState.hidden = true;
  resultContent.hidden = false;
  
  // Hide downstream report buttons until the full recovery plan is generated
  downloadReportButton.hidden = true;
  assistantOpenButton.hidden = true;

  const condition = prediction.condition || "";
  const decision  = prediction.decision  || {};
  const highRisk  = condition === "fracture" || decision.is_uncertain;

  riskBanner.className = `risk-banner ${highRisk ? "high-risk" : "low-risk"}`;
  
  // IMP FIX: Add low-confidence warning banner for uncertain predictions
  if (decision.is_uncertain || prediction.confidence < 0.70) {
    riskLabel.innerHTML = "<strong>⚠ Model confidence is below the reliability threshold.</strong> Clinical confirmation is strongly recommended before proceeding.";
  } else {
    riskLabel.textContent = highRisk ? "Possible Fracture — Clinical Review Recommended" : "No Fracture Pattern Detected";
  }

  if (condition === "normal") {
    doctorAdvice.textContent = "Generating preventive wellness plan...";
  } else {
    doctorAdvice.textContent = "Please complete the clinical context form below for personalised recovery guidance.";
  }

  predictionLabel.textContent  = saferPredictionLabel(prediction.label);
  conditionLabel.textContent   = titleCase(prediction.condition);
  confidenceValue.textContent  = formatPercent(prediction.confidence);
  if (confidenceLevelValue) confidenceLevelValue.textContent = getConfidenceLevel(prediction.confidence);
  predictionDisclaimer.textContent = prediction.disclaimer || "";
  modelMode.textContent        = `Model: ${titleCase(prediction.model_mode)}`;
  explainabilityMethod.textContent = `Explanation: ${(prediction.explainability || {}).method || "Unavailable"}`;
  decisionStatus.textContent   = `Decision: ${decision.is_uncertain ? "Review Recommended" : "Confident"}`;

  renderProbabilities(prediction.class_scores);

  const quality = prediction.image_quality || {};
  const reasons = (decision.reasons || []);
  const qualityWarnings = (quality.warnings || []);

  if (reasons.length || qualityWarnings.length) {
    safetyChecks.hidden = false;
    qualityStatus.textContent = quality.status
      ? `${titleCase(quality.status)}${qualityWarnings.length ? ` — ${qualityWarnings.join(", ")}` : ""}`
      : "No image-quality details returned.";
    decisionReasons.innerHTML = "";
    const visibleReasons = reasons.length ? reasons : [decision.recommended_action || "No review flags returned."];
    visibleReasons.forEach((r) => {
      const li = document.createElement("li");
      li.textContent = r;
      decisionReasons.appendChild(li);
    });
  } else {
    safetyChecks.hidden = true;
  }

  // Heatmap (if available from prediction)
  const explainability = prediction.explainability || {};
  if (explainability.overlay_image) {
    heatmapImage.src = explainability.overlay_image;
    heatmapNote.textContent = explainability.note || "Educational heatmap. Not a clinical localization result.";
    heatmapStatus.textContent = `${explainability.method} overlay`;
    heatmapWrap.classList.add("has-image");
  } else {
    heatmapImage.removeAttribute("src");
    heatmapNote.textContent = explainability.error || "Heatmap available after full analysis.";
    heatmapStatus.textContent = "Pending";
    heatmapWrap.classList.remove("has-image");
  }
}

function renderList(listEl, items, ordered = false) {
  listEl.innerHTML = "";
  (items || []).forEach((item) => {
    const li = document.createElement("li");
    li.textContent = item;
    listEl.appendChild(li);
  });
}

function renderRecoveryTimeline(physio, patientContext) {
  if (!followUpTimeline) return;
  followUpTimeline.innerHTML = "";

  const stages = [
    { key: "acute_phase", label: "Acute Phase" },
    { key: "early_recovery", label: "Early Recovery" },
    { key: "late_recovery", label: "Late Recovery" },
  ];
  const currentKey = patientContext?.recovery_stage || "acute_phase";
  const foundIndex = stages.findIndex((stage) => stage.key === currentKey);
  const currentIndex = foundIndex >= 0 ? foundIndex : 0;
  const currentStage = stages[currentIndex];
  const nextStage = stages[currentIndex + 1];
  const timelineItems = physio?.timeline || [];
  const longTermGoal = timelineItems[timelineItems.length - 1] || "Return to safe function with clinician clearance.";

  const progress = document.createElement("li");
  progress.className = "timeline-progression";
  progress.innerHTML = `
    <div class="stage-track" aria-label="Recovery stage progression">
      ${stages.map((stage, index) => `
        <span class="stage-pill ${index < currentIndex ? "complete" : ""} ${index === currentIndex ? "current" : ""}">
          <b>${index <= currentIndex ? "&check;" : "&rarr;"}</b>
          ${escapeHtml(stage.label)}
        </span>
      `).join("")}
    </div>
    <div class="stage-summary-grid">
      <span><small>Current Stage</small><strong>${escapeHtml(currentStage.label)}</strong></span>
      <span><small>Expected Next Stage</small><strong>${escapeHtml(nextStage ? nextStage.label : "Clinical sign-off / maintenance")}</strong></span>
      <span><small>Long-Term Goal</small><strong>${escapeHtml(longTermGoal)}</strong></span>
    </div>
  `;
  followUpTimeline.appendChild(progress);
}

function renderExerciseEligibility(eligibility) {
  if (!eligibility) return;

  const eligible = eligibility.eligible;
  const reason   = eligibility.reason;

  eligibilityBanner.className = `eligibility-banner ${eligible ? "eligible" : "ineligible"}`;
  eligibilityTitle.textContent = eligible
    ? "✓ Exercise Plan Generated"
    : "✗ Exercise Plan Not Generated";
  eligibilityReason.textContent = eligible
    ? "Body-part and stage-specific exercises are included below."
    : reason || "Exercise plan is not appropriate at this stage.";
}

function renderBodyPartGuidance(recoveryGuidance) {
  const physio = (recoveryGuidance || {}).physio_support || {};
  const patientContext = latestAnalysis?.patient_context || latestPatientContext || {};

  // Summary
  recommendationSummary.textContent = physio.personalized_intro || physio.summary || "";

  // Dos / daily focus
  renderList(doList, physio.daily_focus || []);

  // Avoids
  renderList(avoidList, physio.avoids || []);

  // Timeline
  renderRecoveryTimeline(physio, patientContext);

  // Red flags
  const redFlags = (recoveryGuidance || {}).red_flags || [];
  if (redFlags.length) {
    redFlagAlert.hidden = false;
    redFlagList.innerHTML = "";
    redFlags.forEach((flag) => {
      const li = document.createElement("li");
      li.textContent = flag;
      redFlagList.appendChild(li);
    });
  } else {
    redFlagAlert.hidden = true;
  }
}

function renderRecoveryJustification(patientContextData, prediction) {
  const conditionText  = prediction ? `Fracture detection: ${titleCase(prediction.condition)}` : "";
  const contextText    = buildRecoveryBasisText(patientContextData || {});
  const full = [conditionText, contextText].filter(Boolean).join(" · ");
  if (full) {
    justificationText.textContent =
      `Recovery recommendations are generated using: ${full}.`;
    recoveryJustification.hidden = false;
  } else {
    recoveryJustification.hidden = true;
  }
}

function renderExercises(exercisePlan, eligibility) {
  exerciseGrid.innerHTML = "";

  const exercises  = (exercisePlan || {}).exercises || [];
  const eligible   = (eligibility || {}).eligible;
  const treatNote  = (exercisePlan || {}).treatment_note;

  // Treatment note banner
  if (treatNote) {
    treatmentNoteText.textContent = treatNote;
    treatmentNote.hidden = false;
  } else {
    treatmentNote.hidden = true;
  }

  if (!eligible || exercises.length === 0) {
    const reason = (eligibility || {}).reason || "Exercise plan is not appropriate at this stage.";
    exercisePlanNote.textContent = reason;
    exerciseGrid.innerHTML = `<div class="exercise-empty">
      <p>${escapeHtml(reason)}</p>
      <p>Continue with the recovery guidance and nutrition plan above.</p>
    </div>`;
    return;
  }

  // Stage + body part label
  const planLabel = (exercisePlan.stage_label || "") + (exercisePlan.body_part_label ? ` · ${exercisePlan.body_part_label}` : "");
  exercisePlanNote.textContent = planLabel
    ? `${exercises.length} exercise${exercises.length !== 1 ? "s" : ""} for ${planLabel}`
    : `${exercises.length} exercise${exercises.length !== 1 ? "s" : ""} in your protocol`;

  exercises.forEach((exercise) => {
    const node   = document.createElement("article");
    node.className = "exercise";
    const media  = exercise.media || {};
    const image  = media.image_url
      ? `<img class="exercise-image" src="${escapeHtml(media.image_url)}" alt="${escapeHtml(exercise.name)} exercise reference" loading="lazy" />`
      : "";
    const video  = media.youtube_url
      ? `<a class="media-link" href="${escapeHtml(media.youtube_url)}" target="_blank" rel="noreferrer">Open YouTube form videos</a>`
      : "";
    node.innerHTML = `
      ${image}
      <h3>${escapeHtml(exercise.name)}</h3>
      <p>${escapeHtml(exercise.description)}</p>
      <dl>
        <dt>Sets</dt><dd>${exercise.sets ?? "-"}</dd>
        <dt>Reps</dt><dd>${exercise.reps ?? "-"}</dd>
        <dt>Duration</dt><dd>${exercise.duration_human ?? "-"}</dd>
        <dt>Frequency</dt><dd>${escapeHtml(exercise.frequency ?? "-")}</dd>
        <dt>Precaution</dt><dd>${escapeHtml(exercise.precautions ?? "-")}</dd>
      </dl>
      ${video}
    `;
    exerciseGrid.appendChild(node);
  });
}

function renderDietaryTips(items) {
  dietaryTips.innerHTML = "";
  (items || []).forEach((tip) => {
    const li = document.createElement("li");
    li.textContent = tip;
    dietaryTips.appendChild(li);
  });
}

function renderNutritionPlan(plan) {
  if (!plan) {
    nutritionPlanPanel.hidden = true;
    return;
  }

  nutritionPlanPanel.hidden = false;

  // Stage label in header
  const stageNote = plan.title ? ` — ${plan.title}` : "";
  nutritionSummary.textContent = plan.llm_note || plan.summary || "Recovery meals and hydration guidance.";

  nutritionTargets.innerHTML = "";
  (plan.daily_targets || []).forEach((target) => {
    const li = document.createElement("li");
    li.textContent = target;
    nutritionTargets.appendChild(li);
  });

  mealGrid.innerHTML = "";
  (plan.meals || []).forEach((meal) => {
    const node = document.createElement("article");
    node.className = "meal-card";
    const image = meal.image_url
      ? `<img src="${escapeHtml(meal.image_url)}" alt="${escapeHtml(meal.name)}" loading="lazy" />`
      : "";
    node.innerHTML = `
      ${image}
      <div>
        <span>${escapeHtml(meal.time || "Meal")}</span>
        <h3>${escapeHtml(meal.name || "Recovery meal")}</h3>
        <p>${escapeHtml(meal.why || "")}</p>
        <ul>${(meal.items || []).map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>
      </div>
    `;
    mealGrid.appendChild(node);
  });

  nutritionAvoid.innerHTML = "";
  (plan.avoid || []).forEach((item) => {
    const li = document.createElement("li");
    li.textContent = item;
    nutritionAvoid.appendChild(li);
  });
}

function renderLlmExplanation(explanation, prediction = null) {
  if (!explanation) {
    llmPanel.hidden = true;
    return;
  }
  llmPanel.hidden = false;
  llmPanel.classList.toggle("llm-live", Boolean(explanation.enabled));
  llmStatus.textContent = explanation.enabled
    ? `Generated by ${explanation.provider || "LLM"} (${explanation.model || "model"})`
    : `Local fallback: ${explanation.status || "LLM unavailable"}`;
  llmSummary.textContent        = explanation.summary        || "-";
  llmConfidenceNote.textContent = explanation.confidence_note || "-";
  if (llmConfidenceLevel) llmConfidenceLevel.textContent = getConfidenceLevel(prediction?.confidence);
  llmHeatmapNote.textContent    = explanation.heatmap_note   || "-";
  llmNextSteps.textContent      = explanation.next_steps     || "-";
  llmSafetyNote.textContent     = explanation.safety_note    || "-";
}

function renderOriginalImage(data) {
  const originalUrl = apiImageUrl(
    data.original_image_url || (data.file_id ? `/uploads/${data.file_id}` : "")
  );
  if (!originalUrl) return;
  if (selectedImageUrl) {
    URL.revokeObjectURL(selectedImageUrl);
    selectedImageUrl = "";
  }
  imagePreview.src = originalUrl;
  originalStatus.textContent = "Uploaded X-ray";
  previewWrap.classList.add("has-image");
}

function renderResults(data, persist = true) {
  latestAnalysis = data;
  const prediction        = data.prediction         || {};
  const recommendation    = data.recommendations    || {};
  const recoveryGuidance  = data.recovery_guidance  || {};
  const eligibility       = data.exercise_eligibility || (recoveryGuidance.exercise_eligibility) || {};
  const exercisePlan      = data.exercise_plan      || {};
  const explanation       = prediction.explainability || {};
  const patientCtx        = data.patient_context    || {};

  // Detection result (already shown if two-phase, re-render for restored sessions)
  renderDetectionResult(prediction);
  renderOriginalImage(data);

  // LLM explanation
  renderLlmExplanation(data.llm_explanation, prediction);

  // Show recommendations section
  recommendations.hidden = false;

  // Body-part guidance
  renderBodyPartGuidance(recoveryGuidance);

  // Recovery justification
  renderRecoveryJustification(patientCtx, prediction);

  // Severity badge
  const isNormalCase = (prediction.condition || "").toLowerCase() === "normal";
  severityBadge.textContent = isNormalCase
    ? "Severity Assessment: Not Applicable (No Fracture Detected)"
    : `Severity Assessment: ${titleCase(recommendation.severity || "review")}`;

  // Eligibility banner
  renderExerciseEligibility(eligibility);

  // Exercise plan
  renderExercises(exercisePlan, eligibility);

  // Dietary tips (legacy)
  renderDietaryTips(recommendation.dietary_tips);

  // Nutrition plan
  renderNutritionPlan(recoveryGuidance.nutrition_plan);

  // Report buttons
  downloadReportButton.hidden   = false;
  assistantOpenButton.hidden    = false;

  // Recovery Risk Level
  const riskData = (recoveryGuidance.recovery_risk_level) || null;
  renderRecoveryRiskLevel(riskData);

  // Clinical Context Used
  renderClinicalContextUsed(patientCtx);

  // Recommendation Reasoning
  renderRecommendationReasoning(data.recommendation_reasoning);

  // When to Seek Help
  renderWhenToSeekHelp(recoveryGuidance.when_to_seek_help);

  // Heatmap (full analysis may have overlay)
  if (explanation.overlay_image) {
    heatmapImage.src = explanation.overlay_image;
    heatmapNote.textContent = explanation.note || "Educational heatmap. Not a clinical localization result.";
    heatmapStatus.textContent = `${explanation.method} overlay`;
    heatmapWrap.classList.add("has-image");
  }

  if (persist) persistLatestAnalysis(data);
}

// ---------------------------------------------------------------------------
// New render helpers for enhanced sections
// ---------------------------------------------------------------------------

function renderRecoveryRiskLevel(risk) {
  if (!risk || !recoveryRiskBanner) return;
  const level = risk.level || "low";
  const colors = { high: "#ef4444", moderate: "#f59e0b", low: "#10b981" };
  const color = colors[level] || "#10b981";
  recoveryRiskBanner.hidden = false;
  recoveryRiskBanner.className = `recovery-risk-banner risk-${level}`;
  riskLevelIcon.style.color = color;
  riskLevelLabel.textContent = `Recovery Risk: ${risk.label || "Low Risk"}`;
  riskLevelLabel.style.color = color;
  riskLevelReasons.innerHTML = "";
  const reasons = risk.reasons || [];
  if (!reasons.length) {
    const li = document.createElement("li");
    li.textContent = "No elevated clinical risk factors were reported.";
    riskLevelReasons.appendChild(li);
    return;
  }
  reasons.forEach((r) => {
    const li = document.createElement("li");
    li.textContent = r;
    riskLevelReasons.appendChild(li);
  });
}

function renderClinicalContextUsed(ctx) {
  if (!ctx || !clinicalContextGrid) return;
  const fields = [
    { label: "Body Part", value: ctx.body_part },
    { label: "Treatment Status", value: ctx.treatment_status },
    { label: "Recovery Stage", value: ctx.recovery_stage },
    { label: "Pain Level", value: ctx.pain_level != null ? `${ctx.pain_level}/10` : null },
    { label: "Swelling Level", value: ctx.swelling_level },
    { label: "Mobility Status", value: ctx.mobility_status },
    { label: "Doctor Restrictions", value: ctx.doctor_restrictions },
    { label: "Exercise Approval", value: ctx.exercise_approval },
  ];
  const visible = fields.filter((f) => f.value != null && String(f.value).trim() !== "");
  if (!visible.length) { if (clinicalContextUsed) clinicalContextUsed.hidden = true; return; }
  if (clinicalContextUsed) clinicalContextUsed.hidden = false;
  clinicalContextGrid.innerHTML = "";
  visible.forEach(({ label, value }) => {
    const card = document.createElement("div");
    card.className = "context-field-card";
    card.innerHTML = `<span class="context-field-label">${label}</span><strong class="context-field-value">${escapeHtml(titleCase(String(value)))}</strong>`;
    clinicalContextGrid.appendChild(card);
  });
  if (ctx.additional_notes) {
    const noteCard = document.createElement("div");
    noteCard.className = "context-field-card full-width";
    noteCard.innerHTML = `<span class="context-field-label">Additional Notes</span><p class="context-field-value">${escapeHtml(ctx.additional_notes)}</p>`;
    clinicalContextGrid.appendChild(noteCard);
  }
}

function renderRecommendationReasoning(reasoning) {
  if (!reasoning || !reasoningFactorList) return;
  const factors = reasoning.factors_used || [];
  if (!factors.length) { if (recommendationReasoning) recommendationReasoning.hidden = true; return; }
  if (recommendationReasoning) recommendationReasoning.hidden = false;
  reasoningFactorList.innerHTML = "";
  factors.forEach((f) => {
    const li = document.createElement("li");
    li.textContent = f;
    reasoningFactorList.appendChild(li);
  });
}

function renderWhenToSeekHelp(items) {
  if (!items || !items.length || !seekHelpList) { if (whenToSeekHelp) whenToSeekHelp.hidden = true; return; }
  if (whenToSeekHelp) whenToSeekHelp.hidden = false;
  seekHelpList.innerHTML = "";
  items.forEach((item) => {
    const li = document.createElement("li");
    li.textContent = item;
    seekHelpList.appendChild(li);
  });
}

// ---------------------------------------------------------------------------
// Cases list
// ---------------------------------------------------------------------------

function renderCases(items) {
  if (!currentUser) {
    casesList.innerHTML = `<div class="empty-state compact">Log in to view saved cases.</div>`;
    if (caseCountStat) caseCountStat.textContent = "0";
    if (clearAllCasesButton) clearAllCasesButton.hidden = true;
    renderRiskDistributionStats([]);
    return;
  }

  if (clearAllCasesButton) {
    clearAllCasesButton.hidden = !items || items.length === 0;
  }

  if (!items || items.length === 0) {
    casesList.innerHTML = `<div class="empty-state compact">No saved cases yet.</div>`;
    if (caseCountStat) caseCountStat.textContent = "0";
    renderRiskDistributionStats([]);
    return;
  }

  if (caseCountStat) caseCountStat.textContent = String(items.length);

  // Compute dashboard stats
  const fractureCount = items.filter((c) => c.condition === "fracture").length;
  const normalCount   = items.filter((c) => c.condition === "normal").length;
  const highRiskCount = items.filter((c) => c.recovery_risk_level === "high").length;
  const confidences   = items.map((c) => c.confidence).filter((v) => v != null && !isNaN(v));
  const avgConf = confidences.length ? confidences.reduce((a, b) => a + b, 0) / confidences.length : null;

  if (fractureStat)      fractureStat.textContent      = String(fractureCount);
  if (normalStat)        normalStat.textContent        = String(normalCount);
  if (highRiskStat)      highRiskStat.textContent      = String(highRiskCount);
  if (avgConfidenceStat) avgConfidenceStat.textContent = avgConf != null ? formatPercent(avgConf) : "-";
  renderRiskDistributionStats(items);

  casesList.innerHTML = "";
  items.forEach((c) => {
    const node = document.createElement("article");
    node.className = "case-card";

    const eligible = c.exercise_eligible;
    const isEligible = eligible === 1 || eligible === true;
    const isNotEligible = eligible === 0 || eligible === false;
    const eligibleBadge = isEligible
      ? `<span class="badge badge-eligible">Exercise Eligible</span>`
      : isNotEligible
      ? `<span class="badge badge-ineligible">Exercise Not Eligible</span>`
      : "";

    const conditionCls = c.condition === "fracture" ? "condition-fracture" :
                         c.condition === "normal"   ? "condition-normal"   : "condition-uncertain";

    node.innerHTML = `
      <div class="case-header">
        <span class="case-condition ${conditionCls}">${titleCase(c.condition || c.prediction_label || "Unknown")}</span>
        <span class="case-confidence">${formatPercent(c.confidence)}</span>
        <button class="case-delete-btn" aria-label="Delete report" title="Delete report">
          <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path><line x1="10" y1="11" x2="10" y2="17"></line><line x1="14" y1="11" x2="14" y2="17"></line></svg>
        </button>
      </div>
      <div class="case-meta">
        ${c.body_part ? `<span>📍 ${titleCase(c.body_part)}</span>` : ""}
        ${c.recovery_stage ? `<span>🕐 ${titleCase(c.recovery_stage)}</span>` : ""}
        ${c.treatment_status ? `<span>🩹 ${titleCase(c.treatment_status)}</span>` : ""}
        ${eligibleBadge}
      </div>
      <span class="case-date">${new Date(c.created_at).toLocaleString()}</span>
    `;
    node.addEventListener("click", () => loadCase(c.id));
    const delBtn = node.querySelector(".case-delete-btn");
    if (delBtn) {
      delBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        deleteSingleCase(c.id);
      });
    }
    casesList.appendChild(node);
  });
}

function renderRiskDistributionStats(items) {
  const riskItems = (items || []).filter((c) => c.recovery_risk_level);
  const hasRiskData = riskItems.length > 0;
  [lowRiskStatCard, moderateRiskStatCard, highRiskDistributionStatCard].forEach((card) => {
    if (card) card.hidden = !hasRiskData;
  });
  if (!hasRiskData) return;

  const lowCount = riskItems.filter((c) => c.recovery_risk_level === "low").length;
  const moderateCount = riskItems.filter((c) => c.recovery_risk_level === "moderate").length;
  const highCount = riskItems.filter((c) => c.recovery_risk_level === "high").length;
  if (lowRiskStat) lowRiskStat.textContent = String(lowCount);
  if (moderateRiskStat) moderateRiskStat.textContent = String(moderateCount);
  if (highRiskDistributionStat) highRiskDistributionStat.textContent = String(highCount);
}

async function loadCases() {
  if (!currentUser) return;
  try {
    const response = await apiFetch("/cases");
    if (!response.ok) throw new Error("Failed to load cases");
    const data = await response.json();
    renderCases(data.cases || []);
  } catch {
    renderCases([]);
  }
}

async function loadCase(caseId) {
  try {
    const response = await apiFetch(`/cases/${caseId}`);
    if (!response.ok) throw new Error("Case not found");
    const data = await response.json();
    latestAnalysis = data;
    renderResults(data, false);
    showPage("analysisPage");
  } catch (err) {
    alert(`Could not load case: ${err.message}`);
  }
}

async function deleteAllCases() {
  if (!confirm("Delete all saved cases? This cannot be undone.")) return;
  try {
    const response = await apiFetch("/cases", { method: "DELETE" });
    if (!response.ok) throw new Error("Delete failed");
    await loadCases();
  } catch (err) {
    alert(`Could not delete cases: ${err.message}`);
  }
}

async function deleteSingleCase(caseId) {
  if (!confirm("Delete this report? This cannot be undone.")) return;
  try {
    const response = await apiFetch(`/cases/${caseId}`, { method: "DELETE" });
    if (!response.ok) throw new Error("Delete failed");
    await loadCases();
  } catch (err) {
    alert(`Could not delete report: ${err.message}`);
  }
}

// ---------------------------------------------------------------------------
// Reports
// ---------------------------------------------------------------------------

async function downloadReport(endpoint, filename) {
  if (!latestAnalysis) return;
  try {
    const response = await apiFetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(latestAnalysis),
    });
    if (!response.ok) throw new Error("Report generation failed");
    const blob = await response.blob();
    const url  = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href     = url;
    link.download = filename;
    link.click();
    URL.revokeObjectURL(url);
  } catch (err) {
    alert(`Could not download report: ${err.message}`);
  }
}

// ---------------------------------------------------------------------------
// Auth
// ---------------------------------------------------------------------------

function setAuthState(user, token = authToken) {
  currentUser = user;
  authToken   = token || "";
  if (authToken) {
    localStorage.setItem("ai-physio-token", authToken);
  } else {
    localStorage.removeItem("ai-physio-token");
  }
  authPanel.hidden   = Boolean(currentUser);
  userStatus.hidden  = !currentUser;
  casesPanel.hidden  = !currentUser;
  userLabel.textContent = currentUser ? currentUser.username : "";
}

async function authRequest(mode) {
  const username = authUsername.value.trim();
  const password = authPassword.value;
  if (!username || !password) {
    authMessage.textContent = "Enter a username and password.";
    authMessage.classList.add("error");
    return;
  }
  authMessage.textContent = mode === "login" ? "Logging in…" : "Creating account…";
  authMessage.classList.remove("error");

  const response = await apiFetch(`/auth/${mode}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  });
  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    authMessage.textContent = err.detail || "Authentication failed";
    authMessage.classList.add("error");
    return;
  }
  const data = await response.json();
  setAuthState(data.user, data.token);
  authPassword.value = "";
  authMessage.textContent = "";
  await loadCases();
}

async function restoreSession() {
  if (!authToken) { setAuthState(null, ""); return; }
  try {
    const response = await apiFetch("/auth/me");
    if (!response.ok) throw new Error("Session expired");
    const data = await response.json();
    setAuthState(data.user, authToken);
    await loadCases();
  } catch {
    setAuthState(null, "");
  }
}

async function logout() {
  try { await apiFetch("/auth/logout", { method: "POST" }); } catch {}
  setAuthState(null, "");
  casesList.innerHTML = `<div class="empty-state compact">Log in to view saved cases.</div>`;
}

// ---------------------------------------------------------------------------
// AI Assistant
// ---------------------------------------------------------------------------

function appendAssistantMessage(text, fromUser = false, meta = "") {
  const node = document.createElement("article");
  node.className = `assistant-message ${fromUser ? "assistant-message-user" : "assistant-message-ai"}`;
  node.innerHTML = `
    <p>${escapeHtml(text)}</p>
    ${meta ? `<span>${escapeHtml(meta)}</span>` : ""}
  `;
  assistantThread.appendChild(node);
  assistantThread.scrollTop = assistantThread.scrollHeight;
}

function openAssistant() {
  assistantDrawer.hidden = false;
  assistantQuestion.focus();
}

function closeAssistant() {
  assistantDrawer.hidden = true;
}

async function askAssistant() {
  const question = assistantQuestion.value.trim();
  if (!question) {
    assistantStatus.textContent = "Type a question first.";
    assistantStatus.classList.add("error");
    return;
  }
  appendAssistantMessage(question, true);
  assistantQuestion.value     = "";
  assistantSendButton.disabled = true;
  assistantSendButton.textContent = "Thinking…";
  assistantStatus.textContent = "Asking Groq assistant…";
  assistantStatus.classList.remove("error");

  try {
    const response = await apiFetch("/assistant", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        analysis:        latestAnalysis,
        patient_context: Object.keys(latestPatientContext).length ? latestPatientContext : null,
      }),
    });
    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.detail || "Assistant could not answer right now.");
    }
    const data = await response.json();
    const parts = [data.answer, data.safety_note, data.suggested_next_step].filter(Boolean);
    appendAssistantMessage(
      parts.join("\n\n"),
      false,
      data.enabled ? `Groq: ${data.model}` : `Local fallback: ${data.status}`
    );
    assistantStatus.textContent = data.enabled
      ? "Answered with Groq using the latest available context."
      : `Local fallback: ${data.status || "Groq unavailable"}.`;
  } catch (error) {
    appendAssistantMessage(error.message, false);
    assistantStatus.textContent = error.message;
    assistantStatus.classList.add("error");
  } finally {
    assistantSendButton.disabled = false;
    assistantSendButton.textContent = "Ask";
  }
}

// ---------------------------------------------------------------------------
// Metrics page
// ---------------------------------------------------------------------------

async function loadMetrics() {
  try {
    const response = await apiFetch("/metrics");
    if (!response.ok) throw new Error("Metrics unavailable");
    const data = await response.json();

    const allMetrics = [
      ...(data.validation ? Object.entries(data.validation).map(([k, v]) => ({ split: "validation", key: k, value: v })) : []),
      ...(data.test       ? Object.entries(data.test      ).map(([k, v]) => ({ split: "test",       key: k, value: v })) : []),
    ];

    performanceCards.innerHTML = "";
    allMetrics.slice(0, 8).forEach(({ split, key, value }) => {
      const card = document.createElement("div");
      card.className = "performance-card";
      const displayValue = typeof value === "number" ? `${(value * 100).toFixed(2)}%` : String(value);
      card.innerHTML = `
        <span class="perf-split">${split}</span>
        <strong class="perf-value">${displayValue}</strong>
        <span class="perf-label">${titleCase(key)}</span>
      `;
      performanceCards.appendChild(card);
    });

    // Load test confusion matrix by default
    loadConfusionMatrix("test");
  } catch {
    performanceCards.innerHTML = `<div class="empty-state compact">Metrics unavailable.</div>`;
  }
}

function loadConfusionMatrix(split) {
  confusionMatrixImage.src = `${API_BASE_URL}/confusion-matrix/${split}`;
  matrixTabs.forEach((tab) => {
    tab.classList.toggle("active", tab.dataset.split === split);
  });
}

// ---------------------------------------------------------------------------
// Event listeners
// ---------------------------------------------------------------------------

fileInput.addEventListener("change", (e) => handleFile(e.target.files[0]));

dropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropzone.classList.add("dragging");
});
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragging"));
dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("dragging");
  handleFile(e.dataTransfer.files[0]);
});

detectButton.addEventListener("click", runDetection);
analyzeButton.addEventListener("click", runRecovery);

// Pain slider live update
if (painLevel) {
  painLevel.addEventListener("input", () => {
    const val = Number(painLevel.value);
    painDisplay.textContent = val;
    // Colour the display red when high
    painDisplay.style.color = val >= 7 ? "#ef4444" : val >= 4 ? "#f59e0b" : "#10b981";
    updateRecoveryBasis();
  });
}

// Watch all required clinical dropdowns to enable the analyze button
const clinicalFields = [bodyPart, treatmentStatus, recoveryStage, swellingLevel, mobilityStatus, doctorRestrictions, exerciseApproval, painLevel];
clinicalFields.forEach((el) => {
  if (el) {
    el.addEventListener("change", () => {
      updateAnalyzeButton();
      updateRecoveryBasis();
    });
  }
});

loginButton.addEventListener("click",    () => authRequest("login").catch((e) => { authMessage.textContent = e.message; authMessage.classList.add("error"); }));
registerButton.addEventListener("click", () => authRequest("register").catch((e) => { authMessage.textContent = e.message; authMessage.classList.add("error"); }));
logoutButton.addEventListener("click",   logout);

if (clearAllCasesButton) clearAllCasesButton.addEventListener("click", deleteAllCases);

downloadReportButton.addEventListener("click",   () => downloadReport("/report",           "ai-physio-report.pdf"));

assistantTopButton.addEventListener("click",   openAssistant);
assistantOpenButton.addEventListener("click",  openAssistant);
assistantCloseButton.addEventListener("click", closeAssistant);
assistantSendButton.addEventListener("click",  askAssistant);
assistantQuestion.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    askAssistant();
  }
});

themeToggle.addEventListener("click", () => {
  const next = document.body.dataset.theme === "dark" ? "light" : "dark";
  applyTheme(next);
});

navLinks.forEach((link) => {
  link.addEventListener("click", (e) => {
    e.preventDefault();
    showPage(link.dataset.pageTarget);
    if (link.dataset.pageTarget === "metricsPage") loadMetrics();
  });
});

matrixTabs.forEach((tab) => {
  tab.addEventListener("click", () => loadConfusionMatrix(tab.dataset.split));
});

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

(async function init() {
  // Apply saved theme
  const savedTheme = localStorage.getItem("ai-physio-theme") || "dark";
  applyTheme(savedTheme);

  // Show page from hash
  showPage(pageFromHash());

  // API status
  await checkApiStatus();
  setInterval(checkApiStatus, 30_000);

  // Auth
  await restoreSession();

  // Load metrics if on metrics page
  if (pageFromHash() === "metricsPage") loadMetrics();

  // Restore session analysis
  restoreLatestAnalysis();
})();

const API_BASE_URL = "http://127.0.0.1:8000";

const apiStatus = document.getElementById("apiStatus");
const authPanel = document.getElementById("authPanel");
const casesPanel = document.getElementById("casesPanel");
const casesList = document.getElementById("casesList");
const clearAllCasesButton = document.getElementById("clearAllCasesButton");
const userStatus = document.getElementById("userStatus");
const userLabel = document.getElementById("userLabel");
const authUsername = document.getElementById("authUsername");
const authPassword = document.getElementById("authPassword");
const authMessage = document.getElementById("authMessage");
const loginButton = document.getElementById("loginButton");
const registerButton = document.getElementById("registerButton");
const logoutButton = document.getElementById("logoutButton");
const fileInput = document.getElementById("xrayInput");
const dropzone = document.getElementById("dropzone");
const patientAge = document.getElementById("patientAge");
const patientSex = document.getElementById("patientSex");
const bodyPart = document.getElementById("bodyPart");
const painLevel = document.getElementById("painLevel");
const swelling = document.getElementById("swelling");
const recentTrauma = document.getElementById("recentTrauma");
const symptomNotes = document.getElementById("symptomNotes");
const previewWrap = document.getElementById("previewWrap");
const imagePreview = document.getElementById("imagePreview");
const originalStatus = document.getElementById("originalStatus");
const analyzeButton = document.getElementById("analyzeButton");
const themeToggle = document.getElementById("themeToggle");
const message = document.getElementById("message");
const emptyState = document.getElementById("emptyState");
const resultContent = document.getElementById("resultContent");
const resultPanel = document.getElementById("resultPanel");
const predictionLabel = document.getElementById("predictionLabel");
const conditionLabel = document.getElementById("conditionLabel");
const confidenceValue = document.getElementById("confidenceValue");
const probabilities = document.getElementById("probabilities");
const predictionDisclaimer = document.getElementById("predictionDisclaimer");
const modelMode = document.getElementById("modelMode");
const explainabilityMethod = document.getElementById("explainabilityMethod");
const decisionStatus = document.getElementById("decisionStatus");
const bodyPartStatus = document.getElementById("bodyPartStatus");
const safetyChecks = document.getElementById("safetyChecks");
const qualityStatus = document.getElementById("qualityStatus");
const decisionReasons = document.getElementById("decisionReasons");
const heatmapWrap = document.getElementById("heatmapWrap");
const heatmapImage = document.getElementById("heatmapImage");
const heatmapNote = document.getElementById("heatmapNote");
const heatmapStatus = document.getElementById("heatmapStatus");
const downloadReportButton = document.getElementById("downloadReportButton");
const downloadDietPlanButton = document.getElementById("downloadDietPlanButton");
const assistantTopButton = document.getElementById("assistantTopButton");
const assistantOpenButton = document.getElementById("assistantOpenButton");
const assistantDrawer = document.getElementById("assistantDrawer");
const assistantCloseButton = document.getElementById("assistantCloseButton");
const assistantThread = document.getElementById("assistantThread");
const assistantQuestion = document.getElementById("assistantQuestion");
const assistantSendButton = document.getElementById("assistantSendButton");
const assistantStatus = document.getElementById("assistantStatus");
const recommendations = document.getElementById("recommendations");
const recommendationSummary = document.getElementById("recommendationSummary");
const severityBadge = document.getElementById("severityBadge");
const doctorAdvice = document.getElementById("doctorAdvice");
const exerciseGrid = document.getElementById("exerciseGrid");
const exercisePlanNote = document.getElementById("exercisePlanNote");
const dietaryTips = document.getElementById("dietaryTips");
const nutritionPlanPanel = document.getElementById("nutritionPlanPanel");
const nutritionSummary = document.getElementById("nutritionSummary");
const nutritionTargets = document.getElementById("nutritionTargets");
const mealGrid = document.getElementById("mealGrid");
const nutritionAvoid = document.getElementById("nutritionAvoid");
const riskBanner = document.getElementById("riskBanner");
const riskLabel = document.getElementById("riskLabel");
const llmPanel = document.getElementById("llmPanel");
const llmStatus = document.getElementById("llmStatus");
const llmSummary = document.getElementById("llmSummary");
const llmConfidenceNote = document.getElementById("llmConfidenceNote");
const llmHeatmapNote = document.getElementById("llmHeatmapNote");
const llmNextSteps = document.getElementById("llmNextSteps");
const llmSafetyNote = document.getElementById("llmSafetyNote");
const performanceCards = document.getElementById("performanceCards");
const confusionMatrixImage = document.getElementById("confusionMatrixImage");
const matrixTabs = document.querySelectorAll(".matrix-tab");
const navLinks = document.querySelectorAll("[data-page-target]");
const pages = document.querySelectorAll(".page");
const caseCountStat = document.getElementById("caseCountStat");
const apiStat = document.getElementById("apiStat");
const careRedFlags = document.getElementById("careRedFlags");
const carePainRule = document.getElementById("carePainRule");
const careFollowUp = document.getElementById("careFollowUp");
const careActivity = document.getElementById("careActivity");
const doList = document.getElementById("doList");
const avoidList = document.getElementById("avoidList");
const followUpTimeline = document.getElementById("followUpTimeline");
const timelineSteps = {
  upload: document.getElementById("stepUpload"),
  preprocess: document.getElementById("stepPreprocess"),
  predict: document.getElementById("stepPredict"),
  explain: document.getElementById("stepExplain"),
  recommend: document.getElementById("stepRecommend"),
};

let selectedFile = null;
let selectedImageUrl = "";
let latestAnalysis = null;
let authToken = localStorage.getItem("ai-physio-token") || "";
let currentUser = null;
let latestPatientContext = {};

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

function setMessage(text, isError = false) {
  message.textContent = text;
  message.classList.toggle("error", isError);
}

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
  assistantQuestion.value = "";
  assistantSendButton.disabled = true;
  assistantSendButton.textContent = "Thinking...";
  assistantStatus.textContent = "Asking Groq assistant...";
  assistantStatus.classList.remove("error");

  try {
    const response = await apiFetch("/assistant", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        question,
        analysis: latestAnalysis,
        patient_context: Object.keys(latestPatientContext).length ? latestPatientContext : null,
      }),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.detail || "Assistant could not answer right now.");
    }

    const data = await response.json();
    const answerParts = [data.answer, data.safety_note, data.suggested_next_step].filter(Boolean);
    appendAssistantMessage(answerParts.join("\n\n"), false, data.enabled ? `Groq: ${data.model}` : `Local fallback: ${data.status}`);
    assistantStatus.textContent = data.enabled
      ? "Answered with Groq using the latest available context."
      : `Answered with local fallback: ${data.status || "Groq unavailable"}.`;
  } catch (error) {
    appendAssistantMessage(error.message, false);
    assistantStatus.textContent = error.message;
    assistantStatus.classList.add("error");
  } finally {
    assistantSendButton.disabled = false;
    assistantSendButton.textContent = "Ask";
  }
}

function setAuthMessage(text, isError = false) {
  authMessage.textContent = text;
  authMessage.classList.toggle("error", isError);
}

function setTimeline(activeKeys) {
  const active = new Set(activeKeys);
  Object.entries(timelineSteps).forEach(([key, element]) => {
    element.classList.toggle("active", active.has(key));
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

function formatPercent(value) {
  if (typeof value !== "number") return "-";
  return `${(value * 100).toFixed(2)}%`;
}

function titleCase(value) {
  return String(value || "-")
    .replaceAll("_", " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function saferPredictionLabel(label) {
  const raw = String(label || "");
  if (raw.toLowerCase() === "fractured") {
    return "Possible Fracture Pattern";
  }
  return titleCase(raw);
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

async function apiFetch(path, options = {}) {
  const headers = new Headers(options.headers || {});
  if (authToken && !headers.has("Authorization")) {
    headers.set("Authorization", `Bearer ${authToken}`);
  }

  try {
    return await fetch(`${API_BASE_URL}${path}`, {
      ...options,
      headers,
    });
  } catch {
    throw new Error("Cannot reach the AI Physio API. Start the backend server, then try again.");
  }
}

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

function applyTheme(theme) {
  document.body.dataset.theme = theme;
  themeToggle.textContent = theme === "dark" ? "Light Mode" : "Dark Mode";
  localStorage.setItem("ai-physio-theme", theme);
}

function handleFile(file) {
  if (!file) return;

  selectedFile = file;
  analyzeButton.disabled = false;
  setMessage(`Selected: ${file.name}`);
  setTimeline(["upload"]);

  if (selectedImageUrl) URL.revokeObjectURL(selectedImageUrl);
  selectedImageUrl = URL.createObjectURL(file);
  imagePreview.src = selectedImageUrl;
  originalStatus.textContent = file.name;
  previewWrap.classList.add("has-image");
}

function setAuthState(user, token = authToken) {
  currentUser = user;
  authToken = token || "";

  if (authToken) {
    localStorage.setItem("ai-physio-token", authToken);
  } else {
    localStorage.removeItem("ai-physio-token");
  }

  authPanel.hidden = Boolean(currentUser);
  userStatus.hidden = !currentUser;
  casesPanel.hidden = !currentUser;
  userLabel.textContent = currentUser ? currentUser.username : "";
}

async function authRequest(mode) {
  const username = authUsername.value.trim();
  const password = authPassword.value;

  if (!username || !password) {
    setAuthMessage("Enter a username and password.", true);
    return;
  }

  setAuthMessage(mode === "login" ? "Logging in..." : "Creating account...");
  const response = await apiFetch(`/auth/${mode}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ username, password }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || "Authentication failed");
  }

  const data = await response.json();
  setAuthState(data.user, data.token);
  authPassword.value = "";
  setAuthMessage("");
  await loadCases();
}

async function restoreSession() {
  if (!authToken) {
    setAuthState(null, "");
    return;
  }

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
  try {
    await apiFetch("/auth/logout", { method: "POST" });
  } catch {
    // Local logout should still clear the browser session.
  }
  setAuthState(null, "");
  casesList.innerHTML = `<div class="empty-state compact">Log in to view saved cases.</div>`;
}

fileInput.addEventListener("change", (event) => {
  handleFile(event.target.files[0]);
});

dropzone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropzone.classList.add("dragging");
});

dropzone.addEventListener("dragleave", () => {
  dropzone.classList.remove("dragging");
});

dropzone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropzone.classList.remove("dragging");
  handleFile(event.dataTransfer.files[0]);
});

async function uploadImage(file) {
  const formData = new FormData();
  formData.append("file", file);

  const response = await apiFetch("/upload", {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || "Upload failed");
  }

  return response.json();
}

function numericFieldValue(input) {
  if (!input.value.trim()) return null;
  const value = Number(input.value);
  return Number.isFinite(value) ? value : null;
}

function collectPatientContext() {
  const context = {
    age: numericFieldValue(patientAge),
    sex: patientSex.value || null,
    body_part: bodyPart.value || null,
    pain_level: numericFieldValue(painLevel),
    swelling: swelling.checked ? true : null,
    recent_trauma: recentTrauma.checked ? true : null,
    symptom_notes: symptomNotes.value.trim() || null,
  };

  latestPatientContext = Object.fromEntries(Object.entries(context).filter(([, value]) => value !== null));
  return latestPatientContext;
}

async function analyzeImage(fileId, patientContext) {
  const response = await apiFetch(`/analyze/${fileId}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(patientContext || {}),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || "Analysis failed");
  }

  return response.json();
}

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

function renderExercises(items) {
  exerciseGrid.innerHTML = "";

  if (!items || items.length === 0) {
    exerciseGrid.innerHTML = `<div class="exercise"><h3>No protocol found</h3><p>Please consult a physiotherapist for a personalized plan.</p></div>`;
    return;
  }

  items.forEach((exercise) => {
    const node = document.createElement("article");
    node.className = "exercise";
    const media = exercise.media || {};
    const image = media.image_url
      ? `<img class="exercise-image" src="${escapeHtml(media.image_url)}" alt="${escapeHtml(exercise.name)} exercise reference" loading="lazy" />`
      : "";
    const video = media.youtube_url
      ? `<a class="media-link" href="${escapeHtml(media.youtube_url)}" target="_blank" rel="noreferrer">Open YouTube form videos</a>`
      : "";
    node.innerHTML = `
      ${image}
      <h3>${exercise.name}</h3>
      <p>${exercise.description}</p>
      <dl>
        <dt>Sets</dt><dd>${exercise.sets ?? "-"}</dd>
        <dt>Reps</dt><dd>${exercise.reps ?? "-"}</dd>
        <dt>Duration</dt><dd>${exercise.duration_human ?? "-"}</dd>
        <dt>Frequency</dt><dd>${exercise.frequency ?? "-"}</dd>
        <dt>Precaution</dt><dd>${exercise.precautions ?? "-"}</dd>
      </dl>
      ${video}
    `;
    exerciseGrid.appendChild(node);
  });
}

function updateExercisePlanNote(prediction) {
  const condition = prediction.condition || "";
  const decision = prediction.decision || {};
  const highRisk = ["fracture", "joint_dislocation", "bone_tumor"].includes(condition);

  if (highRisk || decision.is_uncertain) {
    exercisePlanNote.textContent = "Defer exercises until the image and symptoms are reviewed by a qualified clinician.";
    return;
  }

  exercisePlanNote.textContent = "Gentle, pain-free movement options for low-risk screening results.";
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
  nutritionSummary.textContent = plan.summary || "Recovery meals and hydration guidance.";

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

function renderSafetyDetails(data) {
  const prediction = data.prediction || {};
  const decision = prediction.decision || {};
  const quality = prediction.image_quality || {};
  const bodyPart = data.body_part_detection || {};
  const reasons = decision.reasons || [];
  const qualityWarnings = quality.warnings || [];

  decisionStatus.textContent = `Decision: ${decision.is_uncertain ? "Review Recommended" : "Confident"}`;
  bodyPartStatus.textContent = bodyPart.body_part
    ? `Body part: ${titleCase(bodyPart.body_part)} (${bodyPart.source || "provided"})`
    : "Body part: Not detected";

  safetyChecks.hidden = false;
  qualityStatus.textContent = quality.status
    ? `${titleCase(quality.status)}${qualityWarnings.length ? ` - ${qualityWarnings.join(" ")}` : ""}`
    : "No image-quality details returned.";

  decisionReasons.innerHTML = "";
  const visibleReasons = reasons.length ? reasons : [decision.recommended_action || "No review flags returned."];
  visibleReasons.forEach((reason) => {
    const item = document.createElement("li");
    item.textContent = reason;
    decisionReasons.appendChild(item);
  });
}

function updateRiskBanner(condition, severity, doctorText) {
  const highRisk = ["fracture", "joint_dislocation", "bone_tumor"].includes(condition);
  riskBanner.className = `risk-banner ${highRisk ? "high-risk" : "low-risk"}`;
  riskLabel.textContent = highRisk ? "Clinical Review Recommended" : "Low-Risk Screening Result";
  doctorAdvice.textContent = doctorText || "Consult a qualified clinician for confirmation.";
  severityBadge.textContent = `Severity: ${titleCase(severity)}`;
}

function renderList(list, items) {
  list.innerHTML = "";
  items.forEach((item) => {
    const li = document.createElement("li");
    li.textContent = item;
    list.appendChild(li);
  });
}

function patientContextFromAnalysis(data) {
  return data.patient_context || data.context || latestPatientContext || {};
}

function renderCarePlan(data, prediction, recommendation) {
  const condition = prediction.condition || "";
  const decision = prediction.decision || {};
  const context = patientContextFromAnalysis(data);
  const highRisk = ["fracture", "joint_dislocation", "bone_tumor"].includes(condition);
  const confidence = formatPercent(prediction.confidence);
  const pain = typeof context.pain_level === "number" ? context.pain_level : null;
  const bodyPartText = context.body_part ? titleCase(context.body_part) : "affected area";
  const needsReview = highRisk || decision.is_uncertain || pain >= 6 || context.swelling || context.recent_trauma;

  careRedFlags.textContent = needsReview
    ? "Seek clinical review for severe pain, swelling, deformity, numbness, color change, fever, or loss of function."
    : "Monitor for worsening pain, swelling, numbness, weakness, or reduced movement.";
  carePainRule.textContent = "Continue at 0-3/10 pain, reduce at 4-5/10, stop and seek review at 6+/10.";
  careFollowUp.textContent = highRisk
    ? "Arrange medical confirmation today before loading or exercise progression."
    : "Reassess in 48-72 hours and progress only if pain, swelling, and function improve.";
  careActivity.textContent = highRisk
    ? `Protect the ${bodyPartText.toLowerCase()} and avoid weight-bearing or resistance until cleared.`
    : `Begin gentle, pain-free movement for the ${bodyPartText.toLowerCase()} as tolerated. Model confidence: ${confidence}.`;

  const doItems = [
    "Keep the area protected and comfortable.",
    "Use elevation and rest when swelling or pain increases.",
    "Track pain, swelling, and movement daily.",
  ];
  const avoidItems = [
    "Avoid heavy loading, sport, or forceful stretching too early.",
    "Avoid massage or heat over a suspected fracture site.",
    "Do not ignore worsening symptoms after activity.",
  ];
  const timelineItems = highRisk
    ? [
        "Today: get clinical confirmation.",
        "After clearance: begin gentle mobility.",
        "Follow-up: progress strength only when pain and swelling settle.",
      ]
    : [
        "Today: start gentle range-of-motion if comfortable.",
        "48-72 hours: reassess pain, swelling, and function.",
        "1-2 weeks: progress strength if symptoms keep improving.",
      ];

  if (decision.is_uncertain) {
    doItems.unshift("Treat this as uncertain and prioritize human review.");
  }
  if (context.recent_trauma) {
    avoidItems.unshift("Avoid loading after recent trauma until serious injury is ruled out.");
  }
  if (pain >= 6) {
    doItems.unshift("Stop exercises and seek review because pain is high.");
  }

  renderList(doList, doItems);
  renderList(avoidList, avoidItems);
  renderList(followUpTimeline, timelineItems);
}

function renderLlmExplanation(explanation) {
  if (!explanation) {
    llmPanel.hidden = true;
    return;
  }

  llmPanel.hidden = false;
  llmPanel.classList.toggle("llm-live", Boolean(explanation.enabled));
  llmStatus.textContent = explanation.enabled
    ? `Generated by ${explanation.provider || "LLM"} (${explanation.model || "model"})`
    : `Local fallback: ${explanation.status || "LLM unavailable"}`;
  llmSummary.textContent = explanation.summary || "-";
  llmConfidenceNote.textContent = explanation.confidence_note || "-";
  llmHeatmapNote.textContent = explanation.heatmap_note || "-";
  llmNextSteps.textContent = explanation.next_steps || "-";
  llmSafetyNote.textContent = explanation.safety_note || "-";
}

function apiImageUrl(path) {
  if (!path) return "";
  if (/^https?:\/\//i.test(path) || path.startsWith("data:")) return path;
  return `${API_BASE_URL}${path.startsWith("/") ? path : `/${path}`}`;
}

function renderOriginalImage(data) {
  const originalUrl = apiImageUrl(data.original_image_url || (data.file_id ? `/uploads/${data.file_id}` : ""));
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
  const prediction = data.prediction || {};
  const recommendation = data.recommendations || {};
  const explanation = prediction.explainability || {};

  emptyState.hidden = true;
  resultContent.hidden = false;
  recommendations.hidden = false;
  downloadReportButton.hidden = false;
  downloadDietPlanButton.hidden = false;
  assistantOpenButton.hidden = false;
  renderOriginalImage(data);

  predictionLabel.textContent = saferPredictionLabel(prediction.label);
  conditionLabel.textContent = titleCase(prediction.condition);
  confidenceValue.textContent = formatPercent(prediction.confidence);
  predictionDisclaimer.textContent = prediction.disclaimer || "";
  modelMode.textContent = `Model: ${titleCase(prediction.model_mode)}`;
  explainabilityMethod.textContent = `Explanation: ${explanation.method || "Unavailable"}`;
  renderSafetyDetails(data);
  renderProbabilities(prediction.class_scores);
  updateRiskBanner(prediction.condition, recommendation.severity, recommendation.when_to_see_doctor);
  renderCarePlan(data, prediction, recommendation);
  updateExercisePlanNote(prediction);
  renderLlmExplanation(data.llm_explanation);

  if (explanation.overlay_image) {
    heatmapImage.src = explanation.overlay_image;
    heatmapNote.textContent = explanation.note || "Educational heatmap. Not a clinical localization result.";
    heatmapStatus.textContent = `${explanation.method} overlay`;
    heatmapWrap.classList.add("has-image");
  } else {
    heatmapImage.removeAttribute("src");
    heatmapNote.textContent = explanation.error || "Heatmap is unavailable for this prediction.";
    heatmapStatus.textContent = "Unavailable";
    heatmapWrap.classList.remove("has-image");
  }

  recommendationSummary.textContent = recommendation.summary || "";
  renderExercises(recommendation.exercises);
  renderDietaryTips(recommendation.dietary_tips);
  renderNutritionPlan(data.recovery_guidance?.nutrition_plan);
  setTimeline(["upload", "preprocess", "predict", "explain", "recommend"]);

  if (persist) {
    persistLatestAnalysis(data);
  }
}

function renderCases(items) {
  if (!currentUser) {
    casesList.innerHTML = `<div class="empty-state compact">Log in to view saved cases.</div>`;
    if (caseCountStat) caseCountStat.textContent = "0";
    if (clearAllCasesButton) clearAllCasesButton.hidden = true;
    return;
  }

  if (clearAllCasesButton) {
    clearAllCasesButton.hidden = !items || items.length === 0;
  }

  if (!items || items.length === 0) {
    casesList.innerHTML = `<div class="empty-state compact">No saved cases yet.</div>`;
    if (caseCountStat) caseCountStat.textContent = "0";
    return;
  }

  if (caseCountStat) caseCountStat.textContent = String(items.length);

  casesList.innerHTML = items
    .map((item) => {
      const label = saferPredictionLabel(item.prediction_label);
      const confidence = formatPercent(item.confidence);
      const body = item.body_part ? ` - ${escapeHtml(titleCase(item.body_part))}` : "";
      const created = new Date(item.created_at).toLocaleString();
      return `
        <article class="case-row">
          <div>
            <strong>${escapeHtml(label)} - ${escapeHtml(confidence)}</strong>
            <span>${escapeHtml(created)}${body}</span>
          </div>
          <button class="secondary-action compact-action case-open" type="button" data-case-id="${item.id}">
            Open
          </button>
          <button class="danger-action compact-action case-delete" type="button" data-case-id="${item.id}">
            Delete
          </button>
        </article>
      `;
    })
    .join("");
}

async function loadCases() {
  if (!currentUser) return;

  try {
    const response = await apiFetch("/cases");
    if (!response.ok) throw new Error("Could not load cases");
    const data = await response.json();
    renderCases(data.cases);
  } catch (error) {
    casesList.innerHTML = `<div class="empty-state compact">${escapeHtml(error.message)}</div>`;
  }
}

async function openCase(caseId) {
  const response = await apiFetch(`/cases/${caseId}`);
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || "Could not open case");
  }
  const data = await response.json();
  showPage("analysisPage");
  renderResults(data);
  setMessage(`Opened saved case #${caseId}.`);
  resultPanel.scrollIntoView({ behavior: "smooth", block: "start" });
}

async function deleteCase(caseId) {
  const confirmed = window.confirm("Delete this saved case?");
  if (!confirmed) return;

  const response = await apiFetch(`/cases/${caseId}`, {
    method: "DELETE",
  });
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || "Could not delete case");
  }

  if (latestAnalysis?.case_id && String(latestAnalysis.case_id) === String(caseId)) {
    latestAnalysis = null;
    sessionStorage.removeItem("ai-physio-latest-analysis");
  }

  await loadCases();
  setMessage(`Deleted saved case #${caseId}.`);
}

async function clearAllCases() {
  const confirmed = window.confirm(
    "Are you sure you want to permanently delete all saved reports? This action cannot be undone."
  );
  if (!confirmed) return;

  const response = await apiFetch("/cases", {
    method: "DELETE",
  });
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || "Could not delete all cases");
  }

  if (latestAnalysis?.case_id) {
    latestAnalysis = null;
    sessionStorage.removeItem("ai-physio-latest-analysis");
  }

  await loadCases();
  setMessage("Cleared all saved reports.");
}

async function downloadReport() {
  if (!latestAnalysis) return;
  downloadReportButton.disabled = true;
  downloadReportButton.textContent = "Preparing PDF...";

  try {
    const response = await apiFetch("/report", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(latestAnalysis),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.detail || "Could not generate PDF report.");
    }

    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `ai-physio-report-${Date.now()}.pdf`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
    setMessage("PDF report downloaded.");
  } catch (error) {
    setMessage(error.message, true);
  } finally {
    downloadReportButton.disabled = false;
    downloadReportButton.textContent = "Download Full PDF Report";
  }
}

async function downloadDietPlan() {
  if (!latestAnalysis) return;
  downloadDietPlanButton.disabled = true;
  downloadDietPlanButton.textContent = "Preparing Diet PDF...";

  try {
    const response = await apiFetch("/nutrition-report", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(latestAnalysis),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.detail || "Could not generate diet plan.");
    }

    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `ai-physio-diet-plan-${Date.now()}.pdf`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
    setMessage("Diet plan downloaded.");
  } catch (error) {
    setMessage(error.message, true);
  } finally {
    downloadDietPlanButton.disabled = false;
    downloadDietPlanButton.textContent = "Download Diet Plan";
  }
}

async function loadPerformanceMetrics() {
  try {
    const response = await apiFetch("/metrics");
    if (!response.ok) throw new Error("Metrics unavailable");
    const data = await response.json();
    const test = data.test || {};
    const validation = data.validation || {};
    performanceCards.innerHTML = [
      ["Test Accuracy", test.accuracy],
      ["Test F1", test.f1_weighted],
      ["Test ROC-AUC", test.roc_auc],
      ["Validation Accuracy", validation.accuracy],
    ]
      .map(
        ([label, value]) => `
          <article class="performance-card">
            <span>${label}</span>
            <strong>${typeof value === "number" ? formatPercent(value) : "-"}</strong>
          </article>
        `
      )
      .join("");
  } catch (error) {
    performanceCards.innerHTML = `<div class="empty-state compact">${error.message}</div>`;
  }
}

function setConfusionMatrix(split) {
  confusionMatrixImage.src = `${API_BASE_URL}/confusion-matrix/${split}?t=${Date.now()}`;
  matrixTabs.forEach((tab) => tab.classList.toggle("active", tab.dataset.split === split));
}

analyzeButton.addEventListener("click", async (event) => {
  event.preventDefault();

  if (!selectedFile) return;

  analyzeButton.disabled = true;
  analyzeButton.textContent = "Uploading...";
  setMessage("Uploading image...");
  setTimeline(["upload", "preprocess"]);

  try {
    const upload = await uploadImage(selectedFile);
    analyzeButton.textContent = "Analyzing...";
    setMessage("Running AI analysis and heatmap generation...");
    setTimeline(["upload", "preprocess", "predict", "explain"]);
    const result = await analyzeImage(upload.file_id, collectPatientContext());
    renderResults(result);
    setMessage(currentUser ? "Analysis complete and saved to My Cases." : "Analysis complete. Log in to save cases.");
    await loadCases();
  } catch (error) {
    setMessage(error.message, true);
  } finally {
    analyzeButton.disabled = false;
    analyzeButton.textContent = "Analyze X-ray";
  }
});

loginButton.addEventListener("click", async () => {
  try {
    await authRequest("login");
  } catch (error) {
    setAuthMessage(error.message, true);
  }
});

registerButton.addEventListener("click", async () => {
  try {
    await authRequest("register");
  } catch (error) {
    setAuthMessage(error.message, true);
  }
});

logoutButton.addEventListener("click", logout);

clearAllCasesButton.addEventListener("click", async () => {
  try {
    await clearAllCases();
  } catch (error) {
    setMessage(error.message, true);
  }
});

casesList.addEventListener("click", async (event) => {
  const button = event.target.closest(".case-open");
  const deleteButton = event.target.closest(".case-delete");
  if (!button && !deleteButton) return;

  try {
    if (button) {
      await openCase(button.dataset.caseId);
    } else {
      await deleteCase(deleteButton.dataset.caseId);
    }
  } catch (error) {
    setMessage(error.message, true);
  }
});

themeToggle.addEventListener("click", () => {
  applyTheme(document.body.dataset.theme === "dark" ? "light" : "dark");
});

downloadReportButton.addEventListener("click", downloadReport);
downloadDietPlanButton.addEventListener("click", downloadDietPlan);
assistantTopButton.addEventListener("click", openAssistant);
assistantOpenButton.addEventListener("click", openAssistant);
assistantCloseButton.addEventListener("click", closeAssistant);
assistantSendButton.addEventListener("click", askAssistant);
assistantQuestion.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
    event.preventDefault();
    askAssistant();
  }
});
assistantDrawer.addEventListener("click", (event) => {
  if (event.target === assistantDrawer) closeAssistant();
});

matrixTabs.forEach((tab) => {
  tab.addEventListener("click", () => setConfusionMatrix(tab.dataset.split));
});

navLinks.forEach((link) => {
  link.addEventListener("click", (event) => {
    event.preventDefault();
    showPage(link.dataset.pageTarget);
  });
});

window.addEventListener("hashchange", () => showPage(pageFromHash()));

applyTheme(localStorage.getItem("ai-physio-theme") || "dark");
showPage(pageFromHash());
restoreSession();
restoreLatestAnalysis();
checkApiStatus();
setInterval(checkApiStatus, 10000);
loadPerformanceMetrics();
setConfusionMatrix("test");

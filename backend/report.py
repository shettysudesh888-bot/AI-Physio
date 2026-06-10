"""
AI Physio – AI-Assisted Clinical Report Generator (v2.0)
=======================================================
Refactored for clinical trust, investor demos, and research readiness.
All original functionality preserved and extended.
"""
 
from __future__ import annotations
 
import base64
import textwrap
from io import BytesIO
from pathlib import Path
 
import matplotlib
 
matplotlib.use("Agg")
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
 
# ---------------------------------------------------------------------------
# Design Tokens — single source of truth for all colours and sizes
# ---------------------------------------------------------------------------
 
BRAND_TEAL      = "#0f766e"
BRAND_TEAL_DARK = "#0d5c56"
BRAND_TEAL_LIGHT= "#e6f4f3"
BRAND_TEAL_MID  = "#c7dedd"
 
NAVY            = "#172033"
SLATE           = "#607086"
LIGHT_BG        = "#f8fafc"
BORDER          = "#d7e2ea"
 
# Risk palette
RISK_LOW    = {"bg": "#d1fae5", "border": "#10b981", "text": "#065f46", "badge": "#ecfdf5"}
RISK_MOD    = {"bg": "#fef3c7", "border": "#f59e0b", "text": "#92400e", "badge": "#fffbeb"}
RISK_HIGH   = {"bg": "#fee2e2", "border": "#ef4444", "text": "#991b1b", "badge": "#fef2f2"}
 
DISCLAIMER_TEXT = [
    "AI Physio is an educational AI-assisted decision-support system.",
    "Recommendations are generated using AI analysis and user-provided clinical information.",
    "This system is NOT a substitute for professional medical diagnosis, treatment, or clinical judgment.",
    "Always consult a qualified healthcare professional before making any medical decisions.",
]
 
MODEL_INFO = {
    "name":               "DenseNet121",
    "version":            "v2.0",
    "training_samples": "17,221 X-rays",
    "validation_accuracy":"98.78%",
    "sensitivity":        "97.03%",
    "specificity":        "100%",
    "last_updated":       "2026-06-09",
    "framework":          "PyTorch 2.x",
}
 
# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
 
def _tc(value: object) -> str:
    text = str(value or "-").replace("_", " ")
    return " ".join(p.capitalize() for p in text.split())
 
 
def _fmt_pct(value: object) -> str:
    if not isinstance(value, (int, float)):
        return "-"
    return f"{value * 100:.1f}%"
 
 
def _confidence_label(value: object) -> tuple[str, str]:
    """Returns (label, colour) for display confidence."""
    if not isinstance(value, (int, float)):
        return "Unknown", SLATE
    p = value * 100
    if p >= 90:
        return "High", RISK_LOW["text"]
    if p >= 75:
        return "Moderate", RISK_MOD["text"]
    return "Low", RISK_HIGH["text"]
 
 
def _risk_palette(level: str) -> dict:
    lvl = str(level or "low").lower()
    if "high" in lvl:
        return RISK_HIGH
    if "mod" in lvl:
        return RISK_MOD
    return RISK_LOW
 
 
def _wrap(text: object, width: int = 90) -> str:
    clean = " ".join(str(text or "").split())
    return "\n".join(textwrap.wrap(clean, width)) if clean else "-"
 
 
def _add_text(ax, x, y, text, size=9, weight="normal", color=None, alpha=1.0) -> float:
    color = color or NAVY
    wrapped = _wrap(text)
    ax.text(x, y, wrapped, fontsize=size, fontweight=weight, va="top",
            color=color, alpha=alpha)
    return y - (0.036 * max(1, wrapped.count("\n") + 1))
 
 
def _image_from_data_url(data_url: str | None) -> Image.Image | None:
    if not data_url or "," not in data_url:
        return None
    try:
        return Image.open(BytesIO(base64.b64decode(data_url.split(",", 1)[1]))).convert("RGB")
    except Exception:
        return None
 
 
def _image_from_path(path: Path | None) -> Image.Image | None:
    if not path or not path.exists():
        return None
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None
 
 
# ---------------------------------------------------------------------------
# Layout primitives
# ---------------------------------------------------------------------------
 
def _new_page(pdf: PdfPages):
    fig = plt.figure(figsize=(8.27, 11.69))
    ax  = fig.add_axes([0.07, 0.06, 0.86, 0.90])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    return fig, ax
 
 
def _header_band(ax, title: str, subtitle: str = "") -> None:
    """Full-width teal header band at top of page."""
    ax.add_patch(plt.Rectangle((0, 0.955), 1, 0.045, color=BRAND_TEAL, ec="none"))
    ax.text(0.012, 0.978, title, fontsize=15, fontweight="bold",
            va="center", color="white")
    if subtitle:
        ax.text(0.75, 0.978, subtitle, fontsize=8, va="center", color="#b2d8d5",
                ha="left")
 
 
def _footer(ax, page: int, total: int = 7) -> None:
    ax.add_patch(plt.Rectangle((0, -0.028), 1, 0.022, color=LIGHT_BG, ec=BORDER, lw=0.4))
    ax.text(0.012, -0.018, "AI Physio AI-Assisted Analysis Report | For Clinical Review",
            fontsize=7, va="center", color=SLATE)
    ax.text(0.988, -0.018, f"Page {page} of {total}", fontsize=7, va="center",
            color=SLATE, ha="right")
 
 
def _section_label(ax, x, y, text: str, color=None) -> float:
    ax.text(x, y, text, fontsize=11, fontweight="bold", va="top", color=color or NAVY)
    ax.add_patch(plt.Rectangle((x, y - 0.020), 0.20, 0.0015, color=color or BRAND_TEAL, ec="none"))
    return y - 0.030
 
 
def _card(ax, x, y, w, h, bg=None, border=None, lw=0.8, radius=0.015):
    ax.add_patch(FancyBboxPatch(
        (x, y - h), w, h,
        boxstyle=f"round,pad=0",
        linewidth=lw,
        edgecolor=border or BORDER,
        facecolor=bg or LIGHT_BG,
    ))
 
 
def _progress_bar(ax, x, y, w, value: float, label: str, color=None, bg="#e2e8f0") -> float:
    """Render a labelled horizontal progress bar. value 0–1."""
    bar_h = 0.012
    ax.text(x, y, label, fontsize=8, va="top", color=NAVY)
    ax.text(x + w + 0.01, y, f"{int(value * 100)}%", fontsize=8, va="top",
            color=color or BRAND_TEAL, fontweight="bold")
    by = y - 0.022
    ax.add_patch(plt.Rectangle((x, by), w, bar_h, color=bg, ec="none"))
    ax.add_patch(plt.Rectangle((x, by), w * max(0, min(1, value)), bar_h,
                                color=color or BRAND_TEAL, ec="none"))
    return by - 0.022
 
 
# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------
 
def _render_disclaimer_banner(ax, y: float) -> float:
    """Amber disclaimer card."""
    h = 0.085
    ax.add_patch(plt.Rectangle((0, y - h), 1, h, color="#fffbeb", ec="#f59e0b", lw=1.0))
    ax.text(0.012, y - 0.013, "⚠  Educational Disclaimer", fontsize=9,
            fontweight="bold", va="top", color="#92400e")
    ty = y - 0.030
    for line in DISCLAIMER_TEXT:
        ty = _add_text(ax, 0.012, ty, line, size=7.5, color="#92400e")
    return y - h - 0.015
 
 
def _render_model_info_card(ax, y: float) -> float:
    """AI Model transparency card."""
    h = 0.115
    _card(ax, 0, y, 1, h, bg="#f0f7ff", border="#bcd4f0")
    ax.text(0.012, y - 0.014, "AI Model Information", fontsize=10, fontweight="bold",
            va="top", color="#1e40af")
    # Two-column grid
    fields = [
        ("Model",               MODEL_INFO["name"]),
        ("Version",             MODEL_INFO["version"]),
        ("Training Samples",    MODEL_INFO["training_samples"]),
        ("Last Updated",        MODEL_INFO["last_updated"]),
        ("Validation Accuracy", MODEL_INFO["validation_accuracy"]),
        ("Sensitivity",         MODEL_INFO["sensitivity"]),
        ("Specificity",         MODEL_INFO["specificity"]),
        ("Framework",           MODEL_INFO["framework"]),
    ]
    row_y = y - 0.038
    for i, (lbl, val) in enumerate(fields):
        col = i % 2
        cx = 0.012 + col * 0.50
        if i > 0 and col == 0:
            row_y -= 0.022
        ax.text(cx, row_y, f"{lbl}:", fontsize=7.5, fontweight="bold",
                va="top", color=SLATE)
        ax.text(cx + 0.18, row_y, val, fontsize=7.5, va="top", color=NAVY)
    return y - h - 0.015
 
 
# ---------------------------------------------------------------------------
# Page 1 – Executive Summary
# ---------------------------------------------------------------------------
 
def _render_executive_summary(pdf: PdfPages, analysis: dict) -> None:
    prediction     = analysis.get("prediction") or {}
    patient_context= analysis.get("patient_context") or {}
    recommendations= analysis.get("recommendations") or {}
    recovery       = analysis.get("recovery_guidance") or {}
    exercise_plan  = analysis.get("exercise_plan") or {}
    eligibility    = analysis.get("exercise_eligibility") or recovery.get("exercise_eligibility") or {}
    risk_block     = recovery.get("recovery_risk_level") or {}
    nutrition      = recovery.get("nutrition_plan") or {}
 
    conf_val   = prediction.get("confidence")
    conf_label, conf_color = _confidence_label(conf_val)
    risk_level = (risk_block.get("level") or "low").lower()
    pal        = _risk_palette(risk_level)
 
    fig, ax = _new_page(pdf)
    _header_band(ax, "AI Physio  —  AI-Assisted Analysis Report",
                 "Executive Summary  |  Page 1 of 7")
    _footer(ax, 1)
 
    y = 0.945
 
    # ── Brand intro ─────────────────────────────────────────────────────────
    ax.text(0, y, "Patient Summary", fontsize=18, fontweight="bold",
            va="top", color=BRAND_TEAL)
    y -= 0.032
    y = _add_text(ax, 0, y,
        "AI-assisted X-ray screening result. Clinical review is recommended before making treatment decisions.",
        size=9, color=SLATE)
    y -= 0.018
 
    # ── Summary metric cards (2 × 4 grid) ───────────────────────────────────
    cards = [
        ("AI Screening Result",
         _tc(prediction.get("label") or prediction.get("condition") or "Pending"),
         BRAND_TEAL),
        ("Model Confidence",
         f"{conf_label}  ({_fmt_pct(conf_val)})",
         conf_color),
        ("Recovery Stage",
         _tc(patient_context.get("recovery_stage") or "-"),
         BRAND_TEAL),
        ("Risk Level",
         _tc(risk_block.get("label") or risk_level or "Low"),
         pal["text"]),
        ("Body Part",
         _tc(patient_context.get("body_part") or "-"),
         NAVY),
        ("Treatment Status",
         _tc(patient_context.get("treatment_status") or "-"),
         NAVY),
        ("Exercise Plan",
         "Generated ✓" if (eligibility.get("eligible") and exercise_plan.get("exercises")) else "Not Generated",
         RISK_LOW["text"] if eligibility.get("eligible") else RISK_HIGH["text"]),
        ("Nutrition Plan",
         "Generated ✓" if nutrition.get("meals") else "Not Generated",
         RISK_LOW["text"] if nutrition.get("meals") else RISK_HIGH["text"]),
    ]
 
    card_w, card_h, gap = 0.235, 0.080, 0.015
    for i, (label, value, color) in enumerate(cards):
        col = i % 4
        row = i // 4
        cx = col * (card_w + gap)
        cy = y - row * (card_h + gap)
        bg = BRAND_TEAL_LIGHT if color == BRAND_TEAL else LIGHT_BG
        _card(ax, cx, cy, card_w, card_h, bg=bg, border=BORDER)
        ax.text(cx + 0.012, cy - 0.014, label, fontsize=7.5, fontweight="bold",
                va="top", color=SLATE)
        # Wrap long value text
        val_text = _wrap(value, 22)
        ax.text(cx + 0.012, cy - 0.036, val_text, fontsize=9, fontweight="bold",
                va="top", color=color)
    y -= 2 * (card_h + gap) + 0.025
 
    # ── Risk Level Badge ─────────────────────────────────────────────────────
    risk_label = _tc(risk_block.get("label") or risk_level)
    indicator  = {"high": "[HIGH]", "moderate": "[MOD]", "low": "[LOW]"}.get(risk_level, "[LOW]")
    h_badge = 0.055
    ax.add_patch(plt.Rectangle((0, y - h_badge), 1, h_badge,
                                color=pal["bg"], ec=pal["border"], lw=1.2))
    ax.text(0.012, y - 0.013, f"{indicator}  Recovery Risk Level:", fontsize=10,
            fontweight="bold", va="top", color=pal["text"])
    ax.text(0.36,  y - 0.013, risk_label, fontsize=10, fontweight="bold",
            va="top", color=pal["text"])
    reasons = risk_block.get("reasons") or ["No elevated risk factors reported."]
    ax.text(0.012, y - 0.036, "Factors:  " + "  ·  ".join(reasons[:3]),
            fontsize=7.5, va="top", color=pal["text"])
    y -= h_badge + 0.022
 
    # ── Confidence explainer ─────────────────────────────────────────────────
    y = _section_label(ax, 0, y, "Confidence Interpretation")
    _card(ax, 0, y, 1, 0.058, bg="#f0fdf4", border="#86efac")
    ax.text(0.012, y - 0.012, f"Prediction Confidence:  {conf_label}   ({_fmt_pct(conf_val)})",
            fontsize=10, fontweight="bold", va="top", color=RISK_LOW["text"])
    _add_text(ax, 0.012, y - 0.032,
        "Confidence reflects model certainty based on validation data patterns. "
        "It does not represent diagnostic certainty and should be interpreted alongside clinical assessment.",
        size=7.5, color="#166534")
    y -= 0.075
 
    # ── Clinical context quick view ──────────────────────────────────────────
    y = _section_label(ax, 0, y, "Clinical Context")
    fields = [
        ("Age",           patient_context.get("age")),
        ("Sex",           _tc(patient_context.get("sex"))),
        ("Pain Level",    f"{patient_context.get('pain_level')}/10" if patient_context.get("pain_level") is not None else None),
        ("Swelling",      _tc(patient_context.get("swelling_level"))),
        ("Mobility",      _tc(patient_context.get("mobility_status"))),
        ("Restrictions",  _tc(patient_context.get("doctor_restrictions"))),
    ]
    visible = [(l, v) for l, v in fields if v and str(v).strip() not in ("-", "None")]
    _card(ax, 0, y, 1, 0.065, bg=LIGHT_BG, border=BORDER)
    for i, (lbl, val) in enumerate(visible[:6]):
        col = i % 3
        if i > 0 and col == 0:
            y -= 0.024
        cx = 0.012 + col * 0.33
        ax.text(cx, y - 0.018, f"{lbl}:", fontsize=7.5, fontweight="bold",
                va="top", color=SLATE)
        ax.text(cx + 0.10, y - 0.018, str(val), fontsize=7.5, va="top", color=NAVY)
    y -= 0.082
 
    # ── Disclaimer ───────────────────────────────────────────────────────────
    if y > 0.15:
        _render_disclaimer_banner(ax, y if y > 0.15 else 0.15)
 
    pdf.savefig(fig)
    plt.close(fig)
 
 
# ---------------------------------------------------------------------------
# Page 2 – AI Findings + Confidence + Model Info
# ---------------------------------------------------------------------------
 
def _render_ai_findings_page(pdf: PdfPages, analysis: dict) -> None:
    prediction  = analysis.get("prediction") or {}
    llm         = analysis.get("llm_explanation") or {}
    class_scores= prediction.get("class_scores") or {}
    recommendations = analysis.get("recommendations") or {}
 
    conf_val            = prediction.get("confidence")
    conf_label, conf_c  = _confidence_label(conf_val)
 
    fig, ax = _new_page(pdf)
    _header_band(ax, "AI Findings & Model Transparency", "Page 2 of 7")
    _footer(ax, 2)
    y = 0.945
 
    # ── Detection result ─────────────────────────────────────────────────────
    y = _section_label(ax, 0, y, "Detection Result")
    _raw_condition = _tc(prediction.get("condition") or "-")
    _is_no_fracture = "normal" in (prediction.get("condition") or "").lower() or \
                      "no fracture" in (prediction.get("label") or "").lower() or \
                      "normal" in (prediction.get("label") or "").lower()
    _screening_outcome_label = "Screening Outcome" if _is_no_fracture else "Condition"
    _screening_outcome_value = "No Fracture Pattern Detected" if _is_no_fracture else _raw_condition
    det_cards = [
        ("Screening Pattern",     _tc(prediction.get("label") or "-")),
        (_screening_outcome_label, _screening_outcome_value),
        ("Model Confidence",      f"{conf_label}  ({_fmt_pct(conf_val)})"),
        ("Severity",              _tc((recommendations.get("severity") or "Review").replace("_"," "))),
    ]
    cw, ch = 0.235, 0.072
    for i, (lbl, val) in enumerate(det_cards):
        cx = i * (cw + 0.015)
        _card(ax, cx, y, cw, ch, bg=BRAND_TEAL_LIGHT, border=BRAND_TEAL_MID)
        ax.text(cx + 0.01, y - 0.013, lbl, fontsize=7.5, fontweight="bold",
                va="top", color=SLATE)
        ax.text(cx + 0.01, y - 0.036, _wrap(val, 22), fontsize=9, fontweight="bold",
                va="top", color=BRAND_TEAL_DARK)
    y -= ch + 0.022
 
    # ── Confidence explainer card ────────────────────────────────────────────
    y = _section_label(ax, 0, y, "Confidence Interpretation")
    _card(ax, 0, y, 1, 0.065, bg="#f0fdf4", border="#86efac")
    ax.text(0.012, y - 0.013,
            f"Prediction Confidence:  {conf_label}",
            fontsize=11, fontweight="bold", va="top", color=conf_c)
    ax.text(0.50,  y - 0.013,
            f"Model Score:  {_fmt_pct(conf_val)}",
            fontsize=11, fontweight="bold", va="top", color=NAVY)
    _add_text(ax, 0.012, y - 0.037,
        "Confidence reflects the model's certainty based on patterns learned from validation data "
        "and should NOT be interpreted as diagnostic certainty.",
        size=7.5, color="#166534")
    y -= 0.082
 
    # ── AI Explanation ───────────────────────────────────────────────────────
    y = _section_label(ax, 0, y, "AI-Generated Explanation")
    for lbl, key in [("Summary",    "summary"),
                     ("Confidence", "confidence_note"),
                     ("Next Steps", "next_steps"),
                     ("Safety",     "safety_note")]:
        val = llm.get(key)
        if not val:
            continue
        ax.text(0, y, f"{lbl}:", fontsize=8.5, fontweight="bold", va="top", color=NAVY)
        y = _add_text(ax, 0.20, y, val, size=8.5, color=NAVY)
        y -= 0.008
 
    # ── Class probabilities visual bars ──────────────────────────────────────
    if class_scores:
        y -= 0.005
        y = _section_label(ax, 0, y, "Class Probability Distribution")
        max_score = max(class_scores.values()) if class_scores else 1
        for lbl, score in class_scores.items():
            if not isinstance(score, (int, float)):
                continue
            bar_color = BRAND_TEAL if score == max_score else BORDER
            y = _progress_bar(ax, 0, y, 0.65, score, _tc(lbl),
                              color=bar_color)
        y -= 0.008
 
    # ── Model info card ───────────────────────────────────────────────────────
    if y > 0.18:
        y = _section_label(ax, 0, y, "AI Model Information")
        y = _render_model_info_card(ax, y)
 
    if y > 0.14:
        _render_disclaimer_banner(ax, y if y > 0.14 else 0.14)
 
    pdf.savefig(fig)
    plt.close(fig)
 
 
# ---------------------------------------------------------------------------
# Page 3 – Image Review & Heatmap
# ---------------------------------------------------------------------------
 
def _render_image_review_page(pdf: PdfPages, analysis: dict,
                               uploaded_image_path: Path | None) -> None:
    prediction   = analysis.get("prediction") or {}
    explainability = prediction.get("explainability") or {}
    original = _image_from_path(uploaded_image_path)
    heatmap  = _image_from_data_url(explainability.get("overlay_image"))
 
    if not original and not heatmap:
        return
 
    fig, ax = _new_page(pdf)
    _header_band(ax, "Image Review & Heatmap Analysis", "Page 3 of 7")
    _footer(ax, 3)
    y = 0.945
 
    y = _section_label(ax, 0, y, "Radiological Image Review")
    y = _add_text(ax, 0, y,
        "Side-by-side view of the uploaded X-ray and the AI-generated activation heatmap.",
        size=8.5, color=SLATE)
    y -= 0.012
 
    images = [(t, img) for t, img in [("Uploaded X-Ray", original),
                                        ("AI Heatmap Overlay", heatmap)]
              if img is not None]
 
    for idx, (title, image) in enumerate(images):
        x_off = 0.0 if idx == 0 else 0.52
        ax.text(x_off, y, title, fontsize=11, fontweight="bold",
                va="top", color=NAVY)
        # Border frame for image
        ax.add_patch(plt.Rectangle((x_off, y - 0.37), 0.44, 0.345,
                                    color="#f8fafc", ec=BORDER, lw=1.0))
        img_ax = fig.add_axes([0.07 + x_off * 0.86, 0.565, 0.38, 0.32])
        img_ax.imshow(image)
        img_ax.axis("off")
 
    y -= 0.395
 
    # ── Heatmap interpretation card ──────────────────────────────────────────
    y = _section_label(ax, 0, y, "Heatmap Interpretation")
    _card(ax, 0, y, 1, 0.065, bg="#fffbeb", border="#f59e0b")
    ax.text(0.012, y - 0.013,
            "⚠  Important — Heatmap Regions Are Not Confirmed Injury Locations",
            fontsize=9, fontweight="bold", va="top", color="#92400e")
    note = (explainability.get("note") or
            "The highlighted regions indicate areas that most influenced the AI prediction. "
            "These regions are provided only for model interpretability and must be reviewed by a clinician.")
    _add_text(ax, 0.012, y - 0.035, note, size=7.5, color="#92400e")
    y -= 0.082
 
    # ── Prediction disclaimer ─────────────────────────────────────────────────
    disc = prediction.get("disclaimer") or "Educational use only. Not a diagnostic tool."
    _add_text(ax, 0, y, disc, size=8.5, color=SLATE)
    y -= 0.025
 
    if y > 0.12:
        _render_disclaimer_banner(ax, y if y > 0.12 else 0.12)
 
    pdf.savefig(fig)
    plt.close(fig)
 
 
# ---------------------------------------------------------------------------
# Page 4 –  Assessment + Recovery TimeClinical Riskline
# ---------------------------------------------------------------------------
 
def _render_risk_timeline_page(pdf: PdfPages, analysis: dict) -> None:
    recovery       = analysis.get("recovery_guidance") or {}
    physio         = recovery.get("physio_support") or {}
    patient_context= analysis.get("patient_context") or {}
    risk_block     = recovery.get("recovery_risk_level") or {}
 
    risk_level  = (risk_block.get("level") or "low").lower()
    pal         = _risk_palette(risk_level)
    reasons     = risk_block.get("reasons") or []
 
    # Derive a numeric risk score 0–100
    score_map = {"low": 28, "moderate": 58, "high": 82}
    risk_score = score_map.get(risk_level, 30)
 
    fig, ax = _new_page(pdf)
    _header_band(ax, "AI-Based Recovery Risk Assessment", "Page 4 of 7")
    _footer(ax, 4)
    y = 0.945
 
    # ── Risk score card ───────────────────────────────────────────────────────
    y = _section_label(ax, 0, y, "Recovery Risk Score")
    _card(ax, 0, y, 1, 0.110, bg=pal["bg"], border=pal["border"])
 
    indicator = {"high": "[HIGH]", "moderate": "[MOD]", "low": "[LOW]"}.get(risk_level, "[LOW]")
    ax.text(0.012, y - 0.015,
            f"{indicator}  Risk Score:", fontsize=11, fontweight="bold",
            va="top", color=pal["text"])
    ax.text(0.25,  y - 0.015,
            f"{risk_score} / 100", fontsize=18, fontweight="bold",
            va="top", color=pal["text"])
    ax.text(0.50,  y - 0.015,
            f"Category: {_tc(risk_block.get('label') or risk_level)}",
            fontsize=11, fontweight="bold", va="top", color=pal["text"])
 
    # Progress bar for score
    bar_y = y - 0.048
    ax.add_patch(plt.Rectangle((0.012, bar_y), 0.60, 0.012,
                                color="#e2e8f0", ec="none"))
    ax.add_patch(plt.Rectangle((0.012, bar_y),
                                0.60 * (risk_score / 100), 0.012,
                                color=pal["border"], ec="none"))
 
    # Contributing factors
    ax.text(0.012, y - 0.072, "Contributing Factors:", fontsize=8,
            fontweight="bold", va="top", color=pal["text"])
    if reasons:
        ax.text(0.25, y - 0.072,
                "  ·  ".join(reasons[:4]),
                fontsize=7.5, va="top", color=pal["text"])
    else:
        ax.text(0.25, y - 0.072, "No elevated risk factors reported.",
                fontsize=7.5, va="top", color=pal["text"])
    y -= 0.128
 
    # ── Progress tracking bars ────────────────────────────────────────────────
    y = _section_label(ax, 0, y, "Recovery Progress Components")
    pc = analysis.get("patient_context") or {}
    pain_level   = pc.get("pain_level")
    pain_score   = (10 - float(pain_level)) / 10 if pain_level is not None else 0.5
    mob_map = {"normal": 0.90, "mild_limitation": 0.65, "moderate_limitation": 0.45,
               "severe_limitation": 0.20, "non_weight_bearing": 0.10}
    mob_score = mob_map.get(str(pc.get("mobility_status") or ""), 0.55)
    swelling_map = {"none": 0.95, "mild": 0.70, "moderate": 0.45, "severe": 0.15}
    swell_score = swelling_map.get(str(pc.get("swelling_level") or ""), 0.60)
 
    components = [
        ("Pain Management",     pain_score,  "#10b981"),
        ("Mobility Function",   mob_score,   BRAND_TEAL),
        ("Swelling Control",    swell_score, "#3b82f6"),
        ("Exercise Compliance", 0.50,        "#f59e0b"),  # Default – to be updated
        ("Overall Function",    (pain_score + mob_score + swell_score) / 3, "#8b5cf6"),
    ]
    for label, score, color in components:
        y = _progress_bar(ax, 0, y, 0.65, score, label, color=color)
    y -= 0.015
 
    # ── Recovery timeline ─────────────────────────────────────────────────────
    y = _section_label(ax, 0, y, "Recovery Timeline")
 
    timeline_entries = [
        ("Day 1–3",  "Pain reassessment. Ice/elevation. Monitor neurovascular status."),
        ("Week 1",   "Mobility reassessment. Begin gentle range-of-motion exercises."),
        ("Week 2",   "Functional review. Progress loading if pain permits."),
        ("Week 4",   "Return-to-activity assessment. Clinician sign-off required."),
        ("Week 6–8", "Full functional review. Consider imaging if symptoms persist."),
    ]
 
    stages_map = {
        "acute_phase":    0,
        "early_recovery": 1,
        "late_recovery":  2,
    }
    current_key   = str(patient_context.get("recovery_stage") or "acute_phase")
    current_idx   = stages_map.get(current_key, 0)
    stage_names   = ["Acute Phase", "Early Recovery", "Late Recovery"]
 
    # Stage strip
    sw, sh = 0.31, 0.055
    for i, name in enumerate(stage_names):
        is_current  = i == current_idx
        is_complete = i < current_idx
        face  = BRAND_TEAL       if is_current  else (BRAND_TEAL_LIGHT if is_complete else LIGHT_BG)
        tc_   = "white"          if is_current  else (BRAND_TEAL        if is_complete else SLATE)
        edge  = BRAND_TEAL       if is_current  else BORDER
        ax.add_patch(plt.Rectangle((i * (sw + 0.01), y - sh), sw, sh,
                                    color=face, ec=edge, lw=1.0))
        marker = "✓ " if i <= current_idx else ""
        ax.text(0.012 + i * (sw + 0.01), y - 0.022, f"{marker}{name}",
                fontsize=8.5, fontweight="bold" if is_current else "normal",
                va="top", color=tc_)
    y -= sh + 0.018
 
    # Timeline entries
    physio_timeline = physio.get("timeline") or []
    entries = physio_timeline[:5] if physio_timeline else [e for _, e in timeline_entries]
    display = list(zip([t for t, _ in timeline_entries], entries))
    for period, desc in display[:5]:
        if y < 0.15:
            break
        ax.add_patch(plt.Rectangle((0, y - 0.028), 0.10, 0.024,
                                    color=BRAND_TEAL_LIGHT, ec=BORDER, lw=0.5))
        ax.text(0.012, y - 0.008, period, fontsize=7.5, fontweight="bold",
                va="top", color=BRAND_TEAL_DARK)
        y = _add_text(ax, 0.12, y, desc, size=8)
        y -= 0.004
 
    if y > 0.12:
        _render_disclaimer_banner(ax, y if y > 0.12 else 0.12)
 
    pdf.savefig(fig)
    plt.close(fig)
 
 
# ---------------------------------------------------------------------------
# Page 5 – Exercise Plan
# ---------------------------------------------------------------------------
 
def _render_exercise_page_v2(pdf: PdfPages, analysis: dict) -> None:
    recovery       = analysis.get("recovery_guidance") or {}
    physio         = recovery.get("physio_support") or {}
    exercise_plan  = analysis.get("exercise_plan") or {}
    eligibility    = analysis.get("exercise_eligibility") or recovery.get("exercise_eligibility") or {}
    patient_context= analysis.get("patient_context") or {}
 
    fig, ax = _new_page(pdf)
    _header_band(ax, "Exercise & Physiotherapy Plan", "Page 5 of 7")
    _footer(ax, 5)
    y = 0.945
 
    # ── Eligibility banner ────────────────────────────────────────────────────
    eligible    = eligibility.get("eligible", False)
    elig_pal    = RISK_LOW if eligible else RISK_HIGH
    icon        = "✓" if eligible else "✗"
    elig_label  = "Exercise Plan Generated" if eligible else "Exercise Plan Not Available"
    ax.add_patch(plt.Rectangle((0, y - 0.045), 1, 0.042,
                                color=elig_pal["bg"], ec=elig_pal["border"], lw=1.2))
    ax.text(0.015, y - 0.010, f"{icon}  {elig_label}", fontsize=10,
            fontweight="bold", va="top", color=elig_pal["text"])
    if not eligible and eligibility.get("reason"):
        ax.text(0.50, y - 0.010, f"Reason: {eligibility['reason']}",
                fontsize=8, va="top", color=elig_pal["text"])
    y -= 0.060
 
    if not eligible:
        y = _section_label(ax, 0, y, "Recovery Guidance")
        y = _add_text(ax, 0, y, physio.get("personalized_intro") or
            "Exercise has not been prescribed at this stage. Follow clinician advice and rest.", size=9)
        if y > 0.14:
            _render_disclaimer_banner(ax, y if y > 0.14 else 0.14)
        pdf.savefig(fig)
        plt.close(fig)
        return
 
    # ── Phase header ──────────────────────────────────────────────────────────
    phase = physio.get("phase") or physio.get("summary") or "Recovery Phase"
    ax.text(0, y, phase, fontsize=11, fontweight="bold", va="top", color=BRAND_TEAL)
    y -= 0.028
    if physio.get("personalized_intro"):
        y = _add_text(ax, 0, y, physio["personalized_intro"], size=8.5, color=SLATE)
        y -= 0.010
 
    # ── Red flags ────────────────────────────────────────────────────────────
    red_flags = recovery.get("red_flags") or []
    if red_flags and y > 0.50:
        y = _section_label(ax, 0, y, "⚠  Red Flag Alerts", color="#b91c1c")
        _card(ax, 0, y, 1, min(len(red_flags[:3]) * 0.026 + 0.020, 0.095),
              bg="#fff1f2", border="#fca5a5")
        for flag in red_flags[:3]:
            y = _add_text(ax, 0.015, y, f"• {flag}", size=8, color="#991b1b")
        y -= 0.010
 
    # ── Daily focus ───────────────────────────────────────────────────────────
    daily = physio.get("daily_focus") or []
    if daily and y > 0.35:
        y = _section_label(ax, 0, y, "Daily Focus")
        for item in daily[:5]:
            ax.text(0.012, y, "•", fontsize=9, va="top", color=BRAND_TEAL)
            y = _add_text(ax, 0.030, y, item, size=8.5)
        y -= 0.008
 
    # ── Exercise cards ────────────────────────────────────────────────────────
    exercises = exercise_plan.get("exercises") or []
    if exercises:
        y = _section_label(ax, 0, y, "Exercise Protocol")
        for ex in exercises[:5]:
            if y < 0.12:
                break
            name  = _tc(ex.get("name") or "Exercise")
            desc  = ex.get("description") or ""
            freq  = ex.get("frequency") or "-"
            sets_ = ex.get("sets") or "-"
            reps  = ex.get("reps") or "-"
            safety= ex.get("safety_note") or ex.get("notes") or ""
            media = ex.get("media") or {}
 
            card_h = 0.085 + (0.020 if safety else 0) + (0.014 if media.get("youtube_url") else 0)
            _card(ax, 0, y, 1, card_h, bg=LIGHT_BG, border=BRAND_TEAL_MID)
            ax.add_patch(plt.Rectangle((0, y - 0.028), 0.005, 0.025,
                                        color=BRAND_TEAL, ec="none"))
            ax.text(0.015, y - 0.013, name, fontsize=10, fontweight="bold",
                    va="top", color=BRAND_TEAL_DARK)
            # Sets / Reps / Frequency pill row
            pills = [("Sets", sets_), ("Reps", reps), ("Frequency", freq)]
            px = 0.015
            for plbl, pval in pills:
                pw = 0.12
                ax.add_patch(plt.Rectangle((px, y - 0.042), pw, 0.016,
                                            color=BRAND_TEAL_LIGHT, ec=BRAND_TEAL_MID, lw=0.5))
                ax.text(px + 0.005, y - 0.034, f"{plbl}: {pval}",
                        fontsize=7, va="top", color=BRAND_TEAL_DARK)
                px += pw + 0.008
 
            ty = y - 0.060
            if desc:
                ty = _add_text(ax, 0.015, ty, desc, size=7.5, color=SLATE)
            if safety:
                ty = _add_text(ax, 0.015, ty, f"Safety: {safety}", size=7, color="#b45309")
            if media.get("youtube_url"):
                _add_text(ax, 0.015, ty, f"▶  {media['youtube_url']}", size=7, color="#2563eb")
 
            y -= card_h + 0.012
 
    # ── Avoids ────────────────────────────────────────────────────────────────
    avoids = physio.get("avoids") or []
    if avoids and y > 0.18:
        y = _section_label(ax, 0, y, "Avoid", color="#b91c1c")
        for item in avoids[:4]:
            y = _add_text(ax, 0.015, y, f"✗  {item}", size=8, color="#991b1b")
        y -= 0.005
 
    if y > 0.12:
        _render_disclaimer_banner(ax, y if y > 0.12 else 0.12)
 
    pdf.savefig(fig)
    plt.close(fig)
 
 
# ---------------------------------------------------------------------------
# Page 6 – Nutrition Plan
# ---------------------------------------------------------------------------
 
def _render_nutrition_page_v2(pdf: PdfPages, analysis: dict,
                               page_num: int = 6,
                               title: str = "Recovery Nutrition Plan") -> None:
    recovery       = analysis.get("recovery_guidance") or {}
    nutrition      = recovery.get("nutrition_plan") or {}
    patient_context= analysis.get("patient_context") or {}
 
    fig, ax = _new_page(pdf)
    _header_band(ax, title, f"Page {page_num} of 7")
    _footer(ax, page_num)
    y = 0.945
 
    stage = (patient_context.get("recovery_stage") or "").replace("_", " ").title()
    weight = patient_context.get("weight_kg") or patient_context.get("weight")
 
    # ── Personalised protein target ───────────────────────────────────────────
    if weight:
        try:
            w = float(weight)
            protein_g = int(w * 1.5)
            y = _section_label(ax, 0, y, "Personalised Targets")
            _card(ax, 0, y, 1, 0.058, bg=BRAND_TEAL_LIGHT, border=BRAND_TEAL_MID)
            ax.text(0.012, y - 0.014, f"Weight: {w:.0f} kg",
                    fontsize=10, fontweight="bold", va="top", color=BRAND_TEAL_DARK)
            ax.text(0.22,  y - 0.014, f"Protein Target: {protein_g} g/day",
                    fontsize=10, fontweight="bold", va="top", color=BRAND_TEAL_DARK)
            ax.text(0.60,  y - 0.014, f"Stage: {stage or '-'}",
                    fontsize=10, fontweight="bold", va="top", color=BRAND_TEAL_DARK)
            y -= 0.075
        except (ValueError, TypeError):
            pass
 
    # ── Summary ───────────────────────────────────────────────────────────────
    y = _add_text(ax, 0, y,
        nutrition.get("summary") or
        "Balanced recovery meals with adequate protein, calcium, vitamin D, and hydration.",
        size=9, color=SLATE)
    if nutrition.get("llm_note"):
        y = _add_text(ax, 0, y, nutrition["llm_note"], size=8.5, color=SLATE)
    y -= 0.015
 
    # ── Daily targets ─────────────────────────────────────────────────────────
    targets = nutrition.get("daily_targets") or []
    if targets:
        y = _section_label(ax, 0, y, "Daily Nutritional Targets")
        # Show as coloured pill grid
        pills_per_row = 3
        pill_w, pill_h = 0.30, 0.030
        for i, target in enumerate(targets[:9]):
            col = i % pills_per_row
            row = i // pills_per_row
            if col == 0 and i > 0:
                y -= pill_h + 0.006
            cx = col * (pill_w + 0.02)
            ax.add_patch(plt.Rectangle((cx, y - pill_h), pill_w, pill_h,
                                        color=BRAND_TEAL_LIGHT, ec=BRAND_TEAL_MID, lw=0.5))
            ax.text(cx + 0.010, y - 0.008, _wrap(target, 30), fontsize=7.5,
                    va="top", color=BRAND_TEAL_DARK)
        y -= pill_h + 0.025
 
    # ── Meal table ────────────────────────────────────────────────────────────
    meals = nutrition.get("meals") or []
    if meals:
        y = _section_label(ax, 0, y, "Full-Day Meal Plan")
        # Header row
        _card(ax, 0, y, 1, 0.025, bg=BRAND_TEAL, border=BRAND_TEAL)
        for lbl, cx_ in [("Time", 0.012), ("Meal", 0.15), ("Foods", 0.58)]:
            ax.text(cx_, y - 0.008, lbl, fontsize=8, fontweight="bold",
                    va="top", color="white")
        y -= 0.028
 
        for i, meal in enumerate(meals):
            if y < 0.18:
                break
            row_h = 0.062
            bg_ = LIGHT_BG if i % 2 == 0 else "white"
            _card(ax, 0, y, 1, row_h, bg=bg_, border=BORDER)
            ax.text(0.012, y - 0.012, meal.get("time") or "Meal",
                    fontsize=8, fontweight="bold", va="top", color=BRAND_TEAL)
            ax.text(0.15,  y - 0.012, _wrap(meal.get("name") or "", 20),
                    fontsize=8.5, fontweight="bold", va="top", color=NAVY)
            items_str = ", ".join(meal.get("items") or [])
            _add_text(ax, 0.58, y - 0.010, items_str, size=7.5, color=SLATE)
            why = meal.get("why") or ""
            if why:
                _add_text(ax, 0.15, y - 0.035, why, size=7, color=SLATE)
            y -= row_h + 0.005
 
    # ── Avoid section ─────────────────────────────────────────────────────────
    avoid_list = nutrition.get("avoid") or []
    if avoid_list and y > 0.22:
        y = _section_label(ax, 0, y, "Avoid or Limit")
        for item in avoid_list[:4]:
            y = _add_text(ax, 0.015, y, f"✗  {item}", size=8, color="#991b1b")
        y -= 0.005
 
    # ── Nutrition disclaimer ──────────────────────────────────────────────────
    if y > 0.18:
        _add_text(ax, 0, y,
            nutrition.get("disclaimer") or
            "This diet plan is educational and should be adjusted by a registered dietitian.",
            size=8, color=SLATE)
        y -= 0.030
 
    if y > 0.12:
        _render_disclaimer_banner(ax, y if y > 0.12 else 0.12)
 
    pdf.savefig(fig)
    plt.close(fig)
 
 
# ---------------------------------------------------------------------------
# Page 7 – Progress Tracking + Clinical Safety
# ---------------------------------------------------------------------------
 
def _render_progress_safety_page(pdf: PdfPages, analysis: dict) -> None:
    recovery       = analysis.get("recovery_guidance") or {}
    red_flags      = recovery.get("red_flags") or []
    when_help      = recovery.get("when_to_seek_help") or []
    patient_context= analysis.get("patient_context") or {}
 
    fig, ax = _new_page(pdf)
    _header_band(ax, "Progress Tracking & Safety Guidance", "Page 7 of 7")
    _footer(ax, 7)
    y = 0.945
 
    # ── Recovery score card ───────────────────────────────────────────────────
    pain_level   = patient_context.get("pain_level")
    pain_score   = (10 - float(pain_level)) / 10 if pain_level is not None else 0.5
    mob_map = {"normal": 0.90, "mild_limitation": 0.65, "moderate_limitation": 0.45,
               "severe_limitation": 0.20, "non_weight_bearing": 0.10}
    mob_score    = mob_map.get(str(patient_context.get("mobility_status") or ""), 0.55)
    swelling_map = {"none": 0.95, "mild": 0.70, "moderate": 0.45, "severe": 0.15}
    swell_score  = swelling_map.get(str(patient_context.get("swelling_level") or ""), 0.60)
    overall      = (pain_score + mob_score + swell_score) / 3
    overall_100  = int(overall * 100)
 
    y = _section_label(ax, 0, y, "Recovery Score")
    _card(ax, 0, y, 1, 0.080, bg=BRAND_TEAL_LIGHT, border=BRAND_TEAL_MID)
    ax.text(0.012, y - 0.015, "Overall Recovery Score", fontsize=10,
            fontweight="bold", va="top", color=BRAND_TEAL_DARK)
    ax.text(0.55,  y - 0.015, f"{overall_100} / 100", fontsize=22,
            fontweight="bold", va="top", color=BRAND_TEAL)
    bar_y = y - 0.052
    ax.add_patch(plt.Rectangle((0.012, bar_y), 0.65, 0.014,
                                color="#e2e8f0", ec="none"))
    ax.add_patch(plt.Rectangle((0.012, bar_y), 0.65 * overall,
                                0.014, color=BRAND_TEAL, ec="none"))
    ax.text(0.68, bar_y + 0.002, "Based on reported symptoms",
            fontsize=7, va="bottom", color=SLATE)
    y -= 0.098
 
    # ── Progress bars ─────────────────────────────────────────────────────────
    y = _section_label(ax, 0, y, "Recovery Component Tracking")
    components = [
        ("Pain Management",     pain_score,  "#10b981", "Higher = less pain"),
        ("Mobility Function",   mob_score,   BRAND_TEAL, "Range of motion status"),
        ("Swelling Control",    swell_score, "#3b82f6", "Oedema management"),
        ("Exercise Compliance", 0.50,        "#f59e0b", "Update after exercises"),
        ("Overall Function",    overall,     "#8b5cf6", "Composite score"),
    ]
    for label, score, color, note in components:
        y = _progress_bar(ax, 0, y, 0.60, score, label, color=color)
        ax.text(0.72, y + 0.010, note, fontsize=6.5, va="top", color=SLATE)
    y -= 0.020
 
    # ── Clinical safety card ──────────────────────────────────────────────────
    y = _section_label(ax, 0, y, "Recovery Safety Guidance — Seek Medical Attention If:",
                       color="#b91c1c")
    safety_items = red_flags + when_help if red_flags or when_help else [
        "Severe or worsening pain develops suddenly",
        "Swelling increases rapidly or significantly",
        "Numbness, tingling, or loss of sensation occurs",
        "Visible deformity or instability appears",
        "Limb function suddenly worsens",
        "Skin colour changes (pallor, cyanosis, mottling)",
        "Fever or signs of infection develop",
    ]
    _card(ax, 0, y, 1, min(len(safety_items[:7]) * 0.028 + 0.025, 0.22),
          bg="#fff1f2", border="#fca5a5")
    for item in safety_items[:7]:
        if y < 0.20:
            break
        y = _add_text(ax, 0.015, y, f"[!] {item}", size=8.5, color="#991b1b")
    y -= 0.015
 
    # ── Recommendation reasoning ───────────────────────────────────────────────
    reasoning = analysis.get("recommendation_reasoning") or {}
    factors   = reasoning.get("factors_used") or []
    if factors and y > 0.28:
        y = _section_label(ax, 0, y, "Why These Recommendations Were Generated")
        for factor in factors[:7]:
            if y < 0.20:
                break
            y = _add_text(ax, 0.015, y, f"• {factor}", size=8)
        y -= 0.005
 
    if y > 0.12:
        _render_disclaimer_banner(ax, y if y > 0.12 else 0.12)
 
    pdf.savefig(fig)
    plt.close(fig)
 
 
# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
 
def build_analysis_report_pdf(analysis: dict,
                               uploaded_image_path: Path | None = None) -> bytes:
    """
Return a comprehensive AI-assisted clinical report as bytes. 
    Pages
    -----
    1  Executive Summary
    2  AI Findings + Confidence + Model Transparency
    3  Image Review + Heatmap Analysis          (only if images available)
    4  Clinical Risk Assessment + Recovery Timeline
    5  Exercise Plan
    6  Nutrition Plan
    7  Progress Tracking + Clinical Safety
    """
    output = BytesIO()
    with PdfPages(output) as pdf:
        _render_executive_summary(pdf, analysis)
        _render_ai_findings_page(pdf, analysis)
 
        prediction    = analysis.get("prediction") or {}
        explainability= prediction.get("explainability") or {}
        original  = _image_from_path(uploaded_image_path)
        heatmap   = _image_from_data_url(explainability.get("overlay_image"))
        if original or heatmap:
            _render_image_review_page(pdf, analysis, uploaded_image_path)
 
        _render_risk_timeline_page(pdf, analysis)
        _render_exercise_page_v2(pdf, analysis)
        _render_nutrition_page_v2(pdf, analysis, page_num=6)
        _render_progress_safety_page(pdf, analysis)
 
    return output.getvalue()
 
 
def build_nutrition_report_pdf(analysis: dict) -> bytes:
    """Return a focused nutrition-plan PDF as bytes."""
    output = BytesIO()
    with PdfPages(output) as pdf:
        _render_nutrition_page_v2(pdf, analysis, page_num=1,
                                  title="AI Physio Diet Plan")
    return output.getvalue()
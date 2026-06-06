"""PDF report generation for AI Physio analyses."""

from __future__ import annotations

import base64
import textwrap
from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from PIL import Image


def _title_case(value: object) -> str:
    text = str(value or "-").replace("_", " ")
    return " ".join(part.capitalize() for part in text.split())


def _format_percent(value: object) -> str:
    if not isinstance(value, (int, float)):
        return "-"
    return f"{value * 100:.2f}%"


def _wrap(text: object, width: int = 88) -> str:
    clean = " ".join(str(text or "").split())
    if not clean:
        return "-"
    return "\n".join(textwrap.wrap(clean, width=width))


def _add_text(ax, x: float, y: float, text: object, size: int = 10, weight: str = "normal") -> float:
    wrapped = _wrap(text)
    ax.text(x, y, wrapped, fontsize=size, fontweight=weight, va="top", color="#172033")
    return y - (0.038 * max(1, wrapped.count("\n") + 1))


def _image_from_data_url(data_url: str | None) -> Image.Image | None:
    if not data_url or "," not in data_url:
        return None
    try:
        encoded = data_url.split(",", 1)[1]
        return Image.open(BytesIO(base64.b64decode(encoded))).convert("RGB")
    except Exception:
        return None


def _image_from_path(path: Path | None) -> Image.Image | None:
    if not path or not path.exists():
        return None
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


def _new_page(pdf: PdfPages):
    fig = plt.figure(figsize=(8.27, 11.69))
    ax = fig.add_axes([0.07, 0.06, 0.86, 0.9])
    ax.axis("off")
    return fig, ax


def _write_page_number(ax, page_number: int) -> None:
    ax.text(0.5, -0.025, f"AI Physio report | Page {page_number}", ha="center", fontsize=8, color="#607086")


def _render_nutrition_page(pdf: PdfPages, analysis: dict, page_number: int, title: str = "Recovery Nutrition Plan") -> None:
    recovery = analysis.get("recovery_guidance") or {}
    nutrition = recovery.get("nutrition_plan") or {}
    patient_context = analysis.get("patient_context") or {}

    fig, ax = _new_page(pdf)
    y = 0.98
    ax.text(0, y, title, fontsize=20, fontweight="bold", va="top", color="#0f766e")
    y -= 0.052
    y = _add_text(ax, 0, y, nutrition.get("summary") or "Balanced recovery meals with protein, calcium, vitamin D, and hydration.", size=10)
    y -= 0.02

    if patient_context:
        context_text = "; ".join(f"{_title_case(key)}: {value}" for key, value in patient_context.items())
        y = _add_text(ax, 0, y, f"Personal context: {context_text}", size=9)
        y -= 0.015

    ax.text(0, y, "Daily Targets", fontsize=13, fontweight="bold", va="top", color="#172033")
    y -= 0.04
    for target in (nutrition.get("daily_targets") or [])[:5]:
        y = _add_text(ax, 0.02, y, f"- {target}", size=9)
    y -= 0.02

    ax.text(0, y, "Full-Day Meal Plan", fontsize=13, fontweight="bold", va="top", color="#172033")
    y -= 0.045
    for meal in nutrition.get("meals") or []:
        if y < 0.18:
            break
        row_height = 0.125
        ax.add_patch(plt.Rectangle((0, y - row_height), 1, row_height - 0.01, color="#f8fafc", ec="#d7e2ea"))
        ax.text(0.025, y - 0.026, meal.get("time") or "Meal", fontsize=8, fontweight="bold", va="top", color="#0f766e")
        ax.text(0.18, y - 0.026, meal.get("name") or "Recovery meal", fontsize=10, fontweight="bold", va="top", color="#172033")
        details = meal.get("why") or ""
        items = ", ".join(meal.get("items") or [])
        _add_text(ax, 0.18, y - 0.058, f"{details} Foods: {items}", size=8)
        y -= 0.145

    y -= 0.01
    ax.text(0, y, "Avoid or Limit", fontsize=13, fontweight="bold", va="top", color="#172033")
    y -= 0.04
    for item in (nutrition.get("avoid") or [])[:4]:
        y = _add_text(ax, 0.02, y, f"- {item}", size=9)

    y -= 0.02
    _add_text(ax, 0, y, nutrition.get("disclaimer") or "This diet plan is educational and should be adjusted by a clinician or dietitian when needed.", size=9)
    _write_page_number(ax, page_number)
    pdf.savefig(fig)
    plt.close(fig)


def _render_physio_page(pdf: PdfPages, analysis: dict, page_number: int) -> None:
    recovery = analysis.get("recovery_guidance") or {}
    physio = recovery.get("physio_support") or {}
    recommendations = analysis.get("recommendations") or {}

    fig, ax = _new_page(pdf)
    y = 0.98
    ax.text(0, y, "Physio Recovery Support", fontsize=20, fontweight="bold", va="top", color="#0f766e")
    y -= 0.055
    y = _add_text(ax, 0, y, f"{physio.get('phase') or 'Recovery phase'}: {physio.get('summary') or recommendations.get('summary')}", size=10)
    y -= 0.02

    ax.text(0, y, "Daily Focus", fontsize=13, fontweight="bold", va="top", color="#172033")
    y -= 0.04
    for item in (physio.get("daily_focus") or [])[:5]:
        y = _add_text(ax, 0.02, y, f"- {item}", size=9)
    y -= 0.025

    ax.text(0, y, "Exercise Media", fontsize=13, fontweight="bold", va="top", color="#172033")
    y -= 0.04
    for exercise in (recommendations.get("exercises") or [])[:5]:
        media = exercise.get("media") or {}
        text = (
            f"{exercise.get('name')}: {exercise.get('description')} "
            f"Frequency: {exercise.get('frequency') or '-'}; "
            f"YouTube search: {media.get('youtube_url') or '-'}"
        )
        y = _add_text(ax, 0.02, y, text, size=8)
        y -= 0.008
        if y < 0.11:
            break

    _add_text(ax, 0, 0.12, physio.get("exercise_media_note") or "Use media for form education only and stop if pain increases.", size=9)
    _write_page_number(ax, page_number)
    pdf.savefig(fig)
    plt.close(fig)


def build_analysis_report_pdf(analysis: dict, uploaded_image_path: Path | None = None) -> bytes:
    """Return a clinical-style PDF report as bytes."""
    prediction = analysis.get("prediction") or {}
    recommendations = analysis.get("recommendations") or {}
    patient_context = analysis.get("patient_context") or {}
    llm = analysis.get("llm_explanation") or {}
    explainability = prediction.get("explainability") or {}
    class_scores = prediction.get("class_scores") or {}

    output = BytesIO()
    with PdfPages(output) as pdf:
        fig, ax = _new_page(pdf)
        y = 0.98
        ax.text(0, y, "AI Physio Analysis Report", fontsize=22, fontweight="bold", va="top", color="#0f766e")
        y -= 0.055
        y = _add_text(
            ax,
            0,
            y,
            "Educational X-ray screening result. This report is not a medical diagnosis and requires clinical confirmation.",
            size=10,
        )
        y -= 0.03

        metrics = [
            ("Possible Pattern", _title_case(prediction.get("label"))),
            ("Condition", _title_case(prediction.get("condition"))),
            ("Confidence", _format_percent(prediction.get("confidence"))),
        ]
        for index, (label, value) in enumerate(metrics):
            x = index * 0.33
            ax.add_patch(plt.Rectangle((x, y - 0.08), 0.3, 0.075, color="#eef6f6", ec="#c7d8df"))
            ax.text(x + 0.015, y - 0.022, label, fontsize=8, fontweight="bold", color="#607086")
            ax.text(x + 0.015, y - 0.055, value, fontsize=12, fontweight="bold", color="#172033")
        y -= 0.13

        if patient_context:
            ax.text(0, y, "Patient Context", fontsize=13, fontweight="bold", va="top", color="#172033")
            y -= 0.035
            for key, value in patient_context.items():
                y = _add_text(ax, 0.02, y, f"{_title_case(key)}: {value}", size=9)
            y -= 0.02

        ax.text(0, y, "AI Explanation", fontsize=13, fontweight="bold", va="top", color="#172033")
        y -= 0.04
        for label, value in [
            ("Summary", llm.get("summary")),
            ("Confidence", llm.get("confidence_note")),
            ("Next steps", llm.get("next_steps")),
            ("Safety", llm.get("safety_note")),
        ]:
            ax.text(0, y, f"{label}:", fontsize=9, fontweight="bold", va="top", color="#172033")
            y = _add_text(ax, 0.18, y, value, size=9)
            y -= 0.012

        y -= 0.015
        ax.text(0, y, "Class Probabilities", fontsize=13, fontweight="bold", va="top", color="#172033")
        y -= 0.04
        if class_scores:
            for label, score in class_scores.items():
                y = _add_text(ax, 0.02, y, f"{_title_case(label)}: {_format_percent(score)}", size=9)
        else:
            y = _add_text(ax, 0.02, y, "No class probability data available.", size=9)

        y -= 0.02
        ax.text(0, y, "Recommendations", fontsize=13, fontweight="bold", va="top", color="#172033")
        y -= 0.04
        y = _add_text(ax, 0, y, recommendations.get("summary"), size=9)
        y = _add_text(ax, 0, y - 0.01, f"When to see doctor: {recommendations.get('when_to_see_doctor') or '-'}", size=9)
        y -= 0.02
        for exercise in (recommendations.get("exercises") or [])[:4]:
            text = (
                f"{exercise.get('name')}: {exercise.get('description')} "
                f"Frequency: {exercise.get('frequency') or '-'}; "
                f"Sets: {exercise.get('sets') or '-'}; Reps: {exercise.get('reps') or '-'}."
            )
            y = _add_text(ax, 0.02, y, text, size=8)
            if y < 0.08:
                break

        _write_page_number(ax, 1)
        pdf.savefig(fig)
        plt.close(fig)

        original = _image_from_path(uploaded_image_path)
        heatmap = _image_from_data_url(explainability.get("overlay_image"))
        if original or heatmap:
            fig, ax = _new_page(pdf)
            ax.text(0, 0.98, "Image Review", fontsize=18, fontweight="bold", va="top", color="#0f766e")

            images = [("Uploaded X-ray", original), ("Model Focus Heatmap", heatmap)]
            available = [(title, image) for title, image in images if image is not None]
            for index, (title, image) in enumerate(available):
                x = 0.0 if index == 0 else 0.52
                ax.text(x, 0.9, title, fontsize=12, fontweight="bold", va="top", color="#172033")
                image_ax = fig.add_axes([0.07 + x * 0.86, 0.48, 0.38, 0.36])
                image_ax.imshow(image)
                image_ax.axis("off")

            note = explainability.get("note") or "Heatmaps show model focus regions only, not confirmed injury locations."
            _add_text(ax, 0, 0.38, note, size=10)
            _add_text(ax, 0, 0.3, prediction.get("disclaimer") or "Educational use only. Not a diagnostic tool.", size=10)
            _write_page_number(ax, 2)
            pdf.savefig(fig)
            plt.close(fig)

        _render_physio_page(pdf, analysis, 3)
        _render_nutrition_page(pdf, analysis, 4)

    return output.getvalue()


def build_nutrition_report_pdf(analysis: dict) -> bytes:
    """Return a focused diet-plan PDF as bytes."""
    output = BytesIO()
    with PdfPages(output) as pdf:
        _render_nutrition_page(pdf, analysis, 1, "AI Physio Diet Plan")
    return output.getvalue()

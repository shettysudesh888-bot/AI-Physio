"""
Patient-friendly LLM explanation layer for AI Physio.

The vision model remains the source of truth. This module only turns the
structured prediction and recommendation data into bounded educational text.
"""

import json
import os
import re


GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_GROQ_MODEL = os.getenv("AI_PHYSIO_LLM_MODEL", "openai/gpt-oss-20b")


SYSTEM_INSTRUCTIONS = """
You explain AI Physio X-ray screening results in plain language.
Rules:
- Do not diagnose, confirm, or rule out disease.
- Do not add conditions, findings, treatments, or medications not present in the input.
- Keep the explanation educational and cautious.
- Mention that clinical confirmation is required for concerning or uncertain findings.
- Use the model prediction, confidence, patient intake, recommendations, and doctor advice as the only source.
- If patient intake is present, personalize precautions without changing the model result.
- Return concise JSON with keys: summary, confidence_note, heatmap_note, next_steps, safety_note.
""".strip()


def _format_percent(value: float | None) -> str:
    if not isinstance(value, (int, float)):
        return "unknown confidence"
    return f"{value * 100:.1f}% confidence"


def _fallback_explanation(
    prediction: dict,
    recommendations: dict,
    reason: str,
    patient_context: dict | None = None,
) -> dict:
    label = str(prediction.get("label") or "unknown").replace("_", " ")
    condition = str(prediction.get("condition") or "unknown").replace("_", " ")
    confidence = _format_percent(prediction.get("confidence"))
    doctor_advice = recommendations.get("when_to_see_doctor") or "Consult a qualified clinician."
    body_part = (patient_context or {}).get("body_part")
    pain_level = (patient_context or {}).get("pain_level")
    summary = (
        f"The screening model reported {label} ({condition}) with {confidence}. "
        "This is an educational AI result, not a medical diagnosis."
    )
    if body_part:
        summary += f" The uploaded scan was marked as related to the {body_part}."
    if pain_level is not None:
        summary += f" Reported pain level: {pain_level}/10."

    return {
        "enabled": False,
        "model": None,
        "summary": summary,
        "confidence_note": (
            "Use the confidence score as model context only. It does not prove whether an injury is present."
        ),
        "heatmap_note": (
            "If a heatmap is shown, it highlights image regions that influenced the model score, "
            "not a confirmed injury location."
        ),
        "next_steps": doctor_advice,
        "safety_note": "Seek professional medical care for pain, swelling, deformity, numbness, or loss of function.",
        "status": reason,
    }


def _provider_config() -> dict | None:
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        return {
            "provider": "Groq",
            "api_key": groq_key,
            "base_url": GROQ_BASE_URL,
            "model": DEFAULT_GROQ_MODEL,
        }

    return None


def _parse_json_text(text: str) -> dict:
    cleaned = text.strip()
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        cleaned = match.group(0)
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    return json.loads(cleaned)


def _safe_error_status(provider_name: str, exc: Exception) -> str:
    message = str(exc)
    lowered = message.lower()

    if "api key" in lowered or "invalid_api_key" in lowered or "401" in lowered:
        return f"{provider_name} API key failed; using local fallback"
    if "rate" in lowered or "429" in lowered:
        return f"{provider_name} rate limit reached; using local fallback"
    if "quota" in lowered:
        return f"{provider_name} quota unavailable; using local fallback"
    if "model" in lowered and ("not" in lowered or "does not" in lowered or "404" in lowered):
        return f"{provider_name} model unavailable; using local fallback"
    if isinstance(exc, json.JSONDecodeError):
        return f"{provider_name} returned non-JSON text; using local fallback"

    return f"{provider_name} request failed; using local fallback"


def explain_result(
    prediction: dict,
    recommendations: dict,
    patient_context: dict | None = None,
) -> dict:
    """
    Generate a patient-friendly explanation using the configured LLM provider.

    Groq uses an OpenAI-compatible endpoint, so the official OpenAI Python
    client is used as the HTTP client. Falls back to a deterministic explanation
    when the client package or API key is not configured, so local demos remain
    reliable.
    """
    provider = _provider_config()
    if not provider:
        return _fallback_explanation(
            prediction,
            recommendations,
            "GROQ_API_KEY not configured",
            patient_context,
        )

    try:
        from openai import OpenAI
    except ImportError:
        return _fallback_explanation(
            prediction,
            recommendations,
            f"{provider['provider']} client dependency not installed (openai package)",
            patient_context,
        )

    payload = {
        "prediction": {
            "label": prediction.get("label"),
            "condition": prediction.get("condition"),
            "confidence": prediction.get("confidence"),
            "class_scores": prediction.get("class_scores"),
            "model_mode": prediction.get("model_mode"),
            "disclaimer": prediction.get("disclaimer"),
            "explainability": {
                "method": (prediction.get("explainability") or {}).get("method"),
                "note": (prediction.get("explainability") or {}).get("note"),
            },
        },
        "recommendations": {
            "summary": recommendations.get("summary"),
            "severity": recommendations.get("severity"),
            "when_to_see_doctor": recommendations.get("when_to_see_doctor"),
            "medical_disclaimer": recommendations.get("medical_disclaimer"),
        },
        "patient_context": patient_context or {},
    }

    try:
        client_kwargs = {"api_key": provider["api_key"]}
        if provider["base_url"]:
            client_kwargs["base_url"] = provider["base_url"]
        client = OpenAI(**client_kwargs)
        response = client.chat.completions.create(
            model=provider["model"],
            messages=[
                {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                {
                    "role": "user",
                    "content": (
                        "Create a patient-friendly explanation for this structured AI screening result. "
                        "Return only JSON.\n"
                        f"{json.dumps(payload, ensure_ascii=True)}"
                    ),
                },
            ],
            temperature=0.2,
        )
        output_text = response.choices[0].message.content or "{}"
        parsed = _parse_json_text(output_text)
        return {
            "enabled": True,
            "provider": provider["provider"],
            "model": provider["model"],
            "summary": str(parsed.get("summary", "")).strip(),
            "confidence_note": str(parsed.get("confidence_note", "")).strip(),
            "heatmap_note": str(parsed.get("heatmap_note", "")).strip(),
            "next_steps": str(parsed.get("next_steps", "")).strip(),
            "safety_note": str(parsed.get("safety_note", "")).strip(),
            "status": "generated",
        }
    except Exception as exc:
        status = _safe_error_status(provider["provider"], exc)
        return _fallback_explanation(prediction, recommendations, status, patient_context)

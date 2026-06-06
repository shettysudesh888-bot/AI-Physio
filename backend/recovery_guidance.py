"""Physio and nutrition guidance for recovery support."""

from __future__ import annotations

import json
import os
import re
from urllib.parse import quote_plus


GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_GROQ_MODEL = os.getenv("AI_PHYSIO_LLM_MODEL", "llama-3.1-8b-instant")  # Fixed: was "openai/gpt-oss-20b"

HIGH_RISK_CONDITIONS = {"fracture", "joint_dislocation", "bone_tumor"}

EXERCISE_MEDIA = {
    "deep breathing": {
        "image_url": "https://images.unsplash.com/photo-1506126613408-eca07ce68773?auto=format&fit=crop&w=900&q=80",
        "youtube_url": "https://www.youtube.com/results?search_query=diaphragmatic+breathing+exercise+physiotherapy",
    },
    "unaffected-limb gentle range of motion": {
        "image_url": "https://images.unsplash.com/photo-1571019613914-85f342c6a11e?auto=format&fit=crop&w=900&q=80",
        "youtube_url": "https://www.youtube.com/results?search_query=gentle+range+of+motion+exercise+physiotherapy",
    },
    "weight-bearing walking": {
        "image_url": "https://images.unsplash.com/photo-1486218119243-13883505764c?auto=format&fit=crop&w=900&q=80",
        "youtube_url": "https://www.youtube.com/results?search_query=walking+exercise+bone+health+physiotherapy",
    },
    "calf raises": {
        "image_url": "https://images.unsplash.com/photo-1576678927484-cc907957088c?auto=format&fit=crop&w=900&q=80",
        "youtube_url": "https://www.youtube.com/results?search_query=calf+raise+exercise+proper+form+physiotherapy",
    },
    "hip abduction": {
        "image_url": "https://images.unsplash.com/photo-1518611012118-696072aa579a?auto=format&fit=crop&w=900&q=80",
        "youtube_url": "https://www.youtube.com/results?search_query=side+lying+hip+abduction+physiotherapy",
    },
    "tai chi": {
        "image_url": "https://images.unsplash.com/photo-1540206395-68808572332f?auto=format&fit=crop&w=900&q=80",
        "youtube_url": "https://www.youtube.com/results?search_query=tai+chi+balance+exercise+beginners",
    },
    "wall push-ups": {
        "image_url": "https://images.unsplash.com/photo-1599058917212-d750089bc07e?auto=format&fit=crop&w=900&q=80",
        "youtube_url": "https://www.youtube.com/results?search_query=wall+push+up+exercise+physiotherapy",
    },
    "sit-to-stand": {
        "image_url": "https://images.unsplash.com/photo-1571019613454-1cb2f99b2d8b?auto=format&fit=crop&w=900&q=80",
        "youtube_url": "https://www.youtube.com/results?search_query=sit+to+stand+exercise+physiotherapy",
    },
    "water aerobics or hydrotherapy": {
        "image_url": "https://images.unsplash.com/photo-1530549387789-4c1017266635?auto=format&fit=crop&w=900&q=80",
        "youtube_url": "https://www.youtube.com/results?search_query=water+aerobics+arthritis+exercise",
    },
    "chair yoga": {
        "image_url": "https://images.unsplash.com/photo-1602192509154-0b900ee1f851?auto=format&fit=crop&w=900&q=80",
        "youtube_url": "https://www.youtube.com/results?search_query=chair+yoga+for+joint+pain",
    },
    "grip strengthening": {
        "image_url": "https://images.unsplash.com/photo-1581009137042-c552e485697a?auto=format&fit=crop&w=900&q=80",
        "youtube_url": "https://www.youtube.com/results?search_query=hand+grip+strengthening+physiotherapy",
    },
    "rest and protect the area": {
        "image_url": "https://images.unsplash.com/photo-1516841273335-e39b37888115?auto=format&fit=crop&w=900&q=80",
        "youtube_url": "https://www.youtube.com/results?search_query=RICE+injury+rest+ice+compression+elevation",
    },
}

FOOD_IMAGES = {
    "Greek yogurt bowl": "https://images.unsplash.com/photo-1488477181946-6428a0291777?auto=format&fit=crop&w=900&q=80",
    "Lentil dal with rice": "https://images.unsplash.com/photo-1546833999-b9f581a1996d?auto=format&fit=crop&w=900&q=80",
    "Paneer and vegetable plate": "https://images.unsplash.com/photo-1567188040759-fb8a883dc6d8?auto=format&fit=crop&w=900&q=80",
    "Salmon with greens": "https://images.unsplash.com/photo-1467003909585-2f8a72700288?auto=format&fit=crop&w=900&q=80",
    "Eggs and whole-grain toast": "https://images.unsplash.com/photo-1525351484163-7529414344d8?auto=format&fit=crop&w=900&q=80",
    "Chicken quinoa bowl": "https://images.unsplash.com/photo-1512621776951-a57141f2eefd?auto=format&fit=crop&w=900&q=80",
}


SYSTEM_INSTRUCTIONS = """
You create cautious recovery support for an AI Physio educational app.
Rules:
- Do not diagnose, prescribe medication, or replace a clinician.
- For fracture, dislocation, tumor, uncertain result, high pain, swelling, or trauma, prioritize medical review and protection.
- Keep exercise advice gentle and only use the exercise names supplied in the input.
- Build a practical nutrition plan for recovery using ordinary foods.
- Return concise JSON with keys: physio_support, nutrition_plan.
""".strip()

ASSISTANT_SYSTEM_INSTRUCTIONS = """
You are the AI Physio recovery assistant.
Rules:
- You are educational only and must not diagnose, prescribe medication, or replace a clinician.
- If symptoms include severe pain, deformity, numbness, open wound, fever, inability to bear weight, swelling after trauma, or possible fracture/dislocation, advise urgent clinical review.
- Use the provided context fields: condition, confidence, risk_level, phase, exercises, body_part.
- Give practical, concise answers with clear safety limits.
- Do not invent a confirmed diagnosis.
""".strip()


def _provider_config() -> dict | None:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    return {
        "provider": "Groq",
        "api_key": api_key,
        "base_url": GROQ_BASE_URL,
        "model": DEFAULT_GROQ_MODEL,
    }


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


def _media_for_exercise(name: str) -> dict:
    key = str(name or "").lower().strip()
    media = EXERCISE_MEDIA.get(key)
    if media:
        return dict(media)
    query = quote_plus(f"{name} physiotherapy exercise proper form")
    return {
        "image_url": "https://images.unsplash.com/photo-1571019613454-1cb2f99b2d8b?auto=format&fit=crop&w=900&q=80",
        "youtube_url": f"https://www.youtube.com/results?search_query={query}",
    }


def enrich_exercise_media(recommendations: dict) -> dict:
    enriched = dict(recommendations or {})
    exercises = []
    for exercise in enriched.get("exercises") or []:
        item = dict(exercise)
        item["media"] = _media_for_exercise(item.get("name", "exercise"))
        exercises.append(item)
    enriched["exercises"] = exercises
    return enriched


def _risk_level(condition: str, patient_context: dict) -> str:
    pain = patient_context.get("pain_level")
    high_symptoms = bool(patient_context.get("swelling") or patient_context.get("recent_trauma"))
    if condition in HIGH_RISK_CONDITIONS or high_symptoms or (isinstance(pain, int) and pain >= 6):
        return "needs_clinician_review"
    return "low_risk_support"


def _compact_analysis(analysis: dict) -> dict:
    """
    Strip everything the LLM doesn't need to answer a question.
    Reduces token usage significantly — no media URLs, no full meal plans,
    no base64 overlays, no verbose nested dicts.
    """
    pred = analysis.get("prediction") or {}
    guidance = analysis.get("recovery_guidance") or {}
    recs = analysis.get("recommendations") or {}
    physio = guidance.get("physio_support") or {}
    nutrition = guidance.get("nutrition_plan") or {}

    return {
        "condition": pred.get("condition"),
        "confidence": pred.get("confidence"),
        "decision_status": (pred.get("decision") or {}).get("status"),
        "body_part": (analysis.get("patient_context") or {}).get("body_part"),
        "risk_level": physio.get("risk_level"),
        "phase": physio.get("phase"),
        "daily_focus": physio.get("daily_focus"),
        # Only exercise names — not full objects with image/youtube URLs
        "exercises": [e.get("name") for e in (recs.get("exercises") or []) if e.get("name")],
        # Only the avoid list from nutrition — not full meal plans with images
        "nutrition_avoid": nutrition.get("avoid"),
        "nutrition_daily_targets": nutrition.get("daily_targets"),
    }


def _fallback_guidance(prediction: dict, recommendations: dict, patient_context: dict, status: str) -> dict:
    condition = str(prediction.get("condition") or "").lower()
    body_part = patient_context.get("body_part") or "affected area"
    risk_level = _risk_level(condition, patient_context)
    high_risk = risk_level == "needs_clinician_review"

    if high_risk:
        phase = "Protection phase"
        plan_summary = (
            f"Protect the {body_part} and avoid loading, resistance, stretching, massage, or sport "
            "until a clinician confirms the injury and clears movement."
        )
        daily_focus = [
            "Rest the painful area and keep it supported.",
            "Move only unaffected joints gently if comfortable.",
            "Seek urgent care for deformity, numbness, severe swelling, open wound, or inability to bear weight.",
        ]
    else:
        phase = "Gentle recovery phase"
        plan_summary = (
            f"Use pain-free movement for the {body_part}, then progress slowly when symptoms remain mild "
            "and settle within 24 hours."
        )
        daily_focus = [
            "Start with gentle range-of-motion and easy walking if comfortable.",
            "Keep pain at 0-3/10 during and after activity.",
            "Reduce intensity if swelling, stiffness, or pain increases the next day.",
        ]

    meals = [
        {
            "time": "Breakfast",
            "name": "Greek yogurt bowl",
            "why": "Protein, calcium, fruit, and nuts support bone and soft-tissue recovery.",
            "items": ["Greek yogurt or curd", "Banana or berries", "Almonds or walnuts", "Water"],
            "image_url": FOOD_IMAGES["Greek yogurt bowl"],
        },
        {
            "time": "Lunch",
            "name": "Lentil dal with rice",
            "why": "Lentils add protein, iron, magnesium, and steady carbohydrates for healing energy.",
            "items": ["Dal or beans", "Rice or roti", "Leafy vegetables", "Curd if tolerated"],
            "image_url": FOOD_IMAGES["Lentil dal with rice"],
        },
        {
            "time": "Dinner",
            "name": "Paneer and vegetable plate",
            "why": "Calcium-rich paneer plus colorful vegetables helps recovery without heavy processing.",
            "items": ["Paneer, tofu, eggs, fish, or chicken", "Cooked vegetables", "Whole grain", "Fruit"],
            "image_url": FOOD_IMAGES["Paneer and vegetable plate"],
        },
    ]

    if (patient_context.get("sex") or "").lower() == "male":
        meals[2]["name"] = "Chicken quinoa bowl"
        meals[2]["image_url"] = FOOD_IMAGES["Chicken quinoa bowl"]

    return {
        "enabled": False,
        "status": status,
        "physio_support": {
            "phase": phase,
            "risk_level": risk_level,
            "summary": plan_summary,
            "daily_focus": daily_focus,
            "exercise_media_note": "Use videos only for form education. Stop if pain rises, and follow clinician advice first.",
        },
        "nutrition_plan": {
            "title": "7-day recovery nutrition plan",
            "summary": "Aim for protein at each meal, calcium and vitamin D support, hydration, and colorful whole foods.",
            "daily_targets": [
                "Protein with every meal unless medically restricted.",
                "Calcium-rich foods such as milk, curd, paneer, tofu, ragi, sesame, or leafy greens.",
                "Vitamin D from safe sunlight, eggs, fortified foods, or clinician-advised supplements.",
                "2-3 liters water daily unless a doctor has restricted fluids.",
            ],
            "meals": meals,
            "avoid": [
                "Smoking and excessive alcohol because they can slow bone healing.",
                "Skipping meals during recovery.",
                "Very high-sugar snacks replacing protein-rich meals.",
            ],
            "disclaimer": "Diet needs vary with diabetes, kidney disease, allergies, pregnancy, medications, and cultural preferences.",
        },
    }


def build_recovery_guidance(prediction: dict, recommendations: dict, patient_context: dict | None = None) -> dict:
    context = patient_context or {}
    provider = _provider_config()
    fallback = _fallback_guidance(prediction, recommendations, context, "GROQ_API_KEY not configured")

    if not provider:
        return fallback

    try:
        from openai import OpenAI
    except ImportError:
        fallback["status"] = "Groq client dependency not installed (openai package)"
        return fallback

    # Slim payload: only what the LLM needs — no media URLs, no base64
    payload = {
        "prediction": {
            "condition": prediction.get("condition"),
            "confidence": prediction.get("confidence"),
            "decision": prediction.get("decision"),
        },
        "patient_context": context,
        "recommendation_summary": recommendations.get("summary"),
        "exercise_names": [item.get("name") for item in (recommendations.get("exercises") or []) if item.get("name")],
        "dietary_tips": recommendations.get("dietary_tips"),
    }

    try:
        client = OpenAI(api_key=provider["api_key"], base_url=provider["base_url"])
        response = client.chat.completions.create(
            model=provider["model"],
            messages=[
                {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                {"role": "user", "content": f"Return JSON only.\n{json.dumps(payload, ensure_ascii=True)}"},
            ],
            temperature=0.2,
        )
        parsed = _parse_json_text(response.choices[0].message.content or "{}")
        fallback["enabled"] = True
        fallback["provider"] = provider["provider"]
        fallback["model"] = provider["model"]
        fallback["status"] = "generated"
        if isinstance(parsed.get("physio_support"), dict):
            fallback["physio_support"].update(parsed["physio_support"])
        if isinstance(parsed.get("nutrition_plan"), dict):
            fallback["nutrition_plan"].update(parsed["nutrition_plan"])
        return fallback
    except Exception as exc:
        fallback["status"] = _safe_error_status(provider["provider"], exc)
        return fallback


def answer_recovery_question(
    question: str,
    analysis: dict | None = None,
    patient_context: dict | None = None,
) -> dict:
    """Answer a user's recovery question with Groq when configured."""
    cleaned_question = " ".join(str(question or "").split())
    if not cleaned_question:
        return {
            "enabled": False,
            "status": "empty question",
            "answer": "Please type a recovery, exercise, symptom, or diet question.",
            "safety_note": "For severe or worsening symptoms, contact a qualified clinician.",
        }

    safe_fallback = {
        "enabled": False,
        "status": "GROQ_API_KEY not configured",
        "answer": (
            "I can help with general recovery guidance. Keep activity gentle and pain-free, avoid loading a painful area "
            "after a fall or trauma, and follow the report's red-flag guidance. If pain is severe, swelling is present, "
            "or movement is limited, get clinical review before exercise."
        ),
        "safety_note": "Seek urgent care for deformity, numbness, open wound, severe pain, swelling after trauma, or inability to bear weight.",
    }

    provider = _provider_config()
    if not provider:
        return safe_fallback

    try:
        from openai import OpenAI
    except ImportError:
        safe_fallback["status"] = "Groq client dependency not installed (openai package)"
        return safe_fallback

    # FIX: Use compact analysis instead of full nested dicts.
    # Previously sent full recommendations + recovery_guidance which included
    # media URLs, base64 images, and verbose meal plans — hundreds of wasted tokens.
    compact_context = _compact_analysis(analysis or {})

    # Merge explicit patient_context on top (takes precedence)
    if patient_context:
        compact_context["patient_context"] = patient_context
    elif analysis and analysis.get("patient_context"):
        compact_context["patient_context"] = analysis["patient_context"]

    payload = {
        "question": cleaned_question,
        "context": compact_context,
    }

    try:
        client = OpenAI(api_key=provider["api_key"], base_url=provider["base_url"])
        response = client.chat.completions.create(
            model=provider["model"],
            messages=[
                {"role": "system", "content": ASSISTANT_SYSTEM_INSTRUCTIONS},
                {
                    "role": "user",
                    "content": (
                        "Answer this user question for the AI Physio app. Return JSON only with keys: "
                        "answer, safety_note, suggested_next_step.\n"
                        f"{json.dumps(payload, ensure_ascii=True)}"
                    ),
                },
            ],
            temperature=0.25,
        )
        parsed = _parse_json_text(response.choices[0].message.content or "{}")
        return {
            "enabled": True,
            "provider": provider["provider"],
            "model": provider["model"],
            "status": "generated",
            "answer": str(parsed.get("answer") or "").strip() or safe_fallback["answer"],
            "safety_note": str(parsed.get("safety_note") or "").strip() or safe_fallback["safety_note"],
            "suggested_next_step": str(parsed.get("suggested_next_step") or "").strip(),
        }
    except Exception as exc:
        safe_fallback["status"] = _safe_error_status(provider["provider"], exc)
        return safe_fallback
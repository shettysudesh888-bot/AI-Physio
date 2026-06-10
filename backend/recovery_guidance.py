"""
Physio and nutrition guidance for recovery support.

Architecture
------------
1. compute_exercise_eligibility()  — deterministic rule engine (6 rules)
2. BODY_PART_GUIDANCE              — per-body-part, per-stage guidance tables
3. build_recovery_guidance()       — assembles full guidance payload; optionally
                                     enhanced by Groq LLM when configured
4. answer_recovery_question()      — Groq-powered Q&A assistant
"""

from __future__ import annotations

import json
import os
import re
from urllib.parse import quote_plus


GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_GROQ_MODEL = os.getenv("AI_PHYSIO_LLM_MODEL", "llama-3.1-8b-instant")

# ---------------------------------------------------------------------------
# Exercise eligibility rules
# ---------------------------------------------------------------------------

ELIGIBILITY_RULES = [
    # (label, condition_fn, reason)
    # Each lambda receives: (bp, stage, approval, restrictions, mobility)
    # Pain level is checked separately in compute_exercise_eligibility below.
    (
        "skull",
        lambda bp, stage, approval, restrictions, mobility: bp == "skull",
        "Skull injuries require complete rest and medical supervision. No exercise plan is generated.",
    ),
    (
        "spine_acute",
        lambda bp, stage, approval, restrictions, mobility: (
            bp == "spine" and stage == "acute_phase"
        ),
        "Spinal fractures in the acute phase require rest and specialist follow-up. No exercise plan at this stage.",
    ),
    (
        "no_approval",
        lambda bp, stage, approval, restrictions, mobility: approval == "no",
        "Your doctor has not advised exercises at this stage.",
    ),
    (
        # BUG FIX: "not_sure" on exercise approval should block — not treating unknown approval as cleared.
        "approval_unsure",
        lambda bp, stage, approval, restrictions, mobility: approval == "not_sure",
        "It is unclear whether your doctor has approved exercises. Please confirm clearance with your clinician before starting any exercises.",
    ),
    (
        "exercise_restricted",
        lambda bp, stage, approval, restrictions, mobility: restrictions == "exercise_restricted",
        "Exercise is restricted by your doctor. Follow medical guidance before beginning any exercises.",
    ),
    (
        # BUG FIX: "not_sure" on restrictions means restrictions are unknown — conservative block.
        "restrictions_unsure",
        lambda bp, stage, approval, restrictions, mobility: restrictions == "not_sure",
        "Your doctor's restrictions are unknown. Confirm what is safe with your clinician before beginning any exercises.",
    ),
    (
        "severely_limited",
        lambda bp, stage, approval, restrictions, mobility: mobility == "severely_limited",
        "Severely limited mobility prevents safe exercise at this stage. Focus on rest and clinical management.",
    ),
]


def compute_exercise_eligibility(patient_context: dict) -> dict:
    """
    Apply the eligibility decision tree.

    Checks pain level first (BUG FIX: previously pain_level was ignored here),
    then applies the ordered ELIGIBILITY_RULES.

    Returns:
        {
            "eligible": bool,
            "reason": str | None,   # human-readable reason when not eligible
            "rule_triggered": str | None,
        }
    """
    bp = (patient_context.get("body_part") or "").lower().strip()
    stage = (patient_context.get("recovery_stage") or "").lower().strip()
    approval = (patient_context.get("exercise_approval") or "").lower().strip()
    restrictions = (patient_context.get("doctor_restrictions") or "").lower().strip()
    mobility = (patient_context.get("mobility_status") or "").lower().strip()

    # BUG FIX: Pain level was previously ignored in eligibility (only a red-flag warning was raised).
    # Severe pain (>=8) must block exercises, not just display a banner.
    pain_level = patient_context.get("pain_level")
    try:
        pain_level = int(pain_level) if pain_level is not None else 0
    except (TypeError, ValueError):
        pain_level = 0

    if pain_level >= 8:
        return {
            "eligible": False,
            "reason": (
                f"Pain level reported as {pain_level}/10. Exercises are not safe at this pain level. "
                "Seek clinical review before starting any rehabilitation."
            ),
            "rule_triggered": "high_pain",
        }

    for label, condition_fn, reason in ELIGIBILITY_RULES:
        try:
            if condition_fn(bp, stage, approval, restrictions, mobility):
                return {"eligible": False, "reason": reason, "rule_triggered": label}
        except Exception:
            pass

    return {"eligible": True, "reason": None, "rule_triggered": None}


# ---------------------------------------------------------------------------
# Recovery Risk Level
# ---------------------------------------------------------------------------

WHEN_TO_SEEK_HELP = [
    "Sudden or significant increase in pain not relieved by rest.",
    "New or worsening swelling, redness, or warmth around the injury.",
    "Numbness, tingling, or loss of sensation in the injured area or limb.",
    "Loss or significant reduction of movement since last assessment.",
    "Fever (temperature > 38°C / 100.4°F) near or around the injury site.",
    "Visible deformity, wound breakdown, or discharge from a surgical site.",
    "Cast-related complications: tightness, pressure sores, or cast cracks.",
    "Any symptom that is new, worsening, or that concerns you.",
]


def compute_recovery_risk_level(patient_context: dict) -> dict:
    """
    Compute a Recovery Risk Level (High / Moderate / Low) based on clinical context.
    Returns: {"level": str, "label": str, "color": str, "reasons": list[str]}
    """
    pain_level = 0
    try:
        pain_level = int(patient_context.get("pain_level") or 0)
    except (TypeError, ValueError):
        pain_level = 0

    swelling = (patient_context.get("swelling_level") or "").lower().strip()
    mobility = (patient_context.get("mobility_status") or "").lower().strip()
    stage = (patient_context.get("recovery_stage") or "").lower().strip()
    treatment = (patient_context.get("treatment_status") or "").lower().strip()
    restrictions = (patient_context.get("doctor_restrictions") or "").lower().strip()

    reasons = []
    high_score = 0
    moderate_score = 0

    # High risk conditions
    if pain_level >= 8:
        high_score += 2
        reasons.append(f"High pain level ({pain_level}/10).")
    if swelling == "severe":
        high_score += 2
        reasons.append("Severe swelling reported.")
    if mobility == "severely_limited":
        high_score += 2
        reasons.append("Severely limited mobility.")
    if stage == "acute_phase" and treatment == "surgery_performed":
        high_score += 2
        reasons.append("Acute phase post-surgery.")
    if restrictions in ("exercise_restricted",):
        high_score += 1
        reasons.append("Exercises restricted by doctor.")

    # Moderate risk conditions
    if 5 <= pain_level <= 7:
        moderate_score += 2
        reasons.append(f"Moderate pain level ({pain_level}/10).")
    if swelling == "moderate":
        moderate_score += 1
        reasons.append("Moderate swelling reported.")
    if mobility == "moderately_limited":
        moderate_score += 1
        reasons.append("Moderately limited mobility.")
    if restrictions in ("movement_restricted", "weight_bearing_restricted"):
        moderate_score += 1
        reasons.append("Doctor-imposed movement restrictions in place.")
    if stage == "acute_phase" and treatment not in ("surgery_performed",):
        moderate_score += 1
        reasons.append("Acute recovery phase.")

    if high_score >= 2:
        return {"level": "high", "label": "High Risk", "color": "#ef4444", "reasons": reasons}
    if moderate_score >= 2:
        return {"level": "moderate", "label": "Moderate Risk", "color": "#f59e0b", "reasons": reasons}
    return {"level": "low", "label": "Low Risk", "color": "#10b981", "reasons": reasons}


# ---------------------------------------------------------------------------
# Body-part recovery guidance tables
# ---------------------------------------------------------------------------

BODY_PART_GUIDANCE: dict[str, dict] = {
    "skull": {
        "all_stages": {
            "dos": [
                "Rest completely in a quiet, dark room during the acute phase.",
                "Follow strict medical supervision for all activities.",
                "Keep all scheduled neurology or emergency follow-up appointments.",
                "Stay hydrated and maintain nutrition as tolerated.",
                "Report any new or worsening symptoms (vomiting, confusion, vision changes) immediately.",
            ],
            "avoids": [
                "Any physical exertion or sport.",
                "Screen use (phone, TV, computer) — limit if causing headache.",
                "Bending forward or sudden head movements.",
                "Driving or operating machinery.",
                "Alcohol and non-prescribed medications.",
            ],
            "focus": "Rest, observation, and medical supervision. No exercise plan is appropriate for skull fractures.",
            "timeline": [
                "Immediate: Ensure clinical evaluation and imaging.",
                "Days 1–14: Complete rest, symptom monitoring, zero exertion.",
                "Weeks 2–6: Gradual return to light activity only as cleared by neurologist.",
                "Beyond 6 weeks: Return to normal activities determined by specialist review.",
            ],
        },
        "acute_phase": None,
        "early_recovery": None,
        "late_recovery": None,
    },

    "spine": {

        "acute_phase": {
            "dos": [
                "Rest in a position of comfort — usually lying flat or in a supported semi-reclined position.",
                "Use prescribed pain relief as directed by your clinician.",
                "Keep all specialist (orthopaedics or neurosurgery) appointments.",
                "Log roll when turning in bed — do not twist the spine.",
                "Maintain nutrition and hydration to support bone healing.",
            ],
            "avoids": [
                "Bending the spine forward (flexion).",
                "Twisting or rotating the trunk.",
                "Lifting any weight.",
                "Sitting for prolonged periods without support.",
                "All exercise until specialist clearance is obtained.",
            ],
            "focus": "Rest and specialist follow-up. No exercise plan is generated during acute spinal fracture management.",
            "timeline": [
                "Week 1–2: Strict rest, pain management, imaging review.",
                "Week 2–4: Begin gentle positional changes under guidance.",
                "Week 4–6: Transition to early recovery protocol if cleared.",
            ],
        },
        "early_recovery": {
            "dos": [
                "Perform prescribed gentle exercises (pelvic tilts, breathing) in lying position.",
                "Begin short walks on flat ground if cleared by clinician.",
                "Use a supportive brace or corset as prescribed.",
                "Maintain good posture when sitting — use lumbar support.",
                "Keep the spine in a neutral position at all times.",
            ],
            "avoids": [
                "Bending at the waist — bend at the knees instead.",
                "Carrying or lifting objects.",
                "Twisting movements.",
                "High-impact activities (running, jumping).",
                "Prolonged sitting without breaks.",
            ],
            "focus": "Gradual mobility restoration with strict spinal neutral control.",
            "timeline": [
                "Week 2–4: Gentle ROM exercises in lying.",
                "Week 4–6: Sitting and short walking tolerance.",
                "Week 6+: Progress to late recovery exercises if cleared.",
            ],
        },
        "late_recovery": {
            "dos": [
                "Perform core stabilisation exercises (cat-camel, bird-dog) as prescribed.",
                "Progress walking duration gradually.",
                "Maintain good body mechanics for all daily activities.",
                "Engage in hydrotherapy if available and cleared.",
                "Follow up with physiotherapist regularly.",
            ],
            "avoids": [
                "Heavy lifting or manual labour.",
                "High-impact sport.",
                "Prolonged static postures without movement breaks.",
                "Exercises that cause pain or neurological symptoms.",
            ],
            "focus": "Core strengthening, posture, and functional movement restoration.",
            "timeline": [
                "Week 6–10: Core stability exercises begin.",
                "Week 10–16: Functional strengthening, walking progression.",
                "Beyond 16 weeks: Return to light work/activity with specialist sign-off.",
            ],
        },
    },

    "shoulder": {

        "acute_phase": {
            "dos": [
                "Rest the shoulder in a sling or supported position as instructed.",
                "Apply ice (wrapped in cloth) for 15 minutes every 2-3 hours to reduce swelling.",
                "Perform pendulum exercises if cleared by your clinician.",
                "Gently move the elbow and wrist to prevent stiffness in those joints.",
                "Keep elevation above heart level when possible.",
            ],
            "avoids": [
                "Lifting the arm above shoulder height.",
                "Carrying any weight with the injured arm.",
                "Forceful shoulder movements.",
                "Sleeping on the injured shoulder.",
                "Driving until cleared by clinician.",
            ],
            "focus": "Protection, swelling control, and pain management.",
            "timeline": [
                "Days 1–7: Sling use, ice, rest, gentle pendulums if cleared.",
                "Week 1–3: Distal joint ROM (elbow, wrist, fingers).",
                "Week 3–6: Transition to early recovery program.",
            ],
        },
        "early_recovery": {
            "dos": [
                "Begin active-assisted shoulder ROM exercises.",
                "Perform wall slides and gentle isometric exercises.",
                "Sleep in a supported position — avoid lying on injured side.",
                "Continue gentle elbow and wrist exercises.",
                "Apply ice after exercise for 10-15 minutes.",
            ],
            "avoids": [
                "Lifting arm above 90 degrees without clearance.",
                "Carrying heavy objects.",
                "Pulling or pushing movements.",
                "Behind-the-back stretches.",
            ],
            "focus": "Gradual restoration of shoulder range of motion with pain control.",
            "timeline": [
                "Week 3–6: Active-assisted ROM, isometric strengthening.",
                "Week 6–8: Progress to active ROM.",
                "Week 8+: Begin resistance if cleared.",
            ],
        },
        "late_recovery": {
            "dos": [
                "Progress to resistance band and light weight strengthening.",
                "Work on rotator cuff strengthening exercises.",
                "Focus on scapular stability (wall press, retraction).",
                "Gradually return to functional activities.",
                "Continue physiotherapy program.",
            ],
            "avoids": [
                "Heavy overhead lifting without clearance.",
                "Throwing or impact activities before full strength recovery.",
                "Exercises that reproduce pain.",
            ],
            "focus": "Strength, stability, and functional restoration.",
            "timeline": [
                "Week 8–12: Resistance training begins.",
                "Week 12–16: Return to sport-specific training if applicable.",
                "Beyond 16 weeks: Full functional restoration with specialist clearance.",
            ],
        },
    },

    "elbow": {

        "acute_phase": {
            "dos": [
                "Apply ice wrapped in cloth for 15 minutes every 2-3 hours.",
                "Elevate the elbow above heart level to reduce swelling.",
                "Gently move the fingers and wrist to prevent stiffness.",
                "Use a sling or splint as prescribed.",
                "Take prescribed pain relief as directed.",
            ],
            "avoids": [
                "Any forceful bending or straightening of the elbow.",
                "Lifting with the injured arm.",
                "Forearm rotation movements.",
                "Applying heat to the injury site in the first 72 hours.",
            ],
            "focus": "Protection, swelling reduction, and distal joint mobility.",
            "timeline": [
                "Days 1–7: RICE protocol, finger ROM, elevation.",
                "Week 1–3: Gentle wrist exercises added.",
                "Week 3–4: Transition to early recovery.",
            ],
        },
        "early_recovery": {
            "dos": [
                "Begin active elbow flexion and extension in pain-free range.",
                "Add forearm pronation and supination exercises.",
                "Use heat before exercise to improve tissue mobility.",
                "Ice after exercise to control swelling.",
            ],
            "avoids": [
                "Forced stretching at the limits of range.",
                "Lifting objects heavier than 0.5 kg.",
                "Contact sports or impact activities.",
            ],
            "focus": "Range of motion restoration and pain control.",
            "timeline": [
                "Week 3–6: Active ROM restoration.",
                "Week 6–8: Begin light functional activities.",
                "Week 8+: Progress to late recovery exercises.",
            ],
        },
        "late_recovery": {
            "dos": [
                "Progress to resistance band and light dumbbell exercises.",
                "Work on grip and forearm strengthening.",
                "Practice functional activities such as carrying light objects.",
                "Continue physiotherapy program.",
            ],
            "avoids": [
                "Heavy manual labour without clearance.",
                "Racquet sports or throwing before full strength is recovered.",
            ],
            "focus": "Strength restoration and return to function.",
            "timeline": [
                "Week 8–12: Resistance strengthening.",
                "Week 12+: Return to full activity with specialist clearance.",
            ],
        },
    },

    "wrist_hand": {

        "acute_phase": {
            "dos": [
                "Keep the wrist elevated above heart level at all times when resting.",
                "Apply ice (wrapped in cloth) for 15 minutes every 2-3 hours.",
                "Perform gentle finger flexion and extension exercises.",
                "Use a splint, cast, or brace as prescribed.",
                "Keep fingers moving to prevent stiffness.",
            ],
            "avoids": [
                "Gripping, pinching, or weight-bearing through the hand.",
                "Removing the cast or splint without medical guidance.",
                "Wrist movement if the wrist itself is fractured.",
                "Hot water or heat application in the first 72 hours.",
            ],
            "focus": "Protection, elevation, swelling control, and finger mobility.",
            "timeline": [
                "Days 1–7: Elevation, ice, finger ROM.",
                "Week 1–6: Cast/immobilisation period (varies by fracture type).",
                "After cast removal: Transition to early recovery exercises.",
            ],
        },
        "early_recovery": {
            "dos": [
                "Begin active wrist ROM exercises after cast removal or on clinician's advice.",
                "Perform wrist flexion, extension, and radial/ulnar deviation exercises.",
                "Use therapeutic putty for hand and grip exercises.",
                "Apply heat before exercise for 10 minutes to improve tissue mobility.",
                "Ice after exercise for 10-15 minutes.",
            ],
            "avoids": [
                "Forced wrist stretching.",
                "Gripping heavy objects.",
                "Pushing through sharp pain.",
                "Returning to contact sports or heavy manual work.",
            ],
            "focus": "Wrist ROM restoration, oedema management, and hand function.",
            "timeline": [
                "Week 6–8 (after cast removal): Active wrist ROM begins.",
                "Week 8–10: Light grip and putty exercises.",
                "Week 10–12: Progress to late recovery.",
            ],
        },
        "late_recovery": {
            "dos": [
                "Progress to grip strengthening with exercise tools.",
                "Add wrist curls and extensions with very light weights.",
                "Practice functional tasks — writing, keyboard use, light cooking.",
                "Continue physiotherapy if ROM is not fully restored.",
            ],
            "avoids": [
                "Heavy gripping or impact before full bone union is confirmed.",
                "Racquet sports, gymnastics, or impact activities without clearance.",
            ],
            "focus": "Strength, grip endurance, and return to full hand function.",
            "timeline": [
                "Week 10–14: Light resistance exercises.",
                "Week 14–16: Progressive functional activities.",
                "Beyond 16 weeks: Return to sport or work with specialist clearance.",
            ],
        },
    },

    "pelvis_hip": {

        "acute_phase": {
            "dos": [
                "Remain on strict rest as prescribed — mobilise only with authorised assistance.",
                "Perform ankle pumps every hour to reduce DVT risk.",
                "Complete prescribed static exercises (gluteal sets) in bed.",
                "Use walking aids exactly as instructed — do not exceed prescribed weight-bearing status.",
                "Keep all follow-up imaging and specialist appointments.",
            ],
            "avoids": [
                "Crossing the legs (especially important post-hip surgery).",
                "Bending the hip past 90 degrees.",
                "Internal rotation of the hip.",
                "Weight-bearing beyond what is prescribed.",
                "Getting up from a low chair or toilet without raised seat.",
            ],
            "focus": "Protection, DVT prevention, and controlled mobilisation.",
            "timeline": [
                "Days 1–7: Bed rest, ankle pumps, static exercises.",
                "Week 1–3: Begin sitting and standing with support if cleared.",
                "Week 3–6: Transition to early recovery protocol.",
            ],
        },
        "early_recovery": {
            "dos": [
                "Begin seated and lying hip exercises (abduction, knee-to-chest).",
                "Practice sit-to-stand transfers with arm support.",
                "Walk short distances with walking aids as prescribed.",
                "Use hip precautions if post-surgical (avoid flexion >90°, adduction, internal rotation).",
            ],
            "avoids": [
                "Hip flexion beyond 90 degrees (post-surgical precaution).",
                "Crossing legs.",
                "Bending to pick up objects from the floor.",
                "Low seating or soft sofas.",
                "Driving before cleared by surgeon.",
            ],
            "focus": "Controlled mobility, hip precaution compliance, and progressive walking.",
            "timeline": [
                "Week 3–6: Gentle hip ROM, walking progression.",
                "Week 6–8: Progress to late recovery exercises.",
            ],
        },
        "late_recovery": {
            "dos": [
                "Perform bridging, clamshells, and progressive hip strengthening.",
                "Walk progressively longer distances.",
                "Engage in hydrotherapy if available.",
                "Work with physiotherapist on functional activities.",
            ],
            "avoids": [
                "High-impact exercise (running, jumping) without specialist clearance.",
                "Heavy manual labour before confirmed bone union.",
            ],
            "focus": "Hip strength, gait restoration, and functional independence.",
            "timeline": [
                "Week 8–12: Progressive strengthening.",
                "Week 12–16: Return to community walking and light activities.",
                "Beyond 16 weeks: Return to sport with specialist clearance.",
            ],
        },
    },

    "femur": {

        "acute_phase": {
            "dos": [
                "Perform ankle pumps every hour — femur fractures carry high DVT risk.",
                "Complete static quad sets in bed if cleared.",
                "Move uninjured limbs and upper body regularly.",
                "Follow weight-bearing instructions precisely — even partial loading must be authorised.",
                "Keep all post-operative or specialist reviews.",
            ],
            "avoids": [
                "Bearing weight on the injured leg beyond the prescribed amount.",
                "Rotating the leg outward or inward forcefully.",
                "Sitting with the injured hip in extreme flexion.",
                "Any exercise that causes thigh or groin pain.",
            ],
            "focus": "DVT prevention, pain management, and authorised mobilisation.",
            "timeline": [
                "Days 1–7: Ankle pumps, static exercises, authorised position changes.",
                "Week 1–4: Crutch or frame walking as prescribed.",
                "Week 4–8: Transition to early recovery.",
            ],
        },
        "early_recovery": {
            "dos": [
                "Begin straight-leg raises and heel slides in lying position.",
                "Walk with prescribed walking aids within the authorised weight-bearing status.",
                "Perform seated knee bending as tolerated.",
                "Ice and elevate after exercise sessions.",
            ],
            "avoids": [
                "Full weight-bearing before cleared.",
                "Pivoting or twisting on the injured leg.",
                "High steps or stairs without handrail and clearance.",
            ],
            "focus": "Controlled weight-bearing progression and quadriceps activation.",
            "timeline": [
                "Week 4–8: Walking with support, lying exercises.",
                "Week 8–12: Transition to full weight-bearing if cleared.",
                "Week 12+: Progress to late recovery.",
            ],
        },
        "late_recovery": {
            "dos": [
                "Progress to full weight-bearing walking.",
                "Add step-ups, mini squats, and progressive strengthening.",
                "Focus on quadriceps, hamstring, and hip abductor strength.",
                "Work with physiotherapist on gait correction.",
            ],
            "avoids": [
                "Running or impact sport before bone union is confirmed on imaging.",
                "Heavy squats or leg press before cleared by specialist.",
            ],
            "focus": "Strength, gait, and return to full function.",
            "timeline": [
                "Week 12–20: Full weight-bearing exercises.",
                "Week 20–26: Return to light work or sport.",
                "Beyond 26 weeks: Return to unrestricted activity with specialist clearance.",
            ],
        },
    },

    "knee": {

        "acute_phase": {
            "dos": [
                "Apply ice (wrapped in cloth) for 15-20 minutes every 2-3 hours.",
                "Elevate the leg above heart level to reduce swelling.",
                "Perform static quad sets and ankle pumps.",
                "Use a brace, splint, or crutches as prescribed.",
                "Avoid bearing weight beyond the prescribed amount.",
            ],
            "avoids": [
                "Full weight-bearing without clearance.",
                "Bending the knee past the pain-free range.",
                "Twisting or pivoting on the injured leg.",
                "High-impact activities.",
            ],
            "focus": "Swelling control, quad activation, and protected mobilisation.",
            "timeline": [
                "Days 1–7: RICE, quad sets, ankle pumps.",
                "Week 1–3: Begin straight-leg raises, partial weight-bearing if cleared.",
                "Week 3–6: Transition to early recovery.",
            ],
        },
        "early_recovery": {
            "dos": [
                "Begin heel slides and seated knee bending exercises.",
                "Walk with crutches or walking aids as prescribed.",
                "Add hip strengthening exercises (abduction, straight-leg raise).",
                "Use ice after exercise.",
                "Monitor swelling and report worsening to clinician.",
            ],
            "avoids": [
                "Full squats or deep knee bends.",
                "Running or jumping.",
                "Standing for prolonged periods without rest.",
                "Twisting or pivoting on the knee.",
            ],
            "focus": "Range of motion restoration and muscle activation.",
            "timeline": [
                "Week 3–6: Active ROM, supported walking.",
                "Week 6–8: Transition to full weight-bearing if cleared.",
                "Week 8+: Progress to late recovery.",
            ],
        },
        "late_recovery": {
            "dos": [
                "Perform mini squats, step-ups, and terminal knee extensions.",
                "Progress to full weight-bearing walking on varied terrain.",
                "Work on proprioception and balance exercises.",
                "Continue physiotherapy program.",
            ],
            "avoids": [
                "Running before quad strength is at least 80% of uninjured side.",
                "Pivoting or cutting movements without clearance.",
                "High-impact sport before full ROM and strength are recovered.",
            ],
            "focus": "Strength, stability, proprioception, and return to function.",
            "timeline": [
                "Week 8–12: Strengthening and balance.",
                "Week 12–16: Return to light physical activity.",
                "Beyond 16 weeks: Return to sport with specialist clearance.",
            ],
        },
    },

    "lower_leg": {

        "acute_phase": {
            "dos": [
                "Elevate the lower leg above heart level continuously when resting.",
                "Perform ankle pumps every 30-60 minutes.",
                "Wiggle the toes frequently to maintain circulation.",
                "Use crutches or walking frame — non-weight-bearing unless told otherwise.",
                "Keep the cast or brace dry and intact.",
            ],
            "avoids": [
                "Putting weight on the injured leg.",
                "Hanging the leg below heart level for extended periods.",
                "Removing any cast or brace without medical clearance.",
                "Walking without prescribed walking aids.",
            ],
            "focus": "Swelling control, circulation, and protected rest.",
            "timeline": [
                "Days 1–7: RICE, ankle pumps, elevation.",
                "Week 1–6: Cast or brace immobilisation period.",
                "After cast removal: Begin early recovery exercises.",
            ],
        },
        "early_recovery": {
            "dos": [
                "Begin seated calf raises and ankle circles.",
                "Progress weight-bearing as prescribed.",
                "Ice after exercise for 10-15 minutes.",
                "Perform gentle knee bending in sitting to maintain knee mobility.",
            ],
            "avoids": [
                "Running or jumping.",
                "Uneven or wet surfaces.",
                "Removing brace for walking unless instructed.",
            ],
            "focus": "Ankle and calf mobility, progressive weight-bearing.",
            "timeline": [
                "Week 6–10: ROM exercises, partial weight-bearing.",
                "Week 10–12: Transition to full weight-bearing.",
                "Week 12+: Progress to late recovery.",
            ],
        },
        "late_recovery": {
            "dos": [
                "Progress to standing calf raises and single-leg balance.",
                "Walk increasing distances on flat ground.",
                "Work on balance and proprioception training.",
                "Return to stairs and mild inclines progressively.",
            ],
            "avoids": [
                "Running or impact sport before bone union is confirmed.",
                "Uneven surfaces or hiking trails without clearance.",
            ],
            "focus": "Strength, balance, and return to full walking.",
            "timeline": [
                "Week 12–16: Strengthening and balance exercises.",
                "Week 16–20: Return to community activities.",
                "Beyond 20 weeks: Full activity with specialist clearance.",
            ],
        },
    },

    "ankle_foot": {

        "acute_phase": {
            "dos": [
                "Elevate the ankle above heart level as much as possible.",
                "Perform ankle pumps every 30-60 minutes.",
                "Apply ice wrapped in a cloth for 15 minutes every 2-3 hours.",
                "Non-weight-bear using crutches or boot as prescribed.",
                "Wiggle the toes frequently.",
            ],
            "avoids": [
                "Any weight-bearing on the injured ankle unless specifically cleared.",
                "Hot water or heat on the ankle in the first 72 hours.",
                "Removing boot or cast without clinical instruction.",
                "Hanging foot below heart level for extended periods.",
            ],
            "focus": "RICE protocol, swelling control, and strict non-weight-bearing.",
            "timeline": [
                "Days 1–7: RICE, ankle pumps, elevation.",
                "Week 1–6: Boot or cast immobilisation.",
                "After immobilisation: Begin early recovery.",
            ],
        },
        "early_recovery": {
            "dos": [
                "Begin ankle alphabet circles and calf stretches after clearance.",
                "Start seated calf raises.",
                "Progress weight-bearing as prescribed — usually with a boot.",
                "Ice the ankle after each exercise session.",
                "Wear supportive footwear for all walking.",
            ],
            "avoids": [
                "Barefoot walking on hard floors.",
                "Running, jumping, or sports.",
                "Uneven surfaces.",
                "Stairs without handrail.",
            ],
            "focus": "Ankle ROM restoration, swelling management, and weight-bearing progression.",
            "timeline": [
                "Week 6–8: ROM exercises, partial weight-bearing.",
                "Week 8–10: Full weight-bearing in boot.",
                "Week 10–12: Transition to normal footwear and late recovery.",
            ],
        },
        "late_recovery": {
            "dos": [
                "Progress to standing calf raises and single-leg balance.",
                "Practice heel-to-toe walking for balance.",
                "Add resistance band ankle exercises.",
                "Work on proprioception on flat and then soft surfaces.",
            ],
            "avoids": [
                "Running before single-leg balance is stable and pain-free.",
                "Impact sport without specialist clearance.",
                "High heels or unsupportive footwear.",
            ],
            "focus": "Strength, proprioception, and return to full mobility.",
            "timeline": [
                "Week 10–14: Strengthening and balance.",
                "Week 14–16: Walking on varied surfaces.",
                "Week 16+: Return to sport with specialist clearance.",
            ],
        },
    },
}


# ---------------------------------------------------------------------------
# Treatment-status modifiers
# ---------------------------------------------------------------------------

TREATMENT_OVERLAY: dict[str, list[str]] = {
    "surgery_performed": [
        "Follow all post-operative wound care instructions.",
        "Monitor the wound daily for signs of infection (redness, warmth, discharge, fever).",
        "Keep all post-operative review appointments.",
        "Do not progress exercises beyond the surgeon's protocol.",
    ],
    "cast_plaster": [
        "Keep the cast dry at all times — use waterproof covers for showering.",
        "Do not insert objects inside the cast to scratch the skin.",
        "Report any numbness, increasing pain, or pressure sores to your clinician.",
        "Wiggle fingers/toes regularly to maintain circulation.",
    ],
    "brace_support": [
        "Wear the brace as instructed — do not loosen or remove it without guidance.",
        "Check that the brace fits correctly and does not cause pressure sores.",
        "Perform exercises with the brace on unless specifically told to remove it.",
    ],
    "not_evaluated": [
        "Seek clinical evaluation and imaging confirmation before starting any rehabilitation.",
        "Do not make major changes to your activity level until assessed.",
    ],
    "physiotherapy_started": [
        "Coordinate all exercises with your physiotherapist's program.",
        "Do not add exercises from other sources without checking with your physio first.",
    ],
}

# GAP FIX: Doctor restriction advisories — shown as additional daily-focus dos.
# Previously movement_restricted and weight_bearing_restricted had no advisory text at all.
RESTRICTION_OVERLAY: dict[str, list[str]] = {
    "weight_bearing_restricted": [
        "Non-weight-bearing status: use crutches, frame, or walking aids exactly as prescribed.",
        "Do not place any weight on the injured limb until your clinician clears it.",
        "Perform all exercises in lying or seated positions only.",
    ],
    "movement_restricted": [
        "Movement restricted by your clinician: only perform exercises they have explicitly cleared.",
        "Avoid any motion that causes pain above 3/10 at the injury site.",
        "Progress only when your clinician advises — do not self-progress.",
    ],
    "exercise_restricted": [
        "Exercise is restricted: follow medical advice strictly before starting any rehabilitation.",
    ],
}


# ---------------------------------------------------------------------------
# Stage-aware nutrition
# ---------------------------------------------------------------------------

FOOD_IMAGES = {
    "Greek yogurt bowl": "https://images.unsplash.com/photo-1488477181946-6428a0291777?auto=format&fit=crop&w=900&q=80",
    "Lentil dal with rice": "https://images.unsplash.com/photo-1546833999-b9f581a1996d?auto=format&fit=crop&w=900&q=80",
    "Paneer and vegetable plate": "https://images.unsplash.com/photo-1567188040759-fb8a883dc6d8?auto=format&fit=crop&w=900&q=80",
    "Salmon with greens": "https://images.unsplash.com/photo-1467003909585-2f8a72700288?auto=format&fit=crop&w=900&q=80",
    "Eggs and whole-grain toast": "https://images.unsplash.com/photo-1525351484163-7529414344d8?auto=format&fit=crop&w=900&q=80",
    "Chicken quinoa bowl": "https://images.unsplash.com/photo-1512621776951-a57141f2eefd?auto=format&fit=crop&w=900&q=80",
    "Anti-inflammatory bowl": "https://images.unsplash.com/photo-1546069901-ba9599a7e63c?auto=format&fit=crop&w=900&q=80",
    "High-protein meal": "https://images.unsplash.com/photo-1547592166-23ac45744acd?auto=format&fit=crop&w=900&q=80",
}

STAGE_NUTRITION: dict[str, dict] = {
    "acute_phase": {
        "title": "Acute Phase Nutrition Plan (0–2 Weeks)",
        "summary": (
            "Focus on anti-inflammatory foods, adequate calcium, Vitamin D, and hydration. "
            "The body requires controlled energy intake during rest."
        ),
        "daily_targets": [
            "Calcium: 1200 mg/day — milk, curd, paneer, ragi, sesame, leafy greens.",
            "Vitamin D: 1000–2000 IU/day — safe sunlight, eggs, fortified milk, clinician-advised supplements.",
            "Protein: 1.2–1.5 g/kg body weight — with every main meal.",
            "Omega-3: salmon, walnuts, flaxseed — natural anti-inflammatory support.",
            "Hydration: 2.5–3 litres water daily.",
            "Turmeric and ginger: anti-inflammatory — add to meals or warm drinks.",
        ],
        "avoid": [
            "Smoking — significantly slows bone healing.",
            "Excessive alcohol — impairs bone repair and medication effectiveness.",
            "Ultra-processed foods high in sodium — worsen inflammation.",
            "Sugary drinks replacing protein-rich meals.",
        ],
        "meals": [
            {
                "time": "Breakfast",
                "name": "Greek yogurt bowl",
                "why": "High-protein, calcium-rich start with anti-inflammatory berries and anti-oxidant nuts.",
                "items": ["Greek yogurt or curd", "Berries or banana", "Walnuts or almonds", "Honey (optional)", "Water"],
                "image_url": FOOD_IMAGES["Greek yogurt bowl"],
            },
            {
                "time": "Lunch",
                "name": "Anti-inflammatory bowl",
                "why": "Turmeric-spiced dal provides protein, iron, and anti-inflammatory compounds.",
                "items": ["Dal or chickpeas", "Brown rice or roti", "Turmeric-spiced vegetables", "Curd"],
                "image_url": FOOD_IMAGES["Anti-inflammatory bowl"],
            },
            {
                "time": "Dinner",
                "name": "Paneer and vegetable plate",
                "why": "Calcium-rich paneer with colourful vegetables supports bone repair without heavy processing.",
                "items": ["Paneer, tofu, eggs, fish, or chicken", "Cooked seasonal vegetables", "Whole grain", "Warm turmeric milk"],
                "image_url": FOOD_IMAGES["Paneer and vegetable plate"],
            },
        ],
        "disclaimer": "Diet needs vary with diabetes, kidney disease, allergies, pregnancy, and medications. Consult a clinician or dietitian.",
    },
    "early_recovery": {
        "title": "Early Recovery Nutrition Plan (2–6 Weeks)",
        "summary": (
            "Increase protein and collagen-supporting nutrients for tissue repair. "
            "Zinc and Vitamin C support wound healing and bone remodelling."
        ),
        "daily_targets": [
            "Protein: 1.5–1.8 g/kg body weight — essential for bone callus formation.",
            "Calcium: 1200 mg/day — same as acute phase, maintain consistently.",
            "Vitamin C: 500 mg/day — citrus, amla, guava, bell peppers — supports collagen synthesis.",
            "Zinc: 15–25 mg/day — lean meat, pumpkin seeds, lentils — bone repair.",
            "Vitamin D: 1000–2000 IU/day — continue as per acute phase.",
            "Hydration: 2.5–3 litres water daily.",
        ],
        "avoid": [
            "Smoking and excessive alcohol — continue strict avoidance.",
            "Caffeine in very large amounts — may reduce calcium absorption.",
            "Skipping meals — bone healing is energy-intensive.",
        ],
        "meals": [
            {
                "time": "Breakfast",
                "name": "Eggs and whole-grain toast",
                "why": "Eggs provide complete protein and Vitamin D. Whole grain adds zinc and complex carbohydrates.",
                "items": ["2 eggs (boiled or scrambled)", "Whole-grain toast", "Glass of fortified milk", "Orange or amla"],
                "image_url": FOOD_IMAGES["Eggs and whole-grain toast"],
            },
            {
                "time": "Lunch",
                "name": "Chicken quinoa bowl",
                "why": "Quinoa is a complete protein with zinc. Chicken provides high-quality protein for bone callus.",
                "items": ["Grilled chicken or paneer", "Quinoa or brown rice", "Bell peppers, spinach", "Lemon dressing"],
                "image_url": FOOD_IMAGES["Chicken quinoa bowl"],
            },
            {
                "time": "Dinner",
                "name": "High-protein meal",
                "why": "Evening protein supports overnight bone repair processes.",
                "items": ["Fish, chicken, tofu, or lentils", "Steamed vegetables", "Roti or rice", "Curd or buttermilk"],
                "image_url": FOOD_IMAGES["High-protein meal"],
            },
        ],
        "disclaimer": "Diet needs vary with diabetes, kidney disease, allergies, pregnancy, and medications. Consult a clinician or dietitian.",
    },
    "late_recovery": {
        "title": "Late Recovery Nutrition Plan (6+ Weeks)",
        "summary": (
            "Shift toward energy balance for resumed activity. "
            "Maintain calcium and protein for bone consolidation. "
            "Add complex carbohydrates for exercise energy."
        ),
        "daily_targets": [
            "Protein: 1.2–1.6 g/kg body weight — maintain for ongoing bone strengthening.",
            "Calcium: 1000–1200 mg/day — maintain throughout recovery.",
            "Complex carbohydrates: whole grains, legumes, sweet potato — fuel for rehabilitation exercises.",
            "Iron: lean red meat, dark leafy greens, lentils — combat fatigue from recovery.",
            "Magnesium: nuts, seeds, whole grains — supports muscle function and bone health.",
            "Hydration: 2.5–3 litres daily, increasing with exercise.",
        ],
        "avoid": [
            "Smoking and alcohol — remain harmful to bone density.",
            "Ultra-processed snacks replacing nutrient-dense meals.",
            "Excessive caloric restriction during active rehabilitation.",
        ],
        "meals": [
            {
                "time": "Breakfast",
                "name": "Greek yogurt bowl",
                "why": "Sustained energy with protein, calcium, and healthy fats for morning exercise.",
                "items": ["Greek yogurt", "Mixed fruit", "Granola or oats", "Nuts and seeds", "Water"],
                "image_url": FOOD_IMAGES["Greek yogurt bowl"],
            },
            {
                "time": "Lunch",
                "name": "Salmon with greens",
                "why": "Omega-3 from salmon reduces residual inflammation. Leafy greens provide calcium and iron.",
                "items": ["Grilled salmon or mackerel", "Leafy green salad", "Whole grain", "Lemon and olive oil"],
                "image_url": FOOD_IMAGES["Salmon with greens"],
            },
            {
                "time": "Dinner",
                "name": "Lentil dal with rice",
                "why": "Iron-rich lentils with complex carbohydrates support energy and overnight recovery.",
                "items": ["Lentil dal", "Brown rice or roti", "Seasonal vegetables", "Curd"],
                "image_url": FOOD_IMAGES["Lentil dal with rice"],
            },
        ],
        "disclaimer": "Diet needs vary with diabetes, kidney disease, allergies, pregnancy, and medications. Consult a clinician or dietitian.",
    },
}


# ---------------------------------------------------------------------------
# Exercise media library (for enriching exercise cards in UI)
# ---------------------------------------------------------------------------

EXERCISE_MEDIA = {
    "pendulum circles": {
        "image_url": "https://images.unsplash.com/photo-1571019613914-85f342c6a11e?auto=format&fit=crop&w=900&q=80",
        "youtube_url": "https://www.youtube.com/results?search_query=pendulum+shoulder+exercise+physiotherapy",
    },
    "ankle pumps": {
        "image_url": "https://images.unsplash.com/photo-1518611012118-696072aa579a?auto=format&fit=crop&w=900&q=80",
        "youtube_url": "https://www.youtube.com/results?search_query=ankle+pumps+exercise+physiotherapy",
    },
    "diaphragmatic breathing": {
        "image_url": "https://images.unsplash.com/photo-1506126613408-eca07ce68773?auto=format&fit=crop&w=900&q=80",
        "youtube_url": "https://www.youtube.com/results?search_query=diaphragmatic+breathing+exercise+physiotherapy",
    },
    "pelvic tilt (supine)": {
        "image_url": "https://images.unsplash.com/photo-1571019613454-1cb2f99b2d8b?auto=format&fit=crop&w=900&q=80",
        "youtube_url": "https://www.youtube.com/results?search_query=pelvic+tilt+exercise+physiotherapy",
    },
    "straight-leg raise": {
        "image_url": "https://images.unsplash.com/photo-1599058917212-d750089bc07e?auto=format&fit=crop&w=900&q=80",
        "youtube_url": "https://www.youtube.com/results?search_query=straight+leg+raise+physiotherapy",
    },
    "heel slides": {
        "image_url": "https://images.unsplash.com/photo-1576678927484-cc907957088c?auto=format&fit=crop&w=900&q=80",
        "youtube_url": "https://www.youtube.com/results?search_query=heel+slides+exercise+physiotherapy",
    },
    "grip strengthening": {
        "image_url": "https://images.unsplash.com/photo-1581009137042-c552e485697a?auto=format&fit=crop&w=900&q=80",
        "youtube_url": "https://www.youtube.com/results?search_query=grip+strengthening+physiotherapy",
    },
    "mini squats": {
        "image_url": "https://images.unsplash.com/photo-1540206395-68808572332f?auto=format&fit=crop&w=900&q=80",
        "youtube_url": "https://www.youtube.com/results?search_query=mini+squat+exercise+physiotherapy",
    },
    "step-ups": {
        "image_url": "https://images.unsplash.com/photo-1486218119243-13883505764c?auto=format&fit=crop&w=900&q=80",
        "youtube_url": "https://www.youtube.com/results?search_query=step+up+exercise+physiotherapy",
    },
    "single-leg balance": {
        "image_url": "https://images.unsplash.com/photo-1530549387789-4c1017266635?auto=format&fit=crop&w=900&q=80",
        "youtube_url": "https://www.youtube.com/results?search_query=single+leg+balance+proprioception+physiotherapy",
    },
    "balance and proprioception": {
        "image_url": "https://images.unsplash.com/photo-1530549387789-4c1017266635?auto=format&fit=crop&w=900&q=80",
        "youtube_url": "https://www.youtube.com/results?search_query=balance+proprioception+rehabilitation+physiotherapy",
    },
    "wrist flexion and extension rom": {
        "image_url": "https://images.unsplash.com/photo-1602192509154-0b900ee1f851?auto=format&fit=crop&w=900&q=80",
        "youtube_url": "https://www.youtube.com/results?search_query=wrist+flexion+extension+physiotherapy",
    },
    "cat-camel stretch": {
        "image_url": "https://images.unsplash.com/photo-1516841273335-e39b37888115?auto=format&fit=crop&w=900&q=80",
        "youtube_url": "https://www.youtube.com/results?search_query=cat+camel+stretch+spine+physiotherapy",
    },
    "bird-dog": {
        "image_url": "https://images.unsplash.com/photo-1571019613454-1cb2f99b2d8b?auto=format&fit=crop&w=900&q=80",
        "youtube_url": "https://www.youtube.com/results?search_query=bird+dog+exercise+core+physiotherapy",
    },
    "standing calf raises": {
        "image_url": "https://images.unsplash.com/photo-1576678927484-cc907957088c?auto=format&fit=crop&w=900&q=80",
        "youtube_url": "https://www.youtube.com/results?search_query=calf+raise+standing+exercise+physiotherapy",
    },
    "seated calf raises": {
        "image_url": "https://images.unsplash.com/photo-1518611012118-696072aa579a?auto=format&fit=crop&w=900&q=80",
        "youtube_url": "https://www.youtube.com/results?search_query=seated+calf+raise+exercise+physiotherapy",
    },
    "bridging": {
        "image_url": "https://images.unsplash.com/photo-1599058917212-d750089bc07e?auto=format&fit=crop&w=900&q=80",
        "youtube_url": "https://www.youtube.com/results?search_query=glute+bridge+exercise+physiotherapy",
    },
    # IMP FIX: Previously missing entries — these fell back to a generic Unsplash image
    "knee-to-chest stretch": {
        "image_url": "https://images.unsplash.com/photo-1571019613454-1cb2f99b2d8b?auto=format&fit=crop&w=900&q=80",
        "youtube_url": "https://www.youtube.com/results?search_query=knee+to+chest+stretch+physiotherapy",
    },
    "walking progression": {
        "image_url": "https://images.unsplash.com/photo-1476480862126-209bfaa8edc8?auto=format&fit=crop&w=900&q=80",
        "youtube_url": "https://www.youtube.com/results?search_query=walking+rehabilitation+physiotherapy+progression",
    },
    "ankle alphabet": {
        "image_url": "https://images.unsplash.com/photo-1518611012118-696072aa579a?auto=format&fit=crop&w=900&q=80",
        "youtube_url": "https://www.youtube.com/results?search_query=ankle+alphabet+exercise+physiotherapy",
    },
    "towel calf stretch": {
        "image_url": "https://images.unsplash.com/photo-1518611012118-696072aa579a?auto=format&fit=crop&w=900&q=80",
        "youtube_url": "https://www.youtube.com/results?search_query=towel+calf+stretch+physiotherapy",
    },
    "resistance band ankle dorsiflexion": {
        "image_url": "https://images.unsplash.com/photo-1518611012118-696072aa579a?auto=format&fit=crop&w=900&q=80",
        "youtube_url": "https://www.youtube.com/results?search_query=resistance+band+ankle+dorsiflexion+physiotherapy",
    },
    "heel-to-toe walking": {
        "image_url": "https://images.unsplash.com/photo-1476480862126-209bfaa8edc8?auto=format&fit=crop&w=900&q=80",
        "youtube_url": "https://www.youtube.com/results?search_query=heel+to+toe+walking+balance+physiotherapy",
    },
    "toe flexion and extension": {
        "image_url": "https://images.unsplash.com/photo-1518611012118-696072aa579a?auto=format&fit=crop&w=900&q=80",
        "youtube_url": "https://www.youtube.com/results?search_query=toe+flexion+extension+exercise+physiotherapy",
    },
    "elevation protocol": {
        "image_url": "https://images.unsplash.com/photo-1559757148-5c350d0d3c56?auto=format&fit=crop&w=900&q=80",
        "youtube_url": "https://www.youtube.com/results?search_query=limb+elevation+injury+recovery",
    },
    "wrist curls and extensions": {
        "image_url": "https://images.unsplash.com/photo-1602192509154-0b900ee1f851?auto=format&fit=crop&w=900&q=80",
        "youtube_url": "https://www.youtube.com/results?search_query=wrist+curl+extension+strengthening+physiotherapy",
    },
}


# ---------------------------------------------------------------------------
# Red-flag detection
# ---------------------------------------------------------------------------

RED_FLAG_CONDITIONS = [
    (
        lambda ctx: (ctx.get("pain_level") or 0) >= 8,
        "Pain level is 8 or above — seek immediate clinical review before any exercises.",
    ),
    (
        lambda ctx: (ctx.get("swelling_level") or "") == "severe",
        "Severe swelling present — requires clinical assessment before rehabilitation.",
    ),
    (
        lambda ctx: (ctx.get("treatment_status") or "") == "not_evaluated"
        and (ctx.get("doctor_restrictions") or "") == "not_sure",
        "Injury not yet clinically evaluated and restrictions unknown — seek assessment first.",
    ),
]


def detect_red_flags(patient_context: dict) -> list[str]:
    """Return list of red-flag warning strings applicable to this patient context."""
    flags = []
    for condition_fn, message in RED_FLAG_CONDITIONS:
        try:
            if condition_fn(patient_context):
                flags.append(message)
        except Exception:
            pass
    return flags


# ---------------------------------------------------------------------------
# Guidance assembly
# ---------------------------------------------------------------------------


def _get_body_part_guidance(body_part: str, recovery_stage: str) -> dict:
    """Return the appropriate guidance dict for body part and stage."""
    bp_key = (body_part or "").lower().strip()
    stage_key = (recovery_stage or "").lower().strip()

    part = BODY_PART_GUIDANCE.get(bp_key)
    if not part:
        return {
            "dos": ["Follow your clinician's instructions."],
            "avoids": ["Any exercise not cleared by your doctor."],
            "focus": "Clinical assessment required for personalised guidance.",
            "timeline": ["Consult your clinician for a personalised recovery timeline."],
        }

    # Check if there's an all-stages fallback (e.g., skull)
    all_stages = part.get("all_stages")
    if all_stages:
        return all_stages

    stage_data = part.get(stage_key)
    if stage_data:
        return stage_data

    # IMP FIX: Fallback order corrected — most restrictive stage first (acute → early → late)
    # Previously: early_recovery was tried first, which is too permissive as a fallback.
    for fallback in ["acute_phase", "early_recovery", "late_recovery"]:
        if part.get(fallback):
            return part[fallback]

    return {
        "dos": ["Follow your clinician's guidance."],
        "avoids": ["Any activity not cleared by your doctor."],
        "focus": "Clinician assessment required.",
        "timeline": [],
    }


def _build_treatment_dos(treatment_status: str, doctor_restrictions: str | None = None) -> list[str]:
    """Return treatment-specific and restriction-specific additional dos."""
    treatment_key = (treatment_status or "").lower().strip()
    restriction_key = (doctor_restrictions or "").lower().strip()
    result = list(TREATMENT_OVERLAY.get(treatment_key, []))
    # GAP FIX: Append restriction-specific advisories (previously movement_restricted had no overlay)
    restriction_dos = RESTRICTION_OVERLAY.get(restriction_key, [])
    for item in restriction_dos:
        if item not in result:
            result.append(item)
    return result


def _get_nutrition(recovery_stage: str, sex: str | None = None) -> dict:
    """Return stage-appropriate nutrition plan."""
    stage_key = (recovery_stage or "").lower().strip()
    plan = dict(STAGE_NUTRITION.get(stage_key, STAGE_NUTRITION["acute_phase"]))

    # Sex-specific meal adjustment
    meals = [dict(m) for m in plan.get("meals", [])]
    if (sex or "").lower() == "male" and meals:
        # Swap last meal to a higher-protein option
        meals[-1]["name"] = "Chicken quinoa bowl"
        meals[-1]["image_url"] = FOOD_IMAGES["Chicken quinoa bowl"]
        meals[-1]["items"] = ["Grilled chicken (200g)", "Quinoa", "Stir-fried vegetables", "Curd or buttermilk"]
    plan["meals"] = meals
    return plan


# ---------------------------------------------------------------------------
# Media enrichment
# ---------------------------------------------------------------------------


def _media_for_exercise(name: str) -> dict:
    key = str(name or "").lower().strip()
    media = EXERCISE_MEDIA.get(key)
    if media:
        return dict(media)
    query = quote_plus(f"{name} physiotherapy exercise proper form")
    return {
        "image_url": "",
        "youtube_url": f"https://www.youtube.com/results?search_query={query}",
    }


def enrich_exercise_media(recommendations: dict) -> dict:
    """Enrich legacy recommendation exercises with media URLs."""
    enriched = dict(recommendations or {})
    exercises = []
    for exercise in enriched.get("exercises") or []:
        item = dict(exercise)
        item["media"] = _media_for_exercise(item.get("name", "exercise"))
        exercises.append(item)
    enriched["exercises"] = exercises
    return enriched


def enrich_plan_exercise_media(exercise_plan: dict) -> dict:
    """Enrich new exercise plan with media URLs."""
    enriched = dict(exercise_plan or {})
    exercises = []
    for exercise in enriched.get("exercises") or []:
        item = dict(exercise)
        item["media"] = _media_for_exercise(item.get("name", "exercise"))
        exercises.append(item)
    enriched["exercises"] = exercises
    return enriched


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

SYSTEM_INSTRUCTIONS = """
You enhance AI Physio recovery guidance that has already been generated by a deterministic rule engine.
Rules:
- Do NOT change exercise eligibility decisions — they are final.
- Do NOT add conditions, medications, diagnoses, or body parts not in the input.
- Rephrase 'daily_focus' points in clear, encouraging, patient-friendly language.
- Add a 1-sentence personalized intro based on body_part, recovery_stage, and treatment_status.
- Enhance the nutrition summary with 1-2 practical sentences for the specific stage.
- Return concise JSON with keys: personalized_intro, daily_focus (list), nutrition_note.
""".strip()

ASSISTANT_SYSTEM_INSTRUCTIONS = """
You are the AI Physio recovery assistant.
Rules:
- You are educational only and must not diagnose, prescribe medication, or replace a clinician.
- If symptoms include severe pain, deformity, numbness, open wound, fever, inability to bear weight, swelling after trauma, or possible fracture/dislocation, advise urgent clinical review.
- Use the provided context: body_part, treatment_status, recovery_stage, pain_level, mobility_status, exercise_eligible.
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
    """Robustly parse a JSON object from LLM text that may include markdown fences."""
    # BUG FIX: Previous stripping logic failed for fenced code blocks with newlines
    # e.g. ```json\n{...}\n``` was not fully cleaned.
    cleaned = text.strip()
    # Strip markdown code fences (handles ```json, ```JSON, ``` variants)
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()
    # Extract the first complete JSON object
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        cleaned = match.group(0)
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


def _compact_analysis(analysis: dict) -> dict:
    """Build a compact context dict for the assistant endpoint."""
    pred = analysis.get("prediction") or {}
    ctx = analysis.get("patient_context") or {}
    guidance = analysis.get("recovery_guidance") or {}
    physio = guidance.get("physio_support") or {}

    return {
        "condition": pred.get("condition"),
        "confidence": pred.get("confidence"),
        "body_part": ctx.get("body_part"),
        "treatment_status": ctx.get("treatment_status"),
        "recovery_stage": ctx.get("recovery_stage"),
        "pain_level": ctx.get("pain_level"),
        "mobility_status": ctx.get("mobility_status"),
        "exercise_eligible": (guidance.get("exercise_eligibility") or {}).get("eligible"),
        "exercise_ineligible_reason": (guidance.get("exercise_eligibility") or {}).get("reason"),
        "phase": physio.get("phase"),
        "daily_focus": physio.get("daily_focus"),
    }


# ---------------------------------------------------------------------------
# Main guidance builder
# ---------------------------------------------------------------------------


def build_recovery_guidance(
    prediction: dict,
    patient_context: dict,
    exercise_plan: dict | None = None,
) -> dict:
    """
    Build the complete recovery guidance payload.

    Primary source: deterministic rule engine + BODY_PART_GUIDANCE tables.
    Optional enhancement: Groq LLM for personalised phrasing when API key is set.

    Args:
        prediction:      Output from predict_condition()
        patient_context: Cleaned PatientContext dict (8 clinical fields)
        exercise_plan:   Output from get_exercise_plan() (already computed)

    Returns:
        Full guidance dict with exercise_eligibility, physio_support, nutrition_plan.
    """
    ctx = patient_context or {}
    bp = ctx.get("body_part") or "affected area"
    stage = ctx.get("recovery_stage") or "acute_phase"
    treatment = ctx.get("treatment_status") or "not_evaluated"
    sex = ctx.get("sex")

    # ---- 1. Exercise eligibility ----
    eligibility = compute_exercise_eligibility(ctx)

    # ---- 2. Red flags ----
    red_flags = detect_red_flags(ctx)

    # ---- 3. Body-part-specific guidance ----
    body_guidance = _get_body_part_guidance(bp, stage)

    # ---- 4. Treatment & restriction overlay dos ----
    doctor_restrictions = ctx.get("doctor_restrictions")
    treatment_dos = _build_treatment_dos(treatment, doctor_restrictions)

    # ---- 5. Assemble daily focus ----
    all_dos = list(body_guidance.get("dos", []))
    if treatment_dos:
        all_dos = treatment_dos + all_dos  # treatment notes first

    daily_focus = all_dos[:6]  # cap at 6 items for UI

    # ---- 6. Phase label ----
    stage_label = stage.replace("_", " ").title()
    bp_label = bp.replace("_", " ").title()
    phase = f"{bp_label} — {stage_label}"

    # ---- 7. Summary text ----
    focus_text = body_guidance.get("focus", "Follow clinician guidance for your recovery.")
    is_normal = (prediction or {}).get("condition") == "normal"
    
    if not eligibility["eligible"]:
        summary = (
            f"{focus_text} "
            f"Exercise plan is not generated: {eligibility['reason']}"
        )
    else:
        summary = (
            f"{focus_text} "
            f"Exercise plan has been generated for {bp_label} — {stage_label}."
        )

    if is_normal:
        summary = (
            "No fracture detected on X-ray, but your symptoms suggest a soft-tissue injury "
            "(such as a sprain or strain) or minor occult fracture. "
        ) + summary

    # ---- 8. Nutrition ----
    nutrition = _get_nutrition(stage, sex)

    # ---- 9. Risk level & when-to-seek-help ----
    risk = compute_recovery_risk_level(ctx)

    # ---- 10. Build base payload ----
    payload = {
        "enabled": True,
        "generated_by": "rule_engine",
        "exercise_eligibility": eligibility,
        "red_flags": red_flags,
        "recovery_risk_level": risk,
        "when_to_seek_help": WHEN_TO_SEEK_HELP,
        "physio_support": {
            "phase": phase,
            "risk_level": "needs_clinician_review" if red_flags else "structured_recovery",
            "summary": summary,
            "daily_focus": daily_focus,
            "avoids": body_guidance.get("avoids", []),
            "timeline": body_guidance.get("timeline", []),
            "exercise_media_note": (
                "Use videos only for form education. Stop if pain rises, and follow clinician advice first."
            ),
        },
        "nutrition_plan": nutrition,
    }

    # ---- 10. Optional LLM enhancement ----
    provider = _provider_config()
    if not provider:
        payload["status"] = "rule_engine_only: GROQ_API_KEY not configured"
        return payload

    try:
        from openai import OpenAI
    except ImportError:
        payload["status"] = "rule_engine_only: openai package not installed"
        return payload

    llm_payload = {
        "body_part": bp_label,
        "recovery_stage": stage_label,
        "treatment_status": treatment.replace("_", " ").title(),
        "pain_level": ctx.get("pain_level"),
        "exercise_eligible": eligibility["eligible"],
        "ineligible_reason": eligibility.get("reason"),
        "daily_focus": daily_focus,
        "nutrition_summary": nutrition.get("summary"),
    }

    try:
        client = OpenAI(api_key=provider["api_key"], base_url=provider["base_url"])
        response = client.chat.completions.create(
            model=provider["model"],
            messages=[
                {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                {"role": "user", "content": f"Return JSON only.\n{json.dumps(llm_payload, ensure_ascii=True)}"},
            ],
            temperature=0.2,
        )
        parsed = _parse_json_text(response.choices[0].message.content or "{}")
        payload["generated_by"] = "groq_enhanced"
        payload["provider"] = provider["provider"]
        payload["model"] = provider["model"]
        payload["status"] = "generated"

        if isinstance(parsed.get("personalized_intro"), str):
            payload["physio_support"]["personalized_intro"] = parsed["personalized_intro"].strip()
        # GAP FIX: LLM must NOT replace daily_focus — it contains safety-critical instructions
        # (e.g. "Avoid hip flexion > 90°") that the LLM could hallucinate away.
        # The LLM is only permitted to add a personalized_intro above.
        if isinstance(parsed.get("nutrition_note"), str):
            payload["nutrition_plan"]["llm_note"] = parsed["nutrition_note"].strip()
    except Exception as exc:
        payload["status"] = _safe_error_status(provider["provider"], exc)

    return payload


# ---------------------------------------------------------------------------
# Recovery Q&A assistant
# ---------------------------------------------------------------------------


def answer_recovery_question(
    question: str,
    analysis: dict | None = None,
    patient_context: dict | None = None,
) -> dict:
    """Answer a user's recovery question using Groq when configured."""
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
            "I can help with general recovery guidance. Keep activity gentle and pain-free, "
            "follow the report's red-flag guidance, and always defer to your clinician's advice."
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

    compact_context = _compact_analysis(analysis or {})
    if patient_context:
        compact_context["patient_context"] = patient_context
    elif analysis and analysis.get("patient_context"):
        compact_context["patient_context"] = analysis["patient_context"]

    payload = {"question": cleaned_question, "context": compact_context}

    try:
        client = OpenAI(api_key=provider["api_key"], base_url=provider["base_url"])
        response = client.chat.completions.create(
            model=provider["model"],
            messages=[
                {"role": "system", "content": ASSISTANT_SYSTEM_INSTRUCTIONS},
                {
                    "role": "user",
                    "content": (
                        "Answer this question for the AI Physio app. Return JSON only with keys: "
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
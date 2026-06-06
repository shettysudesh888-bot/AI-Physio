"""
Rule-based physiotherapy exercise recommendation engine for AI Physio.

The recommendations are deliberately conservative. High-risk findings return
medical escalation guidance and only low-risk activity suggestions.
"""

EXERCISE_MAP: dict[str, dict] = {
    "normal": {
        "condition_label": "Normal",
        "summary": "No abnormality detected. Preventive exercises are recommended to maintain bone density and joint health.",
        "severity": "none",
        "exercises": [
            {
                "name": "Weight-bearing walking",
                "description": "Walk at a moderate pace on flat ground with supportive footwear.",
                "sets": None,
                "reps": None,
                "duration_sec": 1800,
                "frequency": "5x per week",
                "precautions": "Stay hydrated and avoid uneven terrain.",
            },
            {
                "name": "Calf raises",
                "description": "Stand upright, slowly raise the heels, hold for 2 seconds, then lower.",
                "sets": 3,
                "reps": 15,
                "duration_sec": None,
                "frequency": "3x per week",
                "precautions": "Hold a wall or chair for balance if needed.",
            },
            {
                "name": "Hip abduction",
                "description": "Lie on your side, lift the top leg to about 45 degrees, then lower slowly.",
                "sets": 3,
                "reps": 12,
                "duration_sec": None,
                "frequency": "3x per week",
                "precautions": "Keep movements slow and controlled.",
            },
        ],
        "dietary_tips": [
            "Maintain adequate calcium intake, typically 1000-1200 mg/day.",
            "Ensure sufficient vitamin D through safe sunlight exposure, diet, or supplements.",
        ],
        "when_to_see_doctor": "If you have unexplained pain, swelling, deformity, numbness, or reduced range of motion.",
    },
    "fracture": {
        "condition_label": "Possible Fracture",
        "summary": "Possible fracture identified. Do not begin rehabilitation exercises until a clinician confirms the injury and clears movement.",
        "severity": "high",
        "exercises": [
            {
                "name": "Deep breathing",
                "description": "Inhale slowly for 4 seconds, hold for 2 seconds, then exhale for 4 seconds.",
                "sets": None,
                "reps": None,
                "duration_sec": 300,
                "frequency": "3x per day",
                "precautions": "Safe while resting if comfortable. Stop if dizziness occurs.",
            },
            {
                "name": "Unaffected-limb gentle range of motion",
                "description": "Move joints away from the suspected fracture through a comfortable range.",
                "sets": 2,
                "reps": 10,
                "duration_sec": None,
                "frequency": "2x per day",
                "precautions": "Do not move, load, massage, or stretch the suspected fracture area.",
            },
        ],
        "dietary_tips": [
            "Prioritize protein-rich foods to support tissue repair.",
            "Follow clinician advice for calcium and vitamin D supplementation.",
            "Avoid smoking and excessive alcohol because both can impair bone healing.",
        ],
        "when_to_see_doctor": "Immediately. A possible fracture needs urgent clinical evaluation and imaging confirmation.",
    },
    "uncertain": {
        "condition_label": "Uncertain Screening Result",
        "summary": "The model result or image quality is uncertain. Avoid starting new exercises until a qualified clinician reviews the image and symptoms.",
        "severity": "review",
        "exercises": [
            {
                "name": "Rest and protect the area",
                "description": "Avoid loading, stretching, or exercising the painful area until the result is clinically reviewed.",
                "sets": None,
                "reps": None,
                "duration_sec": None,
                "frequency": "Until reviewed",
                "precautions": "Seek urgent care for deformity, numbness, severe pain, open wound, or inability to bear weight.",
            }
        ],
        "dietary_tips": [
            "Stay hydrated and maintain balanced meals while waiting for clinical guidance.",
            "Do not use this screening result to delay care if symptoms are severe.",
        ],
        "when_to_see_doctor": "As soon as possible, especially if pain, swelling, trauma, numbness, or movement restriction is present.",
    },
    "osteoporosis": {
        "condition_label": "Osteoporosis Signs",
        "summary": "Low bone density signs detected. Exercises emphasize balance, light resistance, and controlled weight-bearing activity.",
        "severity": "moderate",
        "exercises": [
            {
                "name": "Tai Chi",
                "description": "Perform low-impact balance and coordination movements.",
                "sets": None,
                "reps": None,
                "duration_sec": 1800,
                "frequency": "3x per week",
                "precautions": "Start supervised if balance is poor. Avoid deep forward bends.",
            },
            {
                "name": "Wall push-ups",
                "description": "Stand arm's length from a wall, bend the elbows to bring the chest toward the wall, then push back.",
                "sets": 3,
                "reps": 12,
                "duration_sec": None,
                "frequency": "3x per week",
                "precautions": "Keep the movement smooth and avoid elbow locking.",
            },
            {
                "name": "Sit-to-stand",
                "description": "Rise from a chair using controlled movement, then sit down slowly.",
                "sets": 2,
                "reps": 10,
                "duration_sec": None,
                "frequency": "3x per week",
                "precautions": "Use arm support if needed and avoid rushing.",
            },
        ],
        "dietary_tips": [
            "Discuss daily calcium and vitamin D targets with a clinician.",
            "Reduce fall risk by reviewing medications, vision, footwear, and home hazards.",
        ],
        "when_to_see_doctor": "Within 1-2 weeks for bone density assessment and treatment planning.",
    },
    "joint_dislocation": {
        "condition_label": "Possible Joint Dislocation",
        "summary": "Joint displacement signs detected. The joint must be assessed and relocated by a medical professional before exercise.",
        "severity": "high",
        "exercises": [
            {
                "name": "Finger and toe mobility",
                "description": "Gently open and close fingers or wiggle toes away from the affected joint.",
                "sets": 2,
                "reps": 10,
                "duration_sec": None,
                "frequency": "2x per day",
                "precautions": "Avoid moving the suspected dislocated joint.",
            },
            {
                "name": "Breathing and relaxation",
                "description": "Use slow diaphragmatic breathing while waiting for medical care.",
                "sets": None,
                "reps": None,
                "duration_sec": 300,
                "frequency": "As needed",
                "precautions": "Do not attempt to relocate the joint yourself.",
            },
        ],
        "dietary_tips": [
            "Stay hydrated.",
            "Follow clinician guidance for anti-inflammatory nutrition if swelling persists.",
        ],
        "when_to_see_doctor": "Immediately. Joint dislocations require professional reduction and neurovascular assessment.",
    },
    "bone_tumor": {
        "condition_label": "Bone Abnormality / Possible Tumor",
        "summary": "Abnormal bone growth signs detected. Exercise must be restricted until specialist review.",
        "severity": "critical",
        "exercises": [
            {
                "name": "Breathing and relaxation",
                "description": "Perform gentle diaphragmatic breathing to reduce stress and maintain comfort.",
                "sets": None,
                "reps": None,
                "duration_sec": 600,
                "frequency": "2x per day",
                "precautions": "Avoid impact, loading, or stretching around the affected bone.",
            }
        ],
        "dietary_tips": [
            "Follow oncology or physician diet guidance.",
            "Maintain protein intake and hydration unless restricted by a clinician.",
        ],
        "when_to_see_doctor": "Urgently. This finding needs specialist medical review and confirmation.",
    },
    "arthritis": {
        "condition_label": "Arthritis Signs",
        "summary": "Joint degeneration signs detected. Exercises focus on mobility, stiffness reduction, and low-impact strengthening.",
        "severity": "moderate",
        "exercises": [
            {
                "name": "Water aerobics or hydrotherapy",
                "description": "Perform low-impact movement in warm water to reduce joint stress.",
                "sets": None,
                "reps": None,
                "duration_sec": 1800,
                "frequency": "3x per week",
                "precautions": "Avoid overexertion and very hot water.",
            },
            {
                "name": "Chair yoga",
                "description": "Use seated stretches for hips, spine, shoulders, and arms.",
                "sets": None,
                "reps": None,
                "duration_sec": 1200,
                "frequency": "4x per week",
                "precautions": "Avoid positions that cause sharp pain or joint locking.",
            },
            {
                "name": "Grip strengthening",
                "description": "Squeeze a soft foam ball for 5 seconds, then release.",
                "sets": 3,
                "reps": 10,
                "duration_sec": None,
                "frequency": "2x per day",
                "precautions": "Use lower resistance if finger joints are inflamed.",
            },
        ],
        "dietary_tips": [
            "Omega-3 rich foods may help support inflammation control.",
            "Maintaining a healthy weight can reduce joint load.",
        ],
        "when_to_see_doctor": "Within 1-2 weeks for clinical assessment if pain, swelling, or stiffness persists.",
    },
}

_UNKNOWN_RECOMMENDATION = {
    "condition_label": "Unknown Condition",
    "summary": "The detected condition could not be matched to a specific exercise protocol.",
    "severity": "unknown",
    "exercises": [],
    "dietary_tips": ["Maintain a balanced diet with calcium, vitamin D, protein, and hydration."],
    "when_to_see_doctor": "As soon as possible for proper clinical diagnosis.",
}


def _format_duration(seconds: int | None) -> str | None:
    if not seconds:
        return None

    minutes, remaining_seconds = divmod(seconds, 60)
    if minutes and remaining_seconds:
        return f"{minutes}m {remaining_seconds:02d}s"
    if minutes:
        return f"{minutes} min"
    return f"{remaining_seconds} sec"


def get_recommendations(condition: str) -> dict:
    """
    Return exercise recommendations for a detected condition key.
    """
    condition_key = (condition or "").lower().strip()
    rec = EXERCISE_MAP.get(condition_key, _UNKNOWN_RECOMMENDATION)

    exercises = []
    for exercise in rec.get("exercises", []):
        enriched = dict(exercise)
        enriched["duration_human"] = _format_duration(enriched.get("duration_sec"))
        exercises.append(enriched)

    result = dict(rec)
    result["condition_key"] = condition_key or "unknown"
    result["exercises"] = exercises
    result["medical_disclaimer"] = (
        "This educational tool does not replace professional medical diagnosis, "
        "treatment, or emergency care."
    )
    return result


def list_conditions() -> list[str]:
    """Return all supported condition keys."""
    return list(EXERCISE_MAP.keys())

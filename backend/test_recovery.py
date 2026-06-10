import sys
from pathlib import Path

# Setup paths so imports work properly
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from recovery_guidance import compute_exercise_eligibility, detect_red_flags
from recommendation import get_exercise_plan

def test_scenario(name, context, expected_eligibility, expected_reason_substring=None):
    print(f"Running test: {name}...")
    
    # 1. Test eligibility
    elig = compute_exercise_eligibility(context)
    assert elig["eligible"] == expected_eligibility, \
        f"Expected eligibility {expected_eligibility}, got {elig['eligible']}. Context: {context}"
    
    if expected_reason_substring:
        reason = elig["reason"] or ""
        assert expected_reason_substring.lower() in reason.lower(), \
            f"Expected reason containing '{expected_reason_substring}', got '{reason}'"
            
    # 2. Test get_exercise_plan behavior based on eligibility
    plan = get_exercise_plan(
        body_part=context.get("body_part", ""),
        recovery_stage=context.get("recovery_stage", ""),
        treatment_status=context.get("treatment_status", ""),
        exercise_eligible=elig["eligible"]
    )
    
    if expected_eligibility:
        assert len(plan.get("exercises", [])) > 0, "Eligible patient should have exercises generated."
    else:
        assert len(plan.get("exercises", [])) == 0, "Ineligible patient should have 0 exercises."
        
    print(f"  Passed! (Eligible={elig['eligible']}, Reason={elig['reason']})")

def test_red_flags(name, context, expected_has_flags, expected_flag_substring=None):
    print(f"Running test: {name}...")
    flags = detect_red_flags(context)
    
    has_flags = len(flags) > 0
    assert has_flags == expected_has_flags, \
        f"Expected red flags={expected_has_flags}, got {flags}"
        
    if expected_flag_substring:
        found = any(expected_flag_substring.lower() in f.lower() for f in flags)
        assert found, f"Expected red flag containing '{expected_flag_substring}', got {flags}"
        
    print(f"  Passed! (Flags={flags})")

def run_all_tests():
    print("======================================================================")
    # Scenario 1: Skull + any stage (e.g. late_recovery) -> blocked
    test_scenario(
        "Skull + Late Recovery + Approved",
        {
            "body_part": "skull",
            "recovery_stage": "late_recovery",
            "exercise_approval": "yes",
            "doctor_restrictions": "no_restrictions",
            "mobility_status": "normal",
            "treatment_status": "physiotherapy_started"
        },
        expected_eligibility=False,
        expected_reason_substring="skull"
    )

    # Scenario 2: Spine + Acute Phase -> blocked
    test_scenario(
        "Spine + Acute Phase + Approved",
        {
            "body_part": "spine",
            "recovery_stage": "acute_phase",
            "exercise_approval": "yes",
            "doctor_restrictions": "no_restrictions",
            "mobility_status": "normal",
            "treatment_status": "brace_support"
        },
        expected_eligibility=False,
        expected_reason_substring="spinal"
    )

    # Scenario 3: Exercise Approval = No -> blocked
    test_scenario(
        "Wrist + Late Recovery + Approval = No",
        {
            "body_part": "wrist_hand",
            "recovery_stage": "late_recovery",
            "exercise_approval": "no",
            "doctor_restrictions": "no_restrictions",
            "mobility_status": "normal",
            "treatment_status": "physiotherapy_started"
        },
        expected_eligibility=False,
        expected_reason_substring="doctor has not advised"
    )

    # Scenario 4: Exercise Restricted by doctor -> blocked
    test_scenario(
        "Knee + Early Recovery + Restrictions = exercise_restricted",
        {
            "body_part": "knee",
            "recovery_stage": "early_recovery",
            "exercise_approval": "yes",
            "doctor_restrictions": "exercise_restricted",
            "mobility_status": "normal",
            "treatment_status": "brace_support"
        },
        expected_eligibility=False,
        expected_reason_substring="restricted by your doctor"
    )

    # Scenario 5: Severely Limited mobility -> blocked
    test_scenario(
        "Shoulder + Early Recovery + Mobility = severely_limited",
        {
            "body_part": "shoulder",
            "recovery_stage": "early_recovery",
            "exercise_approval": "yes",
            "doctor_restrictions": "no_restrictions",
            "mobility_status": "severely_limited",
            "treatment_status": "physiotherapy_started"
        },
        expected_eligibility=False,
        expected_reason_substring="mobility prevents safe exercise"
    )

    # Scenario 6: Wrist + Early Recovery + Cast + Exercise Approval Yes -> eligible
    test_scenario(
        "Wrist + Early Recovery + Cast + Approved",
        {
            "body_part": "wrist_hand",
            "recovery_stage": "early_recovery",
            "exercise_approval": "yes",
            "doctor_restrictions": "no_restrictions",
            "mobility_status": "slightly_limited",
            "treatment_status": "cast_plaster"
        },
        expected_eligibility=True
    )

    # Scenario 7: Femur + Late Recovery + Physio Started + Yes -> eligible
    test_scenario(
        "Femur + Late Recovery + Physio Started + Approved",
        {
            "body_part": "femur",
            "recovery_stage": "late_recovery",
            "exercise_approval": "yes",
            "doctor_restrictions": "no_restrictions",
            "mobility_status": "normal",
            "treatment_status": "physiotherapy_started"
        },
        expected_eligibility=True
    )

    # Scenario 8: Ankle + Acute + Surgery + No -> blocked (approval = No)
    test_scenario(
        "Ankle + Acute + Surgery + Approval = No",
        {
            "body_part": "ankle_foot",
            "recovery_stage": "acute_phase",
            "exercise_approval": "no",
            "doctor_restrictions": "no_restrictions",
            "mobility_status": "moderately_limited",
            "treatment_status": "surgery_performed"
        },
        expected_eligibility=False,
        expected_reason_substring="doctor has not advised"
    )

    # Red Flag Scenario 1: Pain >= 8
    test_red_flags(
        "High Pain Level (Pain = 9)",
        {
            "pain_level": 9,
            "swelling_level": "none",
            "treatment_status": "physiotherapy_started"
        },
        expected_has_flags=True,
        expected_flag_substring="severe pain"
    )

    # Red Flag Scenario 2: Severe swelling
    test_red_flags(
        "Severe Swelling",
        {
            "pain_level": 4,
            "swelling_level": "severe",
            "treatment_status": "physiotherapy_started"
        },
        expected_has_flags=True,
        expected_flag_substring="severe swelling"
    )

    # Red Flag Scenario 3: Not evaluated
    test_red_flags(
        "Not Evaluated Clinically",
        {
            "pain_level": 4,
            "swelling_level": "none",
            "treatment_status": "not_evaluated"
        },
        expected_has_flags=True,
        expected_flag_substring="clinical evaluation"
    )

    print("======================================================================")
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("======================================================================")

if __name__ == "__main__":
    run_all_tests()

"""
Rule-based physiotherapy exercise recommendation engine for AI Physio.

Exercises are generated based on:
  - Body part (10 anatomical regions)
  - Recovery stage (acute / early / late)
  - Treatment status (cast, brace, surgery, etc.)
  - Exercise eligibility (determined by recovery_guidance engine)

The old EXERCISE_MAP is retained for backward-compatible fallback on
legacy saved analyses that only carry a condition key.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Body-part × stage × treatment exercise library
# ---------------------------------------------------------------------------

# fmt: off
BODY_PART_EXERCISE_MAP: dict[str, dict[str, dict]] = {

    # -----------------------------------------------------------------------
    "skull": {
        # Rule 1: NEVER generate exercises for skull — handled by eligibility engine.
        # This entry exists so the code path is defined; it will never be reached
        # because compute_exercise_eligibility() blocks skull before this is called.
        "acute_phase":    {"exercises": [], "treatment_notes": {}},
        "early_recovery": {"exercises": [], "treatment_notes": {}},
        "late_recovery":  {"exercises": [], "treatment_notes": {}},
    },

    # -----------------------------------------------------------------------
    "spine": {
        "acute_phase": {
            # Rule 2: blocked by eligibility engine — empty fallback
            "exercises": [],
            "treatment_notes": {},
        },
        "early_recovery": {
            "exercises": [
                {
                    "name": "Pelvic Tilt (Supine)",
                    "description": "Lie on your back with knees bent. Gently flatten your lower back against the floor by tightening your abdominals. Hold 5 seconds, release.",
                    "sets": 3, "reps": 10, "duration_sec": None,
                    "frequency": "2x per day",
                    "precautions": "No pain allowed. Stop if you feel shooting pain into the legs.",
                },
                {
                    "name": "Knee-to-Chest Stretch",
                    "description": "Lie on your back. Slowly draw one knee toward your chest, hold 20 seconds, lower. Repeat other side.",
                    "sets": 2, "reps": 5, "duration_sec": None,
                    "frequency": "2x per day",
                    "precautions": "Only perform if cleared by your clinician. Do not force the stretch.",
                },
                {
                    "name": "Diaphragmatic Breathing",
                    "description": "Inhale slowly for 4 seconds expanding the belly, hold 2 seconds, exhale for 4 seconds. Helps manage pain and promote healing.",
                    "sets": None, "reps": None, "duration_sec": 300,
                    "frequency": "3x per day",
                    "precautions": "Safe in all positions. Stop if dizziness occurs.",
                },
            ],
            "treatment_notes": {
                "surgery_performed": "All exercises should be supervised. Do not progress without surgeon sign-off.",
                "cast_plaster": "Perform only upper-limb and breathing exercises if torso is casted.",
                "brace_support": "Keep brace on during all exercises unless instructed otherwise.",
            },
        },
        "late_recovery": {
            "exercises": [
                {
                    "name": "Cat-Camel Stretch",
                    "description": "On hands and knees, arch your back upward (cat), then let it sag downward (camel). Move slowly and smoothly.",
                    "sets": 2, "reps": 10, "duration_sec": None,
                    "frequency": "Daily",
                    "precautions": "Stop if pain radiates to arms or legs.",
                },
                {
                    "name": "Bird-Dog",
                    "description": "On hands and knees, extend one arm and the opposite leg simultaneously. Hold 3 seconds, return, repeat other side.",
                    "sets": 3, "reps": 8, "duration_sec": None,
                    "frequency": "3x per week",
                    "precautions": "Keep the spine neutral — do not allow rotation. Stop at any point of increased pain.",
                },
                {
                    "name": "Walking Progression",
                    "description": "Begin with 10-minute flat-ground walks. Increase by 5 minutes per week if tolerated. Use supportive footwear.",
                    "sets": None, "reps": None, "duration_sec": 600,
                    "frequency": "5x per week",
                    "precautions": "Avoid uneven surfaces. Stop if back pain or leg symptoms worsen.",
                },
            ],
            "treatment_notes": {
                "surgery_performed": "Progress only after surgeon or physio clearance for mobilisation.",
                "physiotherapy_started": "Coordinate exercises with your physiotherapist's program.",
            },
        },
    },

    # -----------------------------------------------------------------------
    "shoulder": {
        "acute_phase": {
            "exercises": [
                {
                    "name": "Pendulum Circles",
                    "description": "Lean forward supported on a table. Let the injured arm hang freely. Make small clockwise then anticlockwise circles using gentle momentum — not muscle force.",
                    "sets": 2, "reps": 10, "duration_sec": None,
                    "frequency": "2x per day",
                    "precautions": "Use gravity only — do not actively lift the arm. Stop if sharp pain occurs.",
                },
                {
                    "name": "Elbow/Wrist ROM (Away from Shoulder)",
                    "description": "While keeping the shoulder still, gently bend and straighten the elbow and wiggle the wrist to prevent stiffness in unaffected joints.",
                    "sets": 2, "reps": 10, "duration_sec": None,
                    "frequency": "3x per day",
                    "precautions": "Do not move the shoulder.",
                },
                {
                    "name": "Diaphragmatic Breathing",
                    "description": "Slow belly breathing to manage pain and support healing.",
                    "sets": None, "reps": None, "duration_sec": 300,
                    "frequency": "3x per day",
                    "precautions": "Stop if dizziness occurs.",
                },
            ],
            "treatment_notes": {
                "surgery_performed": "Pendulum circles only if explicitly cleared post-operatively. Keep sling on otherwise.",
                "cast_plaster": "Perform only exercises distal to the cast.",
                "brace_support": "Keep brace on during exercises.",
            },
        },
        "early_recovery": {
            "exercises": [
                {
                    "name": "Active-Assisted Shoulder Flexion",
                    "description": "Use a cane or the other hand to assist the injured arm in raising forward to 90 degrees or pain-free range. Lower slowly.",
                    "sets": 3, "reps": 10, "duration_sec": None,
                    "frequency": "2x per day",
                    "precautions": "Keep movement in pain-free range only. No forcing.",
                },
                {
                    "name": "Wall Slides",
                    "description": "Stand sideways to a wall. Place the back of the hand on the wall and slide the arm upward as far as comfortable. Hold 2 seconds, slide down.",
                    "sets": 3, "reps": 10, "duration_sec": None,
                    "frequency": "2x per day",
                    "precautions": "Stop before pain. Do not shrug the shoulder.",
                },
                {
                    "name": "Isometric Shoulder External Rotation",
                    "description": "Stand with elbow bent 90 degrees at your side. Press the back of your hand against a doorframe for 5 seconds without moving. Relax.",
                    "sets": 3, "reps": 8, "duration_sec": None,
                    "frequency": "Daily",
                    "precautions": "No motion — only gentle pressure. Stop if pain increases.",
                },
            ],
            "treatment_notes": {
                "surgery_performed": "Follow post-operative rehabilitation protocol exactly. Do not exceed prescribed range.",
                "brace_support": "Remove brace only as instructed. Reapply immediately after exercises.",
            },
        },
        "late_recovery": {
            "exercises": [
                {
                    "name": "Resistance Band External Rotation",
                    "description": "Anchor a resistance band at elbow height. Keeping elbow at 90 degrees at your side, rotate forearm outward against the band. Hold 2 seconds, return.",
                    "sets": 3, "reps": 12, "duration_sec": None,
                    "frequency": "3x per week",
                    "precautions": "Use light resistance initially. No pain beyond 3/10.",
                },
                {
                    "name": "Shoulder Press (Seated)",
                    "description": "Sit tall. Press light dumbbells (or water bottles) from shoulder height overhead. Lower slowly.",
                    "sets": 3, "reps": 10, "duration_sec": None,
                    "frequency": "3x per week",
                    "precautions": "Only with clinician clearance. Stop if impingement pain occurs.",
                },
                {
                    "name": "Horizontal Abduction (Side-Lying)",
                    "description": "Lie on your uninjured side. Raise the top arm toward the ceiling. Hold 2 seconds, lower slowly.",
                    "sets": 3, "reps": 12, "duration_sec": None,
                    "frequency": "3x per week",
                    "precautions": "Use a light weight only when pain-free range is established.",
                },
            ],
            "treatment_notes": {
                "physiotherapy_started": "Coordinate with your physio — do not double-up on resistance exercises without guidance.",
                "surgery_performed": "Only progress to resistance if cleared at post-operative review.",
            },
        },
    },

    # -----------------------------------------------------------------------
    "elbow": {
        "acute_phase": {
            "exercises": [
                {
                    "name": "Hand and Wrist ROM",
                    "description": "Gently open and close the fingers, make a fist, then spread the fingers wide. Flex and extend the wrist slowly.",
                    "sets": 3, "reps": 10, "duration_sec": None,
                    "frequency": "3x per day",
                    "precautions": "Keep the elbow still. Do not rotate the forearm at this stage.",
                },
                {
                    "name": "Elevation and Icing Protocol",
                    "description": "Keep the elbow elevated above heart level for 20-minute intervals to reduce swelling. Apply ice pack wrapped in a cloth for 15 minutes.",
                    "sets": None, "reps": None, "duration_sec": None,
                    "frequency": "Every 2-3 hours",
                    "precautions": "Never apply ice directly to skin. Remove if skin becomes numb.",
                },
            ],
            "treatment_notes": {
                "surgery_performed": "No active exercises until post-op clearance. Elevation only.",
                "cast_plaster": "Move fingers only. Keep cast dry.",
                "brace_support": "Keep brace on during exercises.",
            },
        },
        "early_recovery": {
            "exercises": [
                {
                    "name": "Elbow Flexion and Extension",
                    "description": "Slowly bend the elbow as far as comfortable, hold 2 seconds, then straighten fully. Use the other hand to gently assist if needed.",
                    "sets": 3, "reps": 10, "duration_sec": None,
                    "frequency": "2x per day",
                    "precautions": "Stop at point of pain. Do not force the last degrees of range.",
                },
                {
                    "name": "Forearm Pronation and Supination",
                    "description": "Hold a pen or light stick. Rotate the forearm so the palm faces up, then down. Keep the elbow at your side.",
                    "sets": 3, "reps": 10, "duration_sec": None,
                    "frequency": "2x per day",
                    "precautions": "Only perform if cleared. Stop if there is catching or clicking pain.",
                },
            ],
            "treatment_notes": {
                "cast_plaster": "Only wrist and finger exercises while cast is on.",
                "surgery_performed": "Follow post-op protocol strictly. Do not force range of motion.",
            },
        },
        "late_recovery": {
            "exercises": [
                {
                    "name": "Resistance Band Elbow Curl",
                    "description": "Stand on one end of a resistance band. Curl the forearm toward the shoulder against the band resistance. Lower slowly.",
                    "sets": 3, "reps": 12, "duration_sec": None,
                    "frequency": "3x per week",
                    "precautions": "Start with very light resistance. No pain beyond mild muscle fatigue.",
                },
                {
                    "name": "Grip Strengthening",
                    "description": "Squeeze a soft foam ball or stress ball firmly for 5 seconds, then release fully.",
                    "sets": 3, "reps": 10, "duration_sec": None,
                    "frequency": "Daily",
                    "precautions": "Stop if elbow pain increases. Use lower resistance ball if needed.",
                },
                {
                    "name": "Wrist Curl and Extension",
                    "description": "Rest forearm on a table. Hold a light weight. Curl the wrist upward, then lower. Flip and extend the wrist upward.",
                    "sets": 3, "reps": 12, "duration_sec": None,
                    "frequency": "3x per week",
                    "precautions": "Only very light weight. Stop if elbow discomfort occurs.",
                },
            ],
            "treatment_notes": {
                "physiotherapy_started": "Work within your physio's prescribed load and range.",
            },
        },
    },

    # -----------------------------------------------------------------------
    "wrist_hand": {
        "acute_phase": {
            "exercises": [
                {
                    "name": "Finger Flexion and Extension",
                    "description": "Gently bend and straighten all fingers through comfortable range. Then make a loose fist and open wide.",
                    "sets": 3, "reps": 10, "duration_sec": None,
                    "frequency": "3x per day",
                    "precautions": "Do not move the wrist at this stage if it is the injured area.",
                },
                {
                    "name": "Elevation",
                    "description": "Keep the hand elevated above heart level as much as possible, especially when resting. Use a pillow or sling.",
                    "sets": None, "reps": None, "duration_sec": None,
                    "frequency": "Continuously when resting",
                    "precautions": "Do not keep the arm hanging down — this increases swelling.",
                },
            ],
            "treatment_notes": {
                "cast_plaster": "Move fingers only — do not attempt to move the wrist against the cast.",
                "surgery_performed": "No active wrist movement. Finger ROM only if cleared.",
                "brace_support": "Perform finger exercises with brace on.",
            },
        },
        "early_recovery": {
            "exercises": [
                {
                    "name": "Wrist Flexion and Extension ROM",
                    "description": "Place forearm on a table, wrist over the edge. Gently move the hand up (extension) and down (flexion) through comfortable range.",
                    "sets": 3, "reps": 10, "duration_sec": None,
                    "frequency": "2x per day",
                    "precautions": "Pain-free range only. Do not force at end range.",
                },
                {
                    "name": "Wrist Radial and Ulnar Deviation",
                    "description": "Rest forearm on a table, palm down. Move the hand sideways — toward the thumb, then toward the little finger.",
                    "sets": 3, "reps": 10, "duration_sec": None,
                    "frequency": "2x per day",
                    "precautions": "Move within comfortable range. Stop if clicking or pain worsens.",
                },
                {
                    "name": "Grip Putty Exercise",
                    "description": "Squeeze and knead therapeutic putty (or a soft ball) to restore hand strength and coordination.",
                    "sets": 3, "reps": 15, "duration_sec": None,
                    "frequency": "Daily",
                    "precautions": "Use soft putty initially. Increase resistance only when pain-free.",
                },
            ],
            "treatment_notes": {
                "cast_plaster": "Finger ROM only while cast is on. Wrist exercises begin after cast removal.",
                "surgery_performed": "Follow post-op protocol. Passive ROM first, then active.",
                "brace_support": "Remove brace for exercises only as instructed. Reapply after.",
            },
        },
        "late_recovery": {
            "exercises": [
                {
                    "name": "Grip Strengthening",
                    "description": "Squeeze a stress ball or grip exerciser firmly for 5 seconds, release. Progress to higher resistance as strength improves.",
                    "sets": 3, "reps": 15, "duration_sec": None,
                    "frequency": "Daily",
                    "precautions": "No sharp wrist pain. Muscle fatigue is normal.",
                },
                {
                    "name": "Wrist Curl (Light Weight)",
                    "description": "Sit with forearm on a table, palm facing up. Hold a light weight. Curl the wrist upward, hold 2 seconds, lower slowly.",
                    "sets": 3, "reps": 12, "duration_sec": None,
                    "frequency": "3x per week",
                    "precautions": "Start with 0.5 kg or lighter. Stop if wrist pain exceeds 3/10.",
                },
                {
                    "name": "Wrist Extension Strengthening",
                    "description": "Forearm on table, palm facing down. Hold light weight. Lift the wrist upward (extension), hold 2 seconds, lower slowly.",
                    "sets": 3, "reps": 12, "duration_sec": None,
                    "frequency": "3x per week",
                    "precautions": "Same precautions as wrist curl. Alternate days with flexion exercise.",
                },
                {
                    "name": "Putty Pinch and Roll",
                    "description": "Pinch a ball of putty between thumb and each finger separately. Roll and flatten putty to improve fine motor control.",
                    "sets": 2, "reps": 10, "duration_sec": None,
                    "frequency": "Daily",
                    "precautions": "Avoid if finger joints are acutely swollen.",
                },
            ],
            "treatment_notes": {
                "physiotherapy_started": "Progress resistance only in coordination with your physio.",
                "surgery_performed": "Strengthening only after bone union confirmed on imaging.",
            },
        },
    },

    # -----------------------------------------------------------------------
    "pelvis_hip": {
        "acute_phase": {
            "exercises": [
                {
                    "name": "Ankle Pumps",
                    "description": "While lying or sitting, pump the feet up and down (dorsiflexion and plantarflexion) to maintain circulation and reduce clot risk.",
                    "sets": None, "reps": 20, "duration_sec": None,
                    "frequency": "Every hour while awake",
                    "precautions": "Safe to perform unless ankle is also injured.",
                },
                {
                    "name": "Gluteal Sets (Static)",
                    "description": "Lie on your back. Gently squeeze the buttock muscles together for 5 seconds, then relax.",
                    "sets": 3, "reps": 10, "duration_sec": None,
                    "frequency": "3x per day",
                    "precautions": "No movement of the hip — static contraction only.",
                },
                {
                    "name": "Diaphragmatic Breathing",
                    "description": "Slow belly breathing to manage pain and maintain lung function during bed rest.",
                    "sets": None, "reps": None, "duration_sec": 300,
                    "frequency": "3x per day",
                    "precautions": "Stop if dizziness occurs.",
                },
            ],
            "treatment_notes": {
                "surgery_performed": "Ankle pumps only. All other exercises require post-op clearance.",
                "cast_plaster": "Ankle pumps and upper body exercises only.",
                "brace_support": "Perform all exercises with brace on unless instructed.",
                "weight_bearing_restricted": "No weight-bearing. Exercises in lying only.",
            },
        },
        "early_recovery": {
            "exercises": [
                {
                    "name": "Hip Abduction (Side-Lying)",
                    "description": "Lie on your uninjured side. Lift the top leg to ~30 degrees, hold 2 seconds, lower slowly.",
                    "sets": 3, "reps": 10, "duration_sec": None,
                    "frequency": "2x per day",
                    "precautions": "Keep hips stacked. Do not rotate the leg. Stop if groin pain increases.",
                },
                {
                    "name": "Supine Knee-to-Chest",
                    "description": "Lie on your back. Slowly draw one knee toward your chest using your hands, hold 20 seconds, lower.",
                    "sets": 2, "reps": 5, "duration_sec": None,
                    "frequency": "2x per day",
                    "precautions": "Do not force. Stop if hip or groin pain exceeds 3/10.",
                },
                {
                    "name": "Sit-to-Stand",
                    "description": "Using arm rests or a chair with good support, practice slowly rising to stand and returning to sit in a controlled manner.",
                    "sets": 2, "reps": 8, "duration_sec": None,
                    "frequency": "2x per day",
                    "precautions": "Only if weight-bearing is cleared. Use arm support. Do not use pivot or twist movement.",
                },
            ],
            "treatment_notes": {
                "surgery_performed": "Weight-bearing and hip ROM exercises only as cleared by surgeon.",
                "brace_support": "Check brace is correctly fitted before standing exercises.",
                "weight_bearing_restricted": "Avoid sit-to-stand. Hip abduction and knee-to-chest only.",
            },
        },
        "late_recovery": {
            "exercises": [
                {
                    "name": "Bridging",
                    "description": "Lie on your back, knees bent, feet flat. Lift hips off the bed/floor until body is straight from knees to shoulders. Hold 3 seconds, lower slowly.",
                    "sets": 3, "reps": 12, "duration_sec": None,
                    "frequency": "3x per week",
                    "precautions": "Do not hyperextend the lower back. Stop if hip pain occurs.",
                },
                {
                    "name": "Clamshells",
                    "description": "Lie on your side with hips and knees bent. Keeping feet together, lift the top knee like a clamshell opening. Hold 2 seconds, lower.",
                    "sets": 3, "reps": 12, "duration_sec": None,
                    "frequency": "3x per week",
                    "precautions": "Do not rock the pelvis. Keep core gently engaged.",
                },
                {
                    "name": "Walking Progression",
                    "description": "Progress from short assisted walks to independent walking. Target 20-30 minutes on flat ground.",
                    "sets": None, "reps": None, "duration_sec": 1200,
                    "frequency": "5x per week",
                    "precautions": "Wear supportive footwear. Stop if increased pain or limp worsens.",
                },
            ],
            "treatment_notes": {
                "surgery_performed": "Only advance to full weight-bearing exercises after imaging confirms bone union.",
                "physiotherapy_started": "Do not add exercises outside your physio's program without checking.",
            },
        },
    },

    # -----------------------------------------------------------------------
    "femur": {
        "acute_phase": {
            "exercises": [
                {
                    "name": "Ankle Pumps",
                    "description": "Pump feet up and down (dorsiflexion / plantarflexion) continuously to maintain circulation.",
                    "sets": None, "reps": 20, "duration_sec": None,
                    "frequency": "Every hour while awake",
                    "precautions": "Critical for DVT prevention after femur injury.",
                },
                {
                    "name": "Quad Sets",
                    "description": "Lie with leg straight. Tighten the thigh muscle (push the back of the knee into the bed). Hold 5 seconds, relax.",
                    "sets": 3, "reps": 10, "duration_sec": None,
                    "frequency": "3x per day",
                    "precautions": "Only if approved. No active leg raise at this stage.",
                },
                {
                    "name": "Upper Body Active Exercise",
                    "description": "Gentle upper body movements (shoulder circles, hand squeezes) to maintain circulation and prevent deconditioning.",
                    "sets": 2, "reps": 10, "duration_sec": None,
                    "frequency": "2x per day",
                    "precautions": "Do not twist or shift body weight onto injured leg.",
                },
            ],
            "treatment_notes": {
                "surgery_performed": "Post-operative protocol governs all movement. Quad sets only if explicitly cleared.",
                "cast_plaster": "Ankle pumps and upper body only. Do not move within cast.",
                "weight_bearing_restricted": "No weight-bearing. All exercises in lying position.",
            },
        },
        "early_recovery": {
            "exercises": [
                {
                    "name": "Straight-Leg Raise",
                    "description": "Lie on your back. Tighten the thigh, then lift the straight leg to about 45 degrees. Hold 2 seconds, lower slowly.",
                    "sets": 3, "reps": 10, "duration_sec": None,
                    "frequency": "2x per day",
                    "precautions": "Only if cleared. Stop if hip flexor pain is significant.",
                },
                {
                    "name": "Assisted Walking with Support",
                    "description": "Walk with crutches or walking frame as prescribed. Follow the weight-bearing protocol from your clinician exactly.",
                    "sets": None, "reps": None, "duration_sec": 600,
                    "frequency": "As prescribed",
                    "precautions": "Do not exceed prescribed weight-bearing status. Consult your clinician on progression.",
                },
                {
                    "name": "Heel Slides",
                    "description": "Lie on your back. Slowly slide the heel toward the buttocks by bending the knee, then slide back out.",
                    "sets": 3, "reps": 10, "duration_sec": None,
                    "frequency": "2x per day",
                    "precautions": "Stop if thigh or knee pain exceeds 3/10.",
                },
            ],
            "treatment_notes": {
                "surgery_performed": "Weight-bearing and exercise progression dictated by surgical fixation type. Follow surgeon's protocol.",
                "brace_support": "Check brace alignment before walking exercises.",
            },
        },
        "late_recovery": {
            "exercises": [
                {
                    "name": "Full Weight-Bearing Walking",
                    "description": "Progress to walking without assistive devices. Target 30 minutes on flat ground. Increase by 5 minutes per week.",
                    "sets": None, "reps": None, "duration_sec": 1800,
                    "frequency": "5x per week",
                    "precautions": "Stop if limp worsens or thigh pain increases. Return to assistive device if needed.",
                },
                {
                    "name": "Step-Ups",
                    "description": "Step up onto a low step (10 cm) with the injured leg leading. Step down with the uninjured leg first. Use handrail for safety.",
                    "sets": 3, "reps": 10, "duration_sec": None,
                    "frequency": "3x per week",
                    "precautions": "Only with full weight-bearing clearance. Start with a very low step.",
                },
                {
                    "name": "Mini Squats",
                    "description": "Stand with feet shoulder-width apart. Bend both knees to about 30-45 degrees (mini squat), then rise. Hold onto a support if needed.",
                    "sets": 3, "reps": 12, "duration_sec": None,
                    "frequency": "3x per week",
                    "precautions": "Do not let knees cave inward. Stop if thigh pain exceeds 3/10.",
                },
            ],
            "treatment_notes": {
                "surgery_performed": "Only after bone union confirmed. Running and impact activities require specialist clearance.",
                "physiotherapy_started": "Synchronise progression with your physiotherapist.",
            },
        },
    },

    # -----------------------------------------------------------------------
    "knee": {
        "acute_phase": {
            "exercises": [
                {
                    "name": "Quad Sets",
                    "description": "Sit or lie with knee straight. Tighten the quadriceps (front thigh) and push the back of the knee down. Hold 5 seconds, relax.",
                    "sets": 3, "reps": 10, "duration_sec": None,
                    "frequency": "3x per day",
                    "precautions": "No swinging or active knee movement. Static contraction only.",
                },
                {
                    "name": "Straight-Leg Raise",
                    "description": "Lie on your back. Tighten the quads, then raise the straight leg to 45 degrees. Hold 2 seconds, lower slowly.",
                    "sets": 3, "reps": 10, "duration_sec": None,
                    "frequency": "2x per day",
                    "precautions": "Keep the knee locked straight during the raise. Stop if knee pain increases.",
                },
                {
                    "name": "Ankle Pumps and Circles",
                    "description": "Move the ankle up and down and draw circles to maintain calf circulation and reduce swelling risk.",
                    "sets": None, "reps": 20, "duration_sec": None,
                    "frequency": "Every 1-2 hours",
                    "precautions": "Safe and encouraged. Keep knee still.",
                },
            ],
            "treatment_notes": {
                "surgery_performed": "Post-operative protocol governs all exercises. Quad sets only if cleared.",
                "brace_support": "Keep brace locked in extension during straight-leg raises.",
                "cast_plaster": "Ankle pumps and quad sets only while cast is on.",
            },
        },
        "early_recovery": {
            "exercises": [
                {
                    "name": "Heel Slides",
                    "description": "Lie on your back. Slowly slide the heel toward the buttocks bending the knee, then slide back out to straight.",
                    "sets": 3, "reps": 10, "duration_sec": None,
                    "frequency": "2x per day",
                    "precautions": "Move only through pain-free range. Do not force flexion.",
                },
                {
                    "name": "Seated Knee Bends",
                    "description": "Sit in a chair. Slowly bend and straighten the knee as far as comfortable. Use gravity and the other foot to gently assist.",
                    "sets": 3, "reps": 10, "duration_sec": None,
                    "frequency": "2x per day",
                    "precautions": "Do not apply heavy force. Stop at pain.",
                },
                {
                    "name": "Hip Abduction (Side-Lying)",
                    "description": "Lie on your uninjured side. Lift the top leg to 30 degrees with knee straight. Hold 2 seconds, lower.",
                    "sets": 3, "reps": 12, "duration_sec": None,
                    "frequency": "2x per day",
                    "precautions": "Keeps hip and thigh strength during knee recovery.",
                },
            ],
            "treatment_notes": {
                "surgery_performed": "ROM progression must follow post-operative protocol milestones.",
                "brace_support": "Remove brace for exercises only as instructed.",
            },
        },
        "late_recovery": {
            "exercises": [
                {
                    "name": "Mini Squats",
                    "description": "Stand with feet shoulder-width. Bend knees to 30-45 degrees holding a support. Rise slowly.",
                    "sets": 3, "reps": 12, "duration_sec": None,
                    "frequency": "3x per week",
                    "precautions": "Keep knees tracking over toes. Stop if knee pain exceeds 3/10.",
                },
                {
                    "name": "Step-Ups",
                    "description": "Step up onto a low step with the injured leg. Step down with the other leg leading. Use a handrail.",
                    "sets": 3, "reps": 10, "duration_sec": None,
                    "frequency": "3x per week",
                    "precautions": "Start with a very low step (10 cm). Increase height gradually.",
                },
                {
                    "name": "Terminal Knee Extension",
                    "description": "Loop a resistance band behind the knee. Stand and bend the knee slightly against the band, then straighten fully. Squeeze quads at the top.",
                    "sets": 3, "reps": 15, "duration_sec": None,
                    "frequency": "3x per week",
                    "precautions": "Light resistance only. Stop if knee aches during or after.",
                },
            ],
            "treatment_notes": {
                "physiotherapy_started": "Do not progress beyond your physio's prescribed stage.",
                "surgery_performed": "Strengthening exercises only after wound healing and surgeon clearance.",
            },
        },
    },

    # -----------------------------------------------------------------------
    "lower_leg": {
        "acute_phase": {
            "exercises": [
                {
                    "name": "Ankle Pumps",
                    "description": "Pump the foot up and down briskly to activate the calf muscle pump and reduce swelling.",
                    "sets": None, "reps": 20, "duration_sec": None,
                    "frequency": "Every hour while awake",
                    "precautions": "Critical. Do not skip. Keep leg elevated between sessions.",
                },
                {
                    "name": "Toe Wiggling",
                    "description": "Spread and curl the toes repeatedly to maintain circulation in the foot.",
                    "sets": None, "reps": 20, "duration_sec": None,
                    "frequency": "Frequently throughout the day",
                    "precautions": "Very gentle. Stop if toes are numb or painful.",
                },
                {
                    "name": "Elevation",
                    "description": "Keep the lower leg elevated above heart level as much as possible to reduce swelling and pain.",
                    "sets": None, "reps": None, "duration_sec": None,
                    "frequency": "Continuously when resting",
                    "precautions": "Do not hang the leg below heart level for extended periods.",
                },
            ],
            "treatment_notes": {
                "cast_plaster": "Ankle pumps are possible inside cast. Toe wiggling encouraged.",
                "surgery_performed": "No movement beyond ankle pumps and toe wiggles until post-op clearance.",
                "weight_bearing_restricted": "Non-weight-bearing. Exercises in lying/sitting only.",
            },
        },
        "early_recovery": {
            "exercises": [
                {
                    "name": "Seated Calf Raises",
                    "description": "Sit in a chair with feet flat. Raise heels as high as comfortable, hold 2 seconds, lower slowly.",
                    "sets": 3, "reps": 15, "duration_sec": None,
                    "frequency": "2x per day",
                    "precautions": "Pain-free only. Mild tightness is acceptable.",
                },
                {
                    "name": "Ankle Circles",
                    "description": "Slowly rotate the foot in large circles, clockwise then anticlockwise.",
                    "sets": 2, "reps": 10, "duration_sec": None,
                    "frequency": "3x per day",
                    "precautions": "Move within comfortable range. Stop if sharp pain occurs.",
                },
                {
                    "name": "Knee Bends (Seated)",
                    "description": "Sit and gently bend and straighten the knee to maintain leg mobility without loading the lower leg.",
                    "sets": 3, "reps": 10, "duration_sec": None,
                    "frequency": "2x per day",
                    "precautions": "No weight on the foot during this exercise.",
                },
            ],
            "treatment_notes": {
                "cast_plaster": "Only ankle pumps and toe exercises while cast is on.",
                "surgery_performed": "Follow post-op protocol. Weight-bearing only when cleared.",
                "brace_support": "Perform exercises with brace on unless instructed otherwise.",
            },
        },
        "late_recovery": {
            "exercises": [
                {
                    "name": "Standing Calf Raises",
                    "description": "Stand holding a support. Rise onto the balls of both feet, hold 2 seconds, lower slowly. Progress to single-leg when able.",
                    "sets": 3, "reps": 15, "duration_sec": None,
                    "frequency": "3x per week",
                    "precautions": "Only with full weight-bearing clearance. Use support until balance is safe.",
                },
                {
                    "name": "Balance and Proprioception",
                    "description": "Stand on the injured leg for 30 seconds with eyes open. Progress to eyes closed, then on a soft surface.",
                    "sets": 3, "reps": None, "duration_sec": 30,
                    "frequency": "Daily",
                    "precautions": "Stand near a wall for safety. Stop if leg buckles or pain occurs.",
                },
                {
                    "name": "Walking Progression",
                    "description": "Progress from short walks with a support device to independent walking on flat ground, targeting 20-30 minutes.",
                    "sets": None, "reps": None, "duration_sec": 1200,
                    "frequency": "5x per week",
                    "precautions": "Stop and rest if lower leg pain increases. Avoid uneven surfaces initially.",
                },
            ],
            "treatment_notes": {
                "surgery_performed": "Return to full weight-bearing activities only after bone union is confirmed.",
                "physiotherapy_started": "Balance exercises are most effective under physio supervision initially.",
            },
        },
    },

    # -----------------------------------------------------------------------
    "ankle_foot": {
        "acute_phase": {
            "exercises": [
                {
                    "name": "Ankle Pumps",
                    "description": "Pump the foot up (dorsiflexion) and down (plantarflexion) continuously. This is the most important acute exercise.",
                    "sets": None, "reps": 30, "duration_sec": None,
                    "frequency": "Every 30-60 minutes while awake",
                    "precautions": "Perform in lying or sitting with leg elevated.",
                },
                {
                    "name": "Toe Flexion and Extension",
                    "description": "Curl the toes tightly downward for 3 seconds, then spread and extend them upward.",
                    "sets": 3, "reps": 10, "duration_sec": None,
                    "frequency": "3x per day",
                    "precautions": "Stop if toe or forefoot pain worsens.",
                },
                {
                    "name": "Elevation Protocol",
                    "description": "Keep the foot elevated above heart level using pillows. Elevation is the most effective acute intervention for ankle swelling.",
                    "sets": None, "reps": None, "duration_sec": None,
                    "frequency": "Continuously when resting",
                    "precautions": "Avoid hanging the foot below heart level for extended periods.",
                },
            ],
            "treatment_notes": {
                "cast_plaster": "Ankle pumps inside cast. Toe wiggles. Elevation critical.",
                "surgery_performed": "No active ankle movement. Ankle pumps only if cleared post-operatively.",
                "weight_bearing_restricted": "Strictly non-weight-bearing. All exercises in lying or sitting.",
            },
        },
        "early_recovery": {
            "exercises": [
                {
                    "name": "Ankle Alphabet",
                    "description": "Write the alphabet with your big toe by moving the ankle in all directions. Move slowly and with full letters.",
                    "sets": 1, "reps": None, "duration_sec": None,
                    "frequency": "2x per day",
                    "precautions": "Seated with foot elevated. Stop if sharp pain occurs at any letter.",
                },
                {
                    "name": "Seated Calf Raises",
                    "description": "Sit with foot flat on the floor. Raise the heel as high as possible. Hold 2 seconds, lower slowly.",
                    "sets": 3, "reps": 15, "duration_sec": None,
                    "frequency": "2x per day",
                    "precautions": "Only with partial weight-bearing clearance. Pain-free only.",
                },
                {
                    "name": "Towel Calf Stretch",
                    "description": "Sit with leg outstretched. Loop a towel around the ball of the foot. Gently pull the towel toward you to stretch the calf. Hold 30 seconds.",
                    "sets": 3, "reps": None, "duration_sec": 30,
                    "frequency": "2x per day",
                    "precautions": "Gentle pull only. No bouncing. Stop if ankle pain exceeds 3/10.",
                },
            ],
            "treatment_notes": {
                "cast_plaster": "All exercises after cast removal. Ankle pumps only while cast is on.",
                "surgery_performed": "Weight-bearing and ROM exercises only at surgeon-approved milestones.",
                "brace_support": "Remove brace only for ROM exercises as instructed. Reapply for walking.",
            },
        },
        "late_recovery": {
            "exercises": [
                {
                    "name": "Standing Calf Raises",
                    "description": "Stand holding a support. Rise onto the balls of both feet. Hold 2 seconds, lower slowly. Progress to single-leg.",
                    "sets": 3, "reps": 20, "duration_sec": None,
                    "frequency": "Daily",
                    "precautions": "Use support until balance is fully restored. Progress to single-leg over 2-3 weeks.",
                },
                {
                    "name": "Single-Leg Balance",
                    "description": "Stand on the injured ankle for 30 seconds, eyes open. Progress to eyes closed, then on a foam mat.",
                    "sets": 3, "reps": None, "duration_sec": 30,
                    "frequency": "Daily",
                    "precautions": "Stand near a wall. Stop if ankle wobbles excessively or pain increases.",
                },
                {
                    "name": "Heel-to-Toe Walking",
                    "description": "Walk in a straight line placing the heel of one foot directly in front of the toes of the other. Focus on balance and control.",
                    "sets": None, "reps": None, "duration_sec": 120,
                    "frequency": "Daily",
                    "precautions": "Walk near a wall or counter. Use normal footwear with good ankle support.",
                },
                {
                    "name": "Resistance Band Ankle Dorsiflexion",
                    "description": "Sit with leg outstretched. Loop a resistance band around the foot. Pull the foot toward you against the band. Hold 2 seconds, release.",
                    "sets": 3, "reps": 15, "duration_sec": None,
                    "frequency": "3x per week",
                    "precautions": "Light resistance only. Stop if ankle discomfort occurs.",
                },
            ],
            "treatment_notes": {
                "surgery_performed": "Return to sport activities only after specialist clearance.",
                "physiotherapy_started": "Balance and proprioception training is most effective supervised.",
            },
        },
    },
}
# fmt: on


# ---------------------------------------------------------------------------
# Legacy EXERCISE_MAP (kept for backward compatibility with old saved cases)
# ---------------------------------------------------------------------------

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
}

_UNKNOWN_RECOMMENDATION = {
    "condition_label": "Unknown Condition",
    "summary": "The detected condition could not be matched to a specific exercise protocol.",
    "severity": "unknown",
    "exercises": [],
    "dietary_tips": ["Maintain a balanced diet with calcium, vitamin D, protein, and hydration."],
    "when_to_see_doctor": "As soon as possible for proper clinical diagnosis.",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_duration(seconds: int | None) -> str | None:
    if not seconds:
        return None
    minutes, remaining_seconds = divmod(seconds, 60)
    if minutes and remaining_seconds:
        return f"{minutes}m {remaining_seconds:02d}s"
    if minutes:
        return f"{minutes} min"
    return f"{remaining_seconds} sec"


def _enrich(exercises: list[dict]) -> list[dict]:
    """Add duration_human and medical_disclaimer to each exercise."""
    result = []
    for ex in exercises:
        item = dict(ex)
        item["duration_human"] = _format_duration(item.get("duration_sec"))
        result.append(item)
    return result


# ---------------------------------------------------------------------------
# New body-part-aware exercise plan
# ---------------------------------------------------------------------------


def get_exercise_plan(
    body_part: str,
    recovery_stage: str,
    treatment_status: str,
    exercise_eligible: bool,
) -> dict:
    """
    Return a body-part, stage, and treatment-aware exercise plan.

    When exercise_eligible is False, returns an empty plan with a reason string.
    """
    if not exercise_eligible:
        return {
            "exercises": [],
            "exercise_eligible": False,
            "stage_label": recovery_stage.replace("_", " ").title(),
            "body_part_label": body_part.replace("_", " ").title(),
            "medical_disclaimer": (
                "This educational tool does not replace professional medical diagnosis, "
                "treatment, or emergency care."
            ),
        }

    part_key = (body_part or "").lower().strip()
    stage_key = (recovery_stage or "").lower().strip()
    treatment_key = (treatment_status or "not_evaluated").lower().strip()

    part_data = BODY_PART_EXERCISE_MAP.get(part_key)
    if not part_data:
        return {
            "exercises": [],
            "exercise_eligible": True,
            "note": f"No exercise protocol found for body part: {body_part}. Consult a physiotherapist.",
            "medical_disclaimer": (
                "This educational tool does not replace professional medical diagnosis, "
                "treatment, or emergency care."
            ),
        }

    stage_data = part_data.get(stage_key, {})
    exercises = list(stage_data.get("exercises", []))
    treatment_notes = stage_data.get("treatment_notes", {})

    # BUG FIX: Previous logic had a broken or-fallback that always returned None
    # for any treatment type (the second .get() always resolved to .get("") = None).
    # Now: simply look up by treatment_key directly.
    treatment_note = treatment_notes.get(treatment_key)

    return {
        "exercises": _enrich(exercises),
        "exercise_eligible": True,
        "body_part": part_key,
        "recovery_stage": stage_key,
        "treatment_status": treatment_key,
        "body_part_label": part_key.replace("_", " ").title(),
        "stage_label": stage_key.replace("_", " ").title(),
        "treatment_note": treatment_note or None,
        "medical_disclaimer": (
            "This educational tool does not replace professional medical diagnosis, "
            "treatment, or emergency care."
        ),
    }


# ---------------------------------------------------------------------------
# Legacy API (backward compatibility)
# ---------------------------------------------------------------------------


def get_recommendations(condition: str) -> dict:
    """
    Return legacy exercise recommendations for a condition key.
    Used for backward-compatible rendering of old saved analyses.
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

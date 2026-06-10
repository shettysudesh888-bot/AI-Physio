from pathlib import Path

from fastapi import Body, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field
from datetime import datetime, timezone
import hashlib
import tempfile
import shutil
import secrets
import sqlite3
import uuid
import json

try:
    from .llm_explanation import explain_result
    from .predict import predict_condition
    from .recovery_guidance import (
        answer_recovery_question,
        build_recovery_guidance,
        compute_exercise_eligibility,
        compute_recovery_risk_level,
        detect_red_flags,
        enrich_exercise_media,
        enrich_plan_exercise_media,
    )
    from .report import build_analysis_report_pdf, build_nutrition_report_pdf
    from .recommendation import EXERCISE_MAP, get_recommendations, get_exercise_plan
except ImportError:
    from llm_explanation import explain_result
    from predict import predict_condition
    from recovery_guidance import (
        answer_recovery_question,
        build_recovery_guidance,
        compute_exercise_eligibility,
        compute_recovery_risk_level,
        detect_red_flags,
        enrich_exercise_media,
        enrich_plan_exercise_media,
    )
    from report import build_analysis_report_pdf, build_nutrition_report_pdf
    from recommendation import EXERCISE_MAP, get_recommendations, get_exercise_plan

app = FastAPI(
    title="AI Physio API",
    description="Bone X-ray analysis and physiotherapy recommendation system",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
UPLOAD_DIR = Path(tempfile.gettempdir()) / "ai_physio_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = PROJECT_ROOT / "ai_physio.db"

app.mount("/frontend", StaticFiles(directory=str(PROJECT_ROOT / "frontend")), name="frontend")

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

# Valid values for enum-like fields
VALID_BODY_PARTS = {
    "skull", "spine", "shoulder", "elbow", "wrist_hand",
    "pelvis_hip", "femur", "knee", "lower_leg", "ankle_foot",
}
VALID_TREATMENT_STATUS = {
    "not_evaluated", "cast_plaster", "brace_support",
    "surgery_performed", "physiotherapy_started",
}
VALID_RECOVERY_STAGE = {"acute_phase", "early_recovery", "late_recovery"}
VALID_SWELLING_LEVEL = {"none", "mild", "moderate", "severe"}
VALID_MOBILITY_STATUS = {
    "normal", "slightly_limited", "moderately_limited", "severely_limited",
}
VALID_DOCTOR_RESTRICTIONS = {
    "no_restrictions", "weight_bearing_restricted",
    "movement_restricted", "exercise_restricted", "not_sure",
}
VALID_EXERCISE_APPROVAL = {"yes", "no", "not_sure"}


class PatientContext(BaseModel):
    """
    Clinical context collected from the user.
    The 8 core clinical fields drive the recovery engine.
    Age, sex, and symptom_notes are supplementary.
    """
    # Core clinical fields (required for recovery engine)
    body_part: str | None = Field(default=None)
    treatment_status: str | None = Field(default=None)
    recovery_stage: str | None = Field(default=None)
    pain_level: int | None = Field(default=None, ge=0, le=10)
    swelling_level: str | None = Field(default=None)
    mobility_status: str | None = Field(default=None)
    doctor_restrictions: str | None = Field(default=None)
    exercise_approval: str | None = Field(default=None)

    # Supplementary fields
    age: int | None = Field(default=None, ge=0, le=120)
    sex: str | None = None
    symptom_notes: str | None = Field(default=None, max_length=500)
    additional_notes: str | None = Field(default=None, max_length=1000)


class AuthRequest(BaseModel):
    username: str = Field(min_length=3, max_length=40)
    password: str = Field(min_length=6, max_length=128)


class AssistantRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    analysis: dict | None = None
    patient_context: dict | None = None


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------


def db_connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _add_column_if_missing(conn: sqlite3.Connection, table: str, column: str, col_type: str) -> None:
    """Add a column to a table only if it does not already exist."""
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")


def init_db() -> None:
    with db_connect() as conn:
        # Core tables
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                file_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                prediction_label TEXT,
                condition TEXT,
                confidence REAL,
                body_part TEXT,
                analysis_json TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
            """
        )

        # Migration: add new clinical columns to cases (safe no-op if already present)
        new_columns = [
            ("treatment_status", "TEXT"),
            ("recovery_stage", "TEXT"),
            ("swelling_level", "TEXT"),
            ("mobility_status", "TEXT"),
            ("doctor_restrictions", "TEXT"),
            ("exercise_approval", "TEXT"),
            ("exercise_eligible", "INTEGER"),
            ("exercise_ineligible_reason", "TEXT"),
            ("recovery_risk_level", "TEXT"),
            ("additional_notes", "TEXT"),
        ]
        for col_name, col_type in new_columns:
            _add_column_if_missing(conn, "cases", col_name, col_type)


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_username(username: str) -> str:
    return username.strip().lower()


def hash_password(password: str, salt: str | None = None) -> str:
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt), 120_000)
    return f"{salt}${digest.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        salt, expected = stored_hash.split("$", 1)
    except ValueError:
        return False
    return secrets.compare_digest(hash_password(password, salt), f"{salt}${expected}")


def create_session(user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    with db_connect() as conn:
        conn.execute(
            "INSERT INTO sessions (token, user_id, created_at) VALUES (?, ?, ?)",
            (token, user_id, utc_now()),
        )
    return token


def user_from_authorization(authorization: str | None) -> dict | None:
    if not authorization or not authorization.lower().startswith("bearer "):
        return None

    token = authorization.split(" ", 1)[1].strip()
    with db_connect() as conn:
        row = conn.execute(
            """
            SELECT users.id, users.username
            FROM sessions
            JOIN users ON users.id = sessions.user_id
            WHERE sessions.token = ?
            """,
            (token,),
        ).fetchone()

    if not row:
        return None
    return {"id": row["id"], "username": row["username"], "token": token}


def require_user(authorization: str | None) -> dict:
    user = user_from_authorization(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Please log in first.")
    return user


# ---------------------------------------------------------------------------
# Context helpers
# ---------------------------------------------------------------------------


def clean_patient_context(context: PatientContext | None) -> dict:
    """Return only meaningful intake values for the engine and reports."""
    if context is None:
        return {}

    raw = context.model_dump()
    cleaned = {}
    for key, value in raw.items():
        if value is None:
            continue
        if isinstance(value, str):
            value = value.strip()
            if not value:
                continue
        # BUG FIX: explicitly coerce pain_level to int (fixes edge case where it arrives as a float/string)
        if key == "pain_level":
            try:
                value = int(float(value))
            except (ValueError, TypeError):
                pass
        cleaned[key] = value
    return cleaned


def has_clinical_context(ctx: dict) -> bool:
    """Return True when the 8 required clinical fields are present."""
    required = [
        "body_part", "treatment_status", "recovery_stage", "pain_level",
        "swelling_level", "mobility_status", "doctor_restrictions", "exercise_approval",
    ]
    return all(ctx.get(f) is not None for f in required)


# ---------------------------------------------------------------------------
# Case persistence
# ---------------------------------------------------------------------------


def save_case(user_id: int, file_id: str, analysis: dict) -> int:
    prediction = analysis.get("prediction") or {}
    patient_context = analysis.get("patient_context") or {}
    exercise_eligibility = (analysis.get("recovery_guidance") or {}).get("exercise_eligibility") or {}

    with db_connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO cases (
                user_id, file_id, created_at, prediction_label, condition,
                confidence, body_part, analysis_json,
                treatment_status, recovery_stage, swelling_level, mobility_status,
                doctor_restrictions, exercise_approval, exercise_eligible, exercise_ineligible_reason,
                recovery_risk_level, additional_notes
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                file_id,
                utc_now(),
                prediction.get("label"),
                prediction.get("condition"),
                prediction.get("confidence"),
                patient_context.get("body_part"),
                json.dumps(analysis),
                patient_context.get("treatment_status"),
                patient_context.get("recovery_stage"),
                patient_context.get("swelling_level"),
                patient_context.get("mobility_status"),
                patient_context.get("doctor_restrictions"),
                patient_context.get("exercise_approval"),
                1 if exercise_eligibility.get("eligible") else 0,
                exercise_eligibility.get("reason"),
                (analysis.get("recovery_guidance") or {}).get("recovery_risk_level", {}).get("level"),
                patient_context.get("additional_notes"),
            ),
        )
        return int(cursor.lastrowid)


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------


def find_uploaded_image(file_id: str) -> Path | None:
    for ext in ALLOWED_EXTENSIONS:
        candidate = UPLOAD_DIR / f"{file_id}{ext}"
        if candidate.exists():
            return candidate
    return None


def uploaded_image_url(file_id: str) -> str:
    return f"/uploads/{file_id}"


# ---------------------------------------------------------------------------
# Recovery payload assembly
# ---------------------------------------------------------------------------


def build_full_analysis(
    file_id: str,
    prediction: dict,
    patient_context_data: dict,
) -> dict:
    """
    Assemble the complete analysis response using the new recovery engine.

    Works with both full clinical context (8 fields) and partial context
    (for backward compatibility with the legacy /analyze endpoint).
    """
    condition = (
        "uncertain"
        if (prediction.get("decision") or {}).get("is_uncertain")
        else prediction.get("condition", "unknown")
    )

    if has_clinical_context(patient_context_data):
        # ---- New engine: body-part-aware ----
        eligibility = compute_exercise_eligibility(patient_context_data)
        exercise_plan_raw = get_exercise_plan(
            body_part=patient_context_data.get("body_part", ""),
            recovery_stage=patient_context_data.get("recovery_stage", ""),
            treatment_status=patient_context_data.get("treatment_status", ""),
            exercise_eligible=eligibility["eligible"],
        )
        exercise_plan = enrich_plan_exercise_media(exercise_plan_raw)
        recovery_guidance = build_recovery_guidance(prediction, patient_context_data, exercise_plan)

        # Legacy recommendations block (for backward-compatible UI rendering)
        recommendations = enrich_exercise_media(get_recommendations(condition))

    else:
        # ---- Legacy engine: condition-based fallback ----
        eligibility = {"eligible": False, "reason": "Clinical context not provided.", "rule_triggered": None}
        exercise_plan = {"exercises": [], "exercise_eligible": False}
        recommendations = enrich_exercise_media(get_recommendations(condition))
        recovery_guidance = build_recovery_guidance(prediction, patient_context_data, exercise_plan)

    llm_explanation = explain_result(prediction, recommendations, patient_context_data)

    return {
        "file_id": file_id,
        "original_image_url": uploaded_image_url(file_id),
        "prediction": prediction,
        "patient_context": patient_context_data,
        "exercise_eligibility": eligibility,
        "exercise_plan": exercise_plan,
        "recovery_guidance": recovery_guidance,
        "recommendations": recommendations,  # legacy field kept for PDF renderer
        "llm_explanation": llm_explanation,
        "recommendation_reasoning": {
            "fracture_detection": prediction.get("condition", "unknown"),
            "factors_used": [
                f"Fracture detection result: {prediction.get('condition', 'unknown').title()}",
                f"Body part: {(patient_context_data.get('body_part') or 'not specified').replace('_', ' ').title()}",
                f"Treatment status: {(patient_context_data.get('treatment_status') or 'not specified').replace('_', ' ').title()}",
                f"Recovery stage: {(patient_context_data.get('recovery_stage') or 'not specified').replace('_', ' ').title()}",
                f"Pain level: {patient_context_data.get('pain_level', 'not specified')}/10",
                f"Swelling level: {(patient_context_data.get('swelling_level') or 'not specified').replace('_', ' ').title()}",
                f"Mobility status: {(patient_context_data.get('mobility_status') or 'not specified').replace('_', ' ').title()}",
                f"Doctor restrictions: {(patient_context_data.get('doctor_restrictions') or 'not specified').replace('_', ' ').title()}",
                f"Exercise approval: {(patient_context_data.get('exercise_approval') or 'not specified').replace('_', ' ').title()}",
            ],
        },
    }


def ensure_recovery_payload(analysis: dict) -> dict:
    """Backfill media and nutrition fields for older saved/session analyses."""
    normalized = dict(analysis or {})
    prediction = normalized.get("prediction") or {}
    patient_context = normalized.get("patient_context") or {}
    condition = (
        "uncertain"
        if (prediction.get("decision") or {}).get("is_uncertain")
        else prediction.get("condition", "unknown")
    )

    if not normalized.get("recommendations"):
        normalized["recommendations"] = enrich_exercise_media(get_recommendations(condition))

    if not normalized.get("exercise_plan"):
        if has_clinical_context(patient_context):
            eligibility = compute_exercise_eligibility(patient_context)
            exercise_plan_raw = get_exercise_plan(
                body_part=patient_context.get("body_part", ""),
                recovery_stage=patient_context.get("recovery_stage", ""),
                treatment_status=patient_context.get("treatment_status", ""),
                exercise_eligible=eligibility["eligible"],
            )
            normalized["exercise_plan"] = enrich_plan_exercise_media(exercise_plan_raw)
            normalized["exercise_eligibility"] = eligibility
        else:
            normalized["exercise_plan"] = {"exercises": [], "exercise_eligible": False}
            normalized["exercise_eligibility"] = {
                "eligible": False,
                "reason": "Clinical context not provided.",
                "rule_triggered": None,
            }

    if not (normalized.get("recovery_guidance") or {}).get("nutrition_plan"):
        normalized["recovery_guidance"] = build_recovery_guidance(
            prediction,
            patient_context,
            normalized.get("exercise_plan"),
        )

    return normalized


# ---------------------------------------------------------------------------
# DB init
# ---------------------------------------------------------------------------

init_db()


# ---------------------------------------------------------------------------
# Routes — health & root
# ---------------------------------------------------------------------------


@app.get("/")
def root():
    return {"message": "AI Physio API is running", "version": "2.0.0"}


@app.get("/health")
def health_check():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Routes — auth
# ---------------------------------------------------------------------------


@app.post("/auth/register")
def register(payload: AuthRequest):
    username = normalize_username(payload.username)
    if not username.replace("_", "").replace("-", "").isalnum():
        raise HTTPException(status_code=400, detail="Username can use letters, numbers, hyphen, and underscore.")

    try:
        with db_connect() as conn:
            cursor = conn.execute(
                "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
                (username, hash_password(payload.password), utc_now()),
            )
            user_id = int(cursor.lastrowid)
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=409, detail="Username already exists.")

    token = create_session(user_id)
    return {"token": token, "user": {"id": user_id, "username": username}}


@app.post("/auth/login")
def login(payload: AuthRequest):
    username = normalize_username(payload.username)
    with db_connect() as conn:
        row = conn.execute(
            "SELECT id, username, password_hash FROM users WHERE username = ?",
            (username,),
        ).fetchone()

    if not row or not verify_password(payload.password, row["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid username or password.")

    token = create_session(int(row["id"]))
    return {"token": token, "user": {"id": row["id"], "username": row["username"]}}


@app.get("/auth/me")
def auth_me(authorization: str | None = Header(default=None)):
    user = require_user(authorization)
    return {"user": {"id": user["id"], "username": user["username"]}}


@app.post("/auth/logout")
def logout(authorization: str | None = Header(default=None)):
    user = require_user(authorization)
    with db_connect() as conn:
        conn.execute("DELETE FROM sessions WHERE token = ?", (user["token"],))
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Routes — upload & predict
# ---------------------------------------------------------------------------


@app.post("/upload")
async def upload_xray(file: UploadFile = File(...)):
    """Upload a bone X-ray image. Returns a file_id for prediction."""
    original_filename = file.filename or ""
    ext = Path(original_filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    file_id = str(uuid.uuid4())
    save_path = UPLOAD_DIR / f"{file_id}{ext}"

    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "file_id": file_id,
        "filename": original_filename,
        "saved_as": str(save_path),
        "image_url": uploaded_image_url(file_id),
    }


@app.get("/uploads/{file_id}")
def uploaded_image(file_id: str):
    """Serve an uploaded X-ray image by file id."""
    image_path = find_uploaded_image(file_id)
    if not image_path:
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(image_path)


@app.post("/predict/{file_id}")
async def predict(file_id: str):
    """
    Phase 1: Run AI prediction on a previously uploaded X-ray.
    Returns fracture detection result and confidence score.
    The client shows this result, then collects clinical context for /recovery.
    """
    image_path = find_uploaded_image(file_id)
    if not image_path:
        raise HTTPException(status_code=404, detail="File not found. Please upload first.")

    result = predict_condition(str(image_path))
    if result.get("error"):
        raise HTTPException(status_code=500, detail=result["error"])

    return result


# ---------------------------------------------------------------------------
# Routes — recovery (new two-phase primary endpoint)
# ---------------------------------------------------------------------------


@app.post("/recovery/{file_id}")
async def recovery(
    file_id: str,
    patient_context: PatientContext | None = None,
    authorization: str | None = Header(default=None),
):
    """
    Phase 2: Generate body-part-aware recovery guidance, exercise plan,
    and nutrition recommendations.

    Requires clinical context (8 fields). Runs prediction internally so
    the client only needs to provide file_id + patient context.
    """
    image_path = find_uploaded_image(file_id)
    if not image_path:
        raise HTTPException(status_code=404, detail="File not found. Please upload first.")

    prediction = predict_condition(str(image_path))
    if prediction.get("error"):
        raise HTTPException(status_code=500, detail=prediction["error"])

    patient_context_data = clean_patient_context(patient_context)
    analysis = build_full_analysis(file_id, prediction, patient_context_data)

    user = user_from_authorization(authorization)
    if user:
        analysis["case_id"] = save_case(user["id"], file_id, analysis)

    return analysis


# ---------------------------------------------------------------------------
# Routes — analyze (legacy single-call endpoint, kept for compatibility)
# ---------------------------------------------------------------------------


@app.post("/analyze/{file_id}")
async def analyze(
    file_id: str,
    patient_context: PatientContext | None = None,
    authorization: str | None = Header(default=None),
):
    """
    Full pipeline in one call: predict + recovery guidance.
    Kept for backward compatibility. Prefer /predict + /recovery for new UX.
    """
    image_path = find_uploaded_image(file_id)
    if not image_path:
        raise HTTPException(status_code=404, detail="File not found. Please upload first.")

    prediction = predict_condition(str(image_path))
    if prediction.get("error"):
        raise HTTPException(status_code=500, detail=prediction["error"])

    patient_context_data = clean_patient_context(patient_context)
    analysis = build_full_analysis(file_id, prediction, patient_context_data)

    user = user_from_authorization(authorization)
    if user:
        analysis["case_id"] = save_case(user["id"], file_id, analysis)

    return analysis


# ---------------------------------------------------------------------------
# Routes — cases
# ---------------------------------------------------------------------------


@app.get("/cases")
def list_cases(authorization: str | None = Header(default=None)):
    user = require_user(authorization)
    with db_connect() as conn:
        rows = conn.execute(
            """
            SELECT id, file_id, created_at, prediction_label, condition, confidence,
                   body_part, treatment_status, recovery_stage, exercise_eligible,
                   exercise_ineligible_reason, recovery_risk_level
            FROM cases
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT 50
            """,
            (user["id"],),
        ).fetchall()

    return {"cases": [dict(row) for row in rows]}


@app.get("/cases/{case_id}")
def get_case(case_id: int, authorization: str | None = Header(default=None)):
    user = require_user(authorization)
    with db_connect() as conn:
        row = conn.execute(
            "SELECT analysis_json FROM cases WHERE id = ? AND user_id = ?",
            (case_id, user["id"]),
        ).fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Case not found.")
    return json.loads(row["analysis_json"])


@app.delete("/cases")
def delete_all_cases(authorization: str | None = Header(default=None)):
    user = require_user(authorization)
    with db_connect() as conn:
        conn.execute("DELETE FROM cases WHERE user_id = ?", (user["id"],))
    return {"status": "deleted_all"}


@app.delete("/cases/{case_id}")
def delete_case(case_id: int, authorization: str | None = Header(default=None)):
    user = require_user(authorization)
    with db_connect() as conn:
        cursor = conn.execute(
            "DELETE FROM cases WHERE id = ? AND user_id = ?",
            (case_id, user["id"]),
        )

    if cursor.rowcount == 0:
        raise HTTPException(status_code=404, detail="Case not found.")
    return {"status": "deleted", "case_id": case_id}


# ---------------------------------------------------------------------------
# Routes — reports
# ---------------------------------------------------------------------------


@app.post("/report")
def create_report(analysis: dict = Body(...)):
    """Generate a downloadable PDF report from an analysis response."""
    if not isinstance(analysis, dict) or not analysis.get("prediction"):
        raise HTTPException(status_code=400, detail="Analysis data is required.")

    normalized = ensure_recovery_payload(analysis)
    image_path = find_uploaded_image(str(normalized.get("file_id") or ""))
    pdf_bytes = build_analysis_report_pdf(normalized, image_path)

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="ai-physio-report.pdf"'},
    )


@app.post("/nutrition-report")
def create_nutrition_report(analysis: dict = Body(...)):
    """Generate a focused downloadable PDF diet plan."""
    if not isinstance(analysis, dict):
        raise HTTPException(status_code=400, detail="Analysis data is required.")

    normalized = ensure_recovery_payload(analysis)
    recovery_guidance = normalized.get("recovery_guidance") or {}
    if not recovery_guidance.get("nutrition_plan"):
        raise HTTPException(status_code=400, detail="Nutrition plan data is required.")

    pdf_bytes = build_nutrition_report_pdf(normalized)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="ai-physio-diet-plan.pdf"'},
    )


# ---------------------------------------------------------------------------
# Routes — assistant
# ---------------------------------------------------------------------------


@app.post("/assistant")
def ask_assistant(payload: AssistantRequest):
    """Answer user recovery questions with Groq when configured."""
    analysis = ensure_recovery_payload(payload.analysis or {}) if payload.analysis else None
    return answer_recovery_question(
        payload.question,
        analysis=analysis,
        patient_context=payload.patient_context,
    )


# ---------------------------------------------------------------------------
# Routes — model info
# ---------------------------------------------------------------------------


@app.get("/conditions")
def list_conditions():
    """List all supported conditions and their exercise mappings."""
    return {"conditions": list(EXERCISE_MAP.keys())}


@app.get("/metrics")
def model_metrics():
    """Return saved validation/test metrics for the trained model."""
    metrics_path = PROJECT_ROOT / "evaluation_metrics.json"
    if not metrics_path.exists():
        raise HTTPException(status_code=404, detail="Evaluation metrics not found.")
    return json.loads(metrics_path.read_text(encoding="utf-8"))


@app.get("/confusion-matrix/{split}")
def confusion_matrix_image(split: str):
    """Return a saved confusion matrix image for validation or test split."""
    if split not in {"validation", "test"}:
        raise HTTPException(status_code=400, detail="Split must be 'validation' or 'test'.")

    image_path = PROJECT_ROOT / f"confusion_matrix_{split}.png"
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Confusion matrix image not found.")

    return FileResponse(image_path)

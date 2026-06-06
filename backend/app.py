from pathlib import Path

from fastapi import Body, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
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
    from .recovery_guidance import answer_recovery_question, build_recovery_guidance, enrich_exercise_media
    from .report import build_analysis_report_pdf, build_nutrition_report_pdf
    from .recommendation import EXERCISE_MAP, get_recommendations
except ImportError:
    from llm_explanation import explain_result
    from predict import predict_condition
    from recovery_guidance import answer_recovery_question, build_recovery_guidance, enrich_exercise_media
    from report import build_analysis_report_pdf, build_nutrition_report_pdf
    from recommendation import EXERCISE_MAP, get_recommendations

app = FastAPI(
    title="AI Physio API",
    description="Bone X-ray analysis and physiotherapy recommendation system",
    version="1.0.0",
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

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


class PatientContext(BaseModel):
    age: int | None = Field(default=None, ge=0, le=120)
    sex: str | None = None
    body_part: str | None = None
    pain_level: int | None = Field(default=None, ge=0, le=10)
    swelling: bool | None = None
    recent_trauma: bool | None = None
    symptom_notes: str | None = Field(default=None, max_length=500)


class AuthRequest(BaseModel):
    username: str = Field(min_length=3, max_length=40)
    password: str = Field(min_length=6, max_length=128)


class AssistantRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    analysis: dict | None = None
    patient_context: dict | None = None


def db_connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with db_connect() as conn:
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


def clean_patient_context(context: PatientContext | None) -> dict:
    """Return only meaningful intake values for LLM/report context."""
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
        cleaned[key] = value
    return cleaned


def body_part_detection(patient_context: dict) -> dict:
    """
    Return body-part information for the analysis.

    Automatic body-part detection requires a model trained on body-part labels.
    The current dataset only has fractured/not_fractured folders, so the API is
    explicit about that instead of pretending to infer anatomy.
    """
    provided = patient_context.get("body_part")
    if provided:
        return {
            "body_part": provided,
            "source": "user",
            "status": "provided",
            "note": "Body part was supplied by the user.",
        }

    return {
        "body_part": None,
        "source": "unavailable",
        "status": "needs_labeled_training_data",
        "note": (
            "Automatic body-part detection is not enabled because this project "
            "does not currently include body-part-labeled training folders."
        ),
    }


def save_case(user_id: int, file_id: str, analysis: dict) -> int:
    prediction = analysis.get("prediction") or {}
    patient_context = analysis.get("patient_context") or {}
    with db_connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO cases (
                user_id, file_id, created_at, prediction_label, condition,
                confidence, body_part, analysis_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
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
            ),
        )
        return int(cursor.lastrowid)


def find_uploaded_image(file_id: str) -> Path | None:
    """Return the uploaded image path for a file id, if it exists."""
    for ext in ALLOWED_EXTENSIONS:
        candidate = UPLOAD_DIR / f"{file_id}{ext}"
        if candidate.exists():
            return candidate
    return None


def uploaded_image_url(file_id: str) -> str:
    """Return the API path for a previously uploaded image."""
    return f"/uploads/{file_id}"


def ensure_recovery_payload(analysis: dict) -> dict:
    """Backfill media and nutrition fields for older saved/session analyses."""
    normalized = dict(analysis or {})
    prediction = normalized.get("prediction") or {}
    patient_context = normalized.get("patient_context") or {}
    condition = "uncertain" if (prediction.get("decision") or {}).get("is_uncertain") else prediction.get("condition", "unknown")

    recommendations = normalized.get("recommendations") or get_recommendations(condition)
    normalized["recommendations"] = enrich_exercise_media(recommendations)

    if not (normalized.get("recovery_guidance") or {}).get("nutrition_plan"):
        normalized["recovery_guidance"] = build_recovery_guidance(
            prediction,
            normalized["recommendations"],
            patient_context,
        )

    return normalized


init_db()


@app.get("/")
def root():
    return {"message": "AI Physio API is running", "version": "1.0.0"}


@app.get("/health")
def health_check():
    return {"status": "ok"}


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


@app.post("/upload")
async def upload_xray(file: UploadFile = File(...)):
    """
    Upload a bone X-ray image.
    Returns a file_id that can be used to run prediction.
    """
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
    Run AI prediction on a previously uploaded X-ray.
    Returns detected condition and confidence score.
    """
    image_path = find_uploaded_image(file_id)

    if not image_path:
        raise HTTPException(status_code=404, detail="File not found. Please upload first.")

    result = predict_condition(str(image_path))

    if result.get("error"):
        raise HTTPException(status_code=500, detail=result["error"])

    return result


@app.post("/analyze/{file_id}")
async def analyze(
    file_id: str,
    patient_context: PatientContext | None = None,
    authorization: str | None = Header(default=None),
):
    """
    Full pipeline: predict condition AND return exercise recommendations.
    This is the main endpoint for end-to-end analysis.
    """
    image_path = find_uploaded_image(file_id)

    if not image_path:
        raise HTTPException(status_code=404, detail="File not found. Please upload first.")

    prediction = predict_condition(str(image_path))

    if prediction.get("error"):
        raise HTTPException(status_code=500, detail=prediction["error"])

    patient_context_data = clean_patient_context(patient_context)
    detected_body_part = body_part_detection(patient_context_data)
    condition = "uncertain" if (prediction.get("decision") or {}).get("is_uncertain") else prediction.get("condition", "unknown")
    recommendations = enrich_exercise_media(get_recommendations(condition))
    llm_explanation = explain_result(prediction, recommendations, patient_context_data)
    recovery_guidance = build_recovery_guidance(prediction, recommendations, patient_context_data)

    analysis = {
        "file_id": file_id,
        "original_image_url": uploaded_image_url(file_id),
        "prediction": prediction,
        "recommendations": recommendations,
        "recovery_guidance": recovery_guidance,
        "patient_context": patient_context_data,
        "body_part_detection": detected_body_part,
        "llm_explanation": llm_explanation,
    }

    user = user_from_authorization(authorization)
    if user:
        analysis["case_id"] = save_case(user["id"], file_id, analysis)

    return analysis


@app.get("/cases")
def list_cases(authorization: str | None = Header(default=None)):
    user = require_user(authorization)
    with db_connect() as conn:
        rows = conn.execute(
            """
            SELECT id, file_id, created_at, prediction_label, condition, confidence, body_part
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
    """Generate a focused downloadable PDF diet plan from an analysis response."""
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


@app.post("/assistant")
def ask_assistant(payload: AssistantRequest):
    """Answer user recovery questions with Groq when configured."""
    analysis = ensure_recovery_payload(payload.analysis or {}) if payload.analysis else None
    return answer_recovery_question(
        payload.question,
        analysis=analysis,
        patient_context=payload.patient_context,
    )


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
            

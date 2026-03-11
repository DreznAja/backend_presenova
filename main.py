"""
Presenova Face Recognition API
Run locally: uvicorn main:app --reload --port 8000
Deploy: Hugging Face Spaces (Docker, port 7860)
"""
import os
import base64
import uuid
from pathlib import Path
from io import BytesIO
from typing import Optional
from contextlib import asynccontextmanager

import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

# ─── Supabase client ──────────────────────────────────
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Local face database directory
FACES_DIR = Path("./faces")
FACES_DIR.mkdir(exist_ok=True)


def sync_faces_from_supabase():
    """
    Download all registered face photos from Supabase Storage to local faces/.
    Called on startup — critical for HF Spaces (ephemeral storage).
    """
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("[WARN] Supabase not configured, skipping face sync")
        return
    try:
        files = supabase_client.storage.from_("face-photos").list()
        if not files:
            print("[INFO] No faces in Supabase Storage yet")
            return
        synced = 0
        for f in files:
            name = f.get("name", "")
            if not name.endswith(".jpg"):
                continue
            local = FACES_DIR / name
            if local.exists():
                continue  # already cached
            url = supabase_client.storage.from_("face-photos").get_public_url(name)
            import requests
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                local.write_bytes(r.content)
                synced += 1
        print(f"[INFO] Face sync complete: {synced} new photos downloaded")
    except Exception as e:
        print(f"[WARN] Face sync failed: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: sync faces from Supabase
    print("[INFO] Starting Presenova Face API...")
    sync_faces_from_supabase()
    yield
    # Shutdown (nothing to clean up)


app = FastAPI(title="Presenova Face API", version="1.0.0", lifespan=lifespan)

# CORS — allow Next.js dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Models ───────────────────────────────────────────
class RegisterFaceRequest(BaseModel):
    siswa_id: str
    nama: str
    image_base64: str  # base64 encoded image (data:image/jpeg;base64,... or raw)

class RecognizeRequest(BaseModel):
    image_base64: str  # base64 encoded image from webcam

class RecognizeResponse(BaseModel):
    found: bool
    siswa_id: Optional[str] = None
    nama: Optional[str] = None
    confidence: Optional[float] = None
    foto_url: Optional[str] = None
    message: str

# ─── Helpers ──────────────────────────────────────────
def decode_base64_image(b64_str: str) -> Image.Image:
    """Decode base64 string to PIL Image."""
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    img_bytes = base64.b64decode(b64_str)
    return Image.open(BytesIO(img_bytes)).convert("RGB")


def upload_to_supabase(image: Image.Image, bucket: str, filename: str) -> str:
    """Upload PIL Image to Supabase Storage, return public URL."""
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=85)
    buf.seek(0)
    path = f"{filename}.jpg"
    supabase_client.storage.from_(bucket).upload(
        path, buf.read(), {"content-type": "image/jpeg", "upsert": "true"}
    )
    result = supabase_client.storage.from_(bucket).get_public_url(path)
    return result


def get_face_image_path(siswa_id: str) -> Path:
    return FACES_DIR / f"{siswa_id}.jpg"


# ─── Routes ───────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "service": "Presenova Face API"}


@app.post("/register-face")
def register_face(req: RegisterFaceRequest):
    """
    Register a student's face.
    Saves image locally for DeepFace recognition + uploads to Supabase Storage.
    """
    try:
        img = decode_base64_image(req.image_base64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    face_warning = ""
    # Validate face is present (lenient mode — enforce_detection=False)
    try:
        from deepface import DeepFace
        faces = DeepFace.extract_faces(
            img=np.array(img),
            enforce_detection=False,  # don't crash if model just loaded
            detector_backend="opencv"
        )
        # Filter faces with confidence > 0.5
        real_faces = [f for f in faces if f.get("confidence", 1.0) > 0.5]
        if not real_faces:
            face_warning = "Peringatan: wajah kurang terdeteksi — pastikan foto jelas"
    except Exception as e:
        face_warning = f"Peringatan deteksi wajah: {e}"
        print(f"[WARN] Face detection: {e}")

    # Save locally (used by DeepFace.find)
    face_path = get_face_image_path(req.siswa_id)
    img.save(str(face_path), "JPEG")

    # Upload to Supabase Storage
    foto_url = ""
    try:
        foto_url = upload_to_supabase(img, "face-photos", req.siswa_id)
        # Update siswa table with foto_url
        supabase_client.table("siswa").update({"foto_url": foto_url}).eq("id", req.siswa_id).execute()
    except Exception as e:
        print(f"[WARN] Supabase upload failed: {e}")

    return {
        "success": True,
        "siswa_id": req.siswa_id,
        "foto_url": foto_url,
        "message": f"Wajah {req.nama} berhasil didaftarkan",
        "warning": face_warning
    }


@app.post("/recognize", response_model=RecognizeResponse)
def recognize_face(req: RecognizeRequest):
    """
    Recognize a face from webcam capture.
    Returns matching student info.
    """
    # Check if any faces are registered
    face_files = list(FACES_DIR.glob("*.jpg"))
    if not face_files:
        return RecognizeResponse(
            found=False,
            message="Belum ada wajah terdaftar. Daftarkan siswa terlebih dahulu."
        )

    try:
        img = decode_base64_image(req.image_base64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    # Save temp image for DeepFace
    temp_path = FACES_DIR / f"_temp_{uuid.uuid4().hex}.jpg"
    img.save(str(temp_path), "JPEG")

    try:
        from deepface import DeepFace

        results = DeepFace.find(
            img_path=str(temp_path),
            db_path=str(FACES_DIR),
            model_name="Facenet",
            detector_backend="opencv",
            enforce_detection=False,
            silent=True
        )

        temp_path.unlink(missing_ok=True)

        # Filter out temp file and parse results
        df = results[0] if results else None
        if df is None or df.empty:
            return RecognizeResponse(found=False, message="Wajah tidak dikenali")

        # Filter out temp files from results
        df = df[~df["identity"].str.contains("_temp_")]
        if df.empty:
            return RecognizeResponse(found=False, message="Wajah tidak dikenali")

        best = df.iloc[0]
        identity_path = Path(best["identity"])
        siswa_id = identity_path.stem  # filename = siswa_id

        # Distance → confidence (lower distance = more confident)
        distance = float(best.get("distance", 0.5))
        confidence = round(max(0, (1 - distance / 0.6)) * 100, 1)

        if confidence < 40:
            return RecognizeResponse(found=False, message=f"Wajah kurang jelas (confidence: {confidence}%)")

        # Fetch siswa data from Supabase
        res = supabase_client.table("siswa").select("id, nama, kelas_id, foto_url").eq("id", siswa_id).single().execute()
        if not res.data:
            return RecognizeResponse(found=False, message="Siswa tidak ditemukan di database")

        siswa = res.data
        return RecognizeResponse(
            found=True,
            siswa_id=siswa["id"],
            nama=siswa["nama"],
            confidence=confidence,
            foto_url=siswa.get("foto_url", ""),
            message=f"Wajah dikenali: {siswa['nama']} ({confidence}%)"
        )

    except Exception as e:
        temp_path.unlink(missing_ok=True)
        print(f"[ERROR] Recognition failed: {e}")
        return RecognizeResponse(found=False, message=f"Error: {str(e)}")


@app.post("/upload-scan-capture")
def upload_scan_capture(req: RegisterFaceRequest):
    """Upload webcam capture at scan time to Supabase scan-captures bucket."""
    try:
        img = decode_base64_image(req.image_base64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    filename = f"{req.siswa_id}_{uuid.uuid4().hex[:8]}"
    try:
        url = upload_to_supabase(img, "scan-captures", filename)
        return {"success": True, "foto_url": url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/delete-face/{siswa_id}")
def delete_face(siswa_id: str):
    """Remove student face from local DB."""
    face_path = get_face_image_path(siswa_id)
    if face_path.exists():
        face_path.unlink()
    return {"success": True, "message": f"Face for {siswa_id} deleted"}

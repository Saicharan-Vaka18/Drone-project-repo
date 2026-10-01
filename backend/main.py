import csv
import io
import os
import secrets
import smtplib
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from pathlib import Path
import typing

import cv2
import jwt
import numpy as np
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, EmailStr, Field
from ultralytics import YOLO

# ------------------------------------------------------------
# AeroTrash Detect - FastAPI + YOLO backend
# ------------------------------------------------------------
# IMPORTANT:
# Your trained model is named my_model.pt.
# The .env file in this folder points to the exact Windows path you gave:
# C:\Users\chara\OneDrive\Desktop\training_yolo_model\second_try\my_model\my_model.pt
#
# You can also copy my_model.pt to backend/models/my_model.pt and change
# YOLO_MODEL_PATH in .env to ./models/my_model.pt.
# ------------------------------------------------------------

BASE = Path(__file__).resolve().parent
load_dotenv(BASE / ".env")

DB_PATH = Path(os.getenv("DATABASE_PATH", BASE / "aerotrashdetect.db"))
MODEL_PATH = Path(os.getenv("YOLO_MODEL_PATH", BASE / "models" / "my_model.pt")).expanduser()
MEDIA_DIR = BASE / "media"
MEDIA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

JWT_SECRET = os.getenv("JWT_SECRET", "development-only-change-me")
JWT_ALGORITHM = "HS256"
TOKEN_HOURS = 24
MODEL_CONFIDENCE = float(os.getenv("YOLO_CONFIDENCE", "0.55"))

app = FastAPI(title="AeroTrash Detect API", version="2.0.0")
origins = [
    x.strip()
    for x in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:5500,http://127.0.0.1:5500,http://localhost:8000,http://127.0.0.1:8000",
    ).split(",")
    if x.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer(auto_error=False)
MODEL = None
CAMERAS = {}
LAST_TELEMETRY = {}


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def hash_password(password: str, salt: typing.Optional[bytes] = None):
    import hashlib

    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 210_000)
    return salt.hex() + "$" + digest.hex()


def verify_password(password: str, stored: str):
    import hashlib

    try:
        salt_hex, digest_hex = stored.split("$", 1)
        test = hashlib.pbkdf2_hmac(
            "sha256", password.encode(), bytes.fromhex(salt_hex), 210_000
        )
        return secrets.compare_digest(test.hex(), digest_hex)
    except Exception:
        return False


def init_db():
    conn = db()
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS drones (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            model TEXT NOT NULL DEFAULT 'Unknown',
            camera_source TEXT NOT NULL,
            api_key TEXT UNIQUE NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        );

        CREATE TABLE IF NOT EXISTS telemetry (
            drone_id INTEGER PRIMARY KEY,
            battery REAL,
            latitude REAL,
            longitude REAL,
            altitude REAL,
            speed REAL,
            connected INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(drone_id) REFERENCES drones(id)
        );

        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            drone_id INTEGER NOT NULL,
            label TEXT NOT NULL,
            confidence REAL NOT NULL,
            latitude REAL,
            longitude REAL,
            altitude REAL,
            image_path TEXT,
            detected_at TEXT NOT NULL,
            reported_at TEXT,
            status TEXT NOT NULL DEFAULT 'pending',
            FOREIGN KEY(drone_id) REFERENCES drones(id)
        );
        """
    )
    conn.commit()
    conn.close()


init_db()


class RegisterIn(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)


class LoginIn(BaseModel):
    email: EmailStr
    password: str


class DroneIn(BaseModel):
    name: str = Field(min_length=2, max_length=80)
    model: str = Field(default="Unknown", max_length=80)
    camera_source: str = Field(min_length=1, max_length=500)


class TelemetryIn(BaseModel):
    api_key: str
    battery: typing.Optional[float] = Field(default=None, ge=0, le=100)
    latitude: typing.Optional[float] = Field(default=None, ge=-90, le=90)
    longitude: typing.Optional[float] = Field(default=None, ge=-180, le=180)
    altitude: typing.Optional[float] = None
    speed: typing.Optional[float] = None
    connected: bool = True


class ReportIn(BaseModel):
    message: typing.Optional[str] = None


def make_token(user_id: int):
    payload = {
        "sub": str(user_id),
        "exp": datetime.now(timezone.utc) + timedelta(hours=TOKEN_HOURS),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_user_token(token: str):
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return int(payload["sub"])
    except Exception as exc:
        raise HTTPException(401, "Invalid or expired token") from exc


def current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if not credentials:
        raise HTTPException(401, "Authentication required")
    user_id = decode_user_token(credentials.credentials)
    conn = db()
    user = conn.execute(
        "SELECT id, email, created_at FROM users WHERE id=?", (user_id,)
    ).fetchone()
    conn.close()
    if not user:
        raise HTTPException(401, "User not found")
    return dict(user)


def get_user_drone(user_id: int, drone_id: int):
    conn = db()
    drone = conn.execute(
        "SELECT * FROM drones WHERE id=? AND user_id=?", (drone_id, user_id)
    ).fetchone()
    conn.close()
    if not drone:
        raise HTTPException(404, "Drone not found")
    return dict(drone)


def load_model():
    global MODEL
    if MODEL is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                503,
                f"YOLO model not found at: {MODEL_PATH}. "
                "Check backend/.env or copy my_model.pt into backend/models/.",
            )
        try:
            MODEL = YOLO(str(MODEL_PATH))
        except Exception as exc:
            raise HTTPException(500, f"Could not load YOLO model: {exc}") from exc
    return MODEL

@app.on_event("startup")
def startup_load_model():
    load_model()

def source_value(source: str):
    source = source.strip()
    try:
        return int(source)
    except ValueError:
        return source


def camera_for(drone_id: int, source: str):
    existing = CAMERAS.get(drone_id)
    if (
        existing
        and existing.get("source") == source
        and existing.get("cap")
        and existing["cap"].isOpened()
    ):
        return existing

    if existing and existing.get("cap"):
        existing["cap"].release()

    cap = cv2.VideoCapture(source_value(source))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    state = {"cap": cap, "source": source, "last_open_attempt": time.time()}
    CAMERAS[drone_id] = state
    return state


def save_detection(drone_id, label, confidence, telemetry, frame):
    # Prevent the same label from flooding the database on consecutive frames.
    key = f"{drone_id}:{label}"
    stamp = time.time()
    previous = getattr(save_detection, "_last", {}).get(key, 0)
    if stamp - previous < 3:
        return
    if not hasattr(save_detection, "_last"):
        save_detection._last = {}
    save_detection._last[key] = stamp

    filename = f"detection_{drone_id}_{int(stamp * 1000)}.jpg"
    path = MEDIA_DIR / filename
    cv2.imwrite(str(path), frame)

    conn = db()
    conn.execute(
        """
        INSERT INTO detections
        (drone_id,label,confidence,latitude,longitude,altitude,image_path,detected_at,status)
        VALUES (?,?,?,?,?,?,?,?,?)
        """,
        (
            drone_id,
            label,
            confidence,
            telemetry.get("latitude"),
            telemetry.get("longitude"),
            telemetry.get("altitude"),
            filename,
            now_iso(),
            "pending",
        ),
    )
    conn.commit()
    conn.close()


def run_yolo(frame, drone_id=None, telemetry=None, save_events=True):
    model = load_model()
    telemetry = telemetry or {}
    results = model.predict(frame, conf=MODEL_CONFIDENCE, verbose=False)
    annotated = frame.copy()
    detections = []

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        names = result.names
        for box in boxes:
            confidence = float(box.conf[0])
            cls = int(box.cls[0])
            label = str(names.get(cls, cls))
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (220, 38, 38), 2)
            cv2.putText(
                annotated,
                f"{label} {confidence:.0%}",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (220, 38, 38),
                2,
            )

            detections.append(
                {
                    "label": label,
                    "confidence": confidence,
                    "class_id": cls,
                    "box": [x1, y1, x2, y2],
                }
            )

            if save_events and drone_id is not None:
                save_detection(drone_id, label, confidence, telemetry, frame)

    return annotated, detections


def mjpeg_generator(drone_id: int, source: str):
    while True:
        state = camera_for(drone_id, source)
        cap = state["cap"]
        ok, frame = cap.read()
        if not ok:
            cap.release()
            CAMERAS.pop(drone_id, None)
            time.sleep(1)
            continue

        telemetry = LAST_TELEMETRY.get(drone_id, {})
        try:
            annotated, _ = run_yolo(
                frame,
                drone_id=drone_id,
                telemetry=telemetry,
                save_events=True,
            )
        except Exception as exc:
            # Keep the stream alive, but display the backend/model problem on the frame.
            annotated = frame.copy()
            cv2.putText(
                annotated,
                f"YOLO ERROR: {str(exc)[:100]}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2,
            )

        ok, encoded = cv2.imencode(
            ".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 82]
        )
        if not ok:
            continue
        yield (
            b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
            + encoded.tobytes()
            + b"\r\n"
        )


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "service": "AeroTrash Detect API",
        "model_path": str(MODEL_PATH),
        "model_exists": MODEL_PATH.exists(),
        "model_loaded": MODEL is not None,
        "confidence": MODEL_CONFIDENCE,
        "time": now_iso(),
    }


@app.get("/api/model/status")
def model_status():
    return {
        "name": MODEL_PATH.name,
        "path": str(MODEL_PATH),
        "exists": MODEL_PATH.exists(),
        "loaded": MODEL is not None,
        "confidence": MODEL_CONFIDENCE,
    }


@app.post("/api/model/test")
async def model_test(file: UploadFile = File(...), user=Depends(current_user)):
    content = await file.read()
    image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(400, "The uploaded file is not a valid image")

    _, detections = run_yolo(image, save_events=False)
    return {
        "model": MODEL_PATH.name,
        "detections": detections,
        "count": len(detections),
    }


@app.post("/api/auth/register")
def register(data: RegisterIn):
    conn = db()
    exists = conn.execute(
        "SELECT id FROM users WHERE email=?", (data.email.lower(),)
    ).fetchone()
    if exists:
        conn.close()
        raise HTTPException(409, "An account with this email already exists")

    cur = conn.execute(
        "INSERT INTO users(email,password_hash,created_at) VALUES(?,?,?)",
        (data.email.lower(), hash_password(data.password), now_iso()),
    )
    conn.commit()
    user_id = cur.lastrowid
    conn.close()
    return {
        "token": make_token(user_id),
        "user": {"id": user_id, "email": data.email.lower()},
    }


@app.post("/api/auth/login")
def login(data: LoginIn):
    conn = db()
    user = conn.execute(
        "SELECT * FROM users WHERE email=?", (data.email.lower(),)
    ).fetchone()
    conn.close()
    if not user or not verify_password(data.password, user["password_hash"]):
        raise HTTPException(401, "Invalid email or password")
    return {
        "token": make_token(user["id"]),
        "user": {"id": user["id"], "email": user["email"]},
    }


@app.get("/api/auth/me")
def me(user=Depends(current_user)):
    return user


@app.get("/api/drones")
def list_drones(user=Depends(current_user)):
    conn = db()
    rows = conn.execute(
        """
        SELECT d.id,d.name,d.model,d.camera_source,d.created_at,
               t.battery,t.latitude,t.longitude,t.altitude,t.speed,t.connected,t.updated_at
        FROM drones d LEFT JOIN telemetry t ON t.drone_id=d.id
        WHERE d.user_id=? ORDER BY d.id DESC
        """,
        (user["id"],),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@app.post("/api/drones")
def create_drone(data: DroneIn, user=Depends(current_user)):
    api_key = "AS_" + secrets.token_urlsafe(24)
    conn = db()
    cur = conn.execute(
        """
        INSERT INTO drones(user_id,name,model,camera_source,api_key,created_at)
        VALUES(?,?,?,?,?,?)
        """,
        (user["id"], data.name, data.model, data.camera_source, api_key, now_iso()),
    )
    conn.commit()
    drone_id = cur.lastrowid
    conn.close()
    return {
        "id": drone_id,
        "name": data.name,
        "model": data.model,
        "camera_source": data.camera_source,
        "api_key": api_key,
    }


@app.delete("/api/drones/{drone_id}")
def delete_drone(drone_id: int, user=Depends(current_user)):
    get_user_drone(user["id"], drone_id)
    if drone_id in CAMERAS:
        CAMERAS[drone_id]["cap"].release()
        CAMERAS.pop(drone_id, None)

    conn = db()
    conn.execute("DELETE FROM telemetry WHERE drone_id=?", (drone_id,))
    conn.execute("DELETE FROM detections WHERE drone_id=?", (drone_id,))
    conn.execute("DELETE FROM drones WHERE id=?", (drone_id,))
    conn.commit()
    conn.close()
    return {"ok": True}


@app.post("/api/telemetry")
def telemetry(data: TelemetryIn):
    conn = db()
    drone = conn.execute(
        "SELECT id FROM drones WHERE api_key=?", (data.api_key,)
    ).fetchone()
    if not drone:
        conn.close()
        raise HTTPException(401, "Invalid drone API key")

    drone_id = drone["id"]
    payload = data.model_dump(exclude={"api_key"})
    updated = now_iso()

    conn.execute(
        """
        INSERT INTO telemetry(drone_id,battery,latitude,longitude,altitude,speed,connected,updated_at)
        VALUES(?,?,?,?,?,?,?,?)
        ON CONFLICT(drone_id) DO UPDATE SET
        battery=excluded.battery,
        latitude=excluded.latitude,
        longitude=excluded.longitude,
        altitude=excluded.altitude,
        speed=excluded.speed,
        connected=excluded.connected,
        updated_at=excluded.updated_at
        """,
        (
            drone_id,
            payload["battery"],
            payload["latitude"],
            payload["longitude"],
            payload["altitude"],
            payload["speed"],
            int(payload["connected"]),
            updated,
        ),
    )
    conn.commit()
    conn.close()
    LAST_TELEMETRY[drone_id] = {**payload, "updated_at": updated}
    return {"ok": True, "drone_id": drone_id}


@app.get("/api/drones/{drone_id}/stream")
def stream(drone_id: int, token: typing.Optional[str] = None):
    if not token:
        raise HTTPException(401, "Stream token required")
    user_id = decode_user_token(token)
    drone = get_user_drone(user_id, drone_id)
    return StreamingResponse(
        mjpeg_generator(drone_id, drone["camera_source"]),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/detections")
def detections(user=Depends(current_user)):
    conn = db()
    rows = conn.execute(
        """
        SELECT x.*, d.name AS drone_name
        FROM detections x JOIN drones d ON d.id=x.drone_id
        WHERE d.user_id=? ORDER BY x.id DESC LIMIT 500
        """,
        (user["id"],),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@app.get("/api/detections/{detection_id}")
def detection(detection_id: int, user=Depends(current_user)):
    conn = db()
    row = conn.execute(
        """
        SELECT x.*, d.name AS drone_name
        FROM detections x JOIN drones d ON d.id=x.drone_id
        WHERE x.id=? AND d.user_id=?
        """,
        (detection_id, user["id"]),
    ).fetchone()
    conn.close()
    if not row:
        raise HTTPException(404, "Detection not found")
    return dict(row)


@app.get("/media/{filename}")
def media(filename: str):
    safe = Path(filename).name
    path = MEDIA_DIR / safe
    if not path.exists():
        raise HTTPException(404, "Image not found")
    return FileResponse(path)


def send_email(detection, user_message=""):
    host = os.getenv("SMTP_HOST")
    port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER")
    smtp_pass = os.getenv("SMTP_PASS")
    recipient = os.getenv("MUNICIPALITY_EMAIL")

    if not all([host, smtp_user, smtp_pass, recipient]):
        raise RuntimeError(
            "SMTP_HOST, SMTP_USER, SMTP_PASS and MUNICIPALITY_EMAIL must be configured"
        )

    msg = EmailMessage()
    msg["Subject"] = f"AeroTrash Detect Trash Detection #{detection['id']}"
    msg["From"] = smtp_user
    msg["To"] = recipient
    msg.set_content(
        f"""AeroTrash Detect municipality alert

Detection: #{detection['id']}
Drone: {detection['drone_name']}
Type: {detection['label']}
Confidence: {detection['confidence']:.1%}
Latitude: {detection['latitude']}
Longitude: {detection['longitude']}
Altitude: {detection['altitude']}
Detected: {detection['detected_at']}

{user_message}
"""
    )
    with smtplib.SMTP(host, port) as server:
        server.starttls()
        server.login(smtp_user, smtp_pass)
        server.send_message(msg)


@app.post("/api/detections/{detection_id}/report")
def report_detection(
    detection_id: int, data: ReportIn, user=Depends(current_user)
):
    conn = db()
    row = conn.execute(
        """
        SELECT x.*, d.name AS drone_name
        FROM detections x JOIN drones d ON d.id=x.drone_id
        WHERE x.id=? AND d.user_id=?
        """,
        (detection_id, user["id"]),
    ).fetchone()
    if not row:
        conn.close()
        raise HTTPException(404, "Detection not found")

    detection_row = dict(row)
    try:
        send_email(detection_row, data.message or "")
    except RuntimeError as exc:
        conn.close()
        raise HTTPException(503, str(exc)) from exc
    except Exception as exc:
        conn.close()
        raise HTTPException(502, f"Municipality email failed: {exc}") from exc

    conn.execute(
        "UPDATE detections SET status='reported', reported_at=? WHERE id=?",
        (now_iso(), detection_id),
    )
    conn.commit()
    conn.close()
    return {"ok": True, "status": "reported"}


@app.get("/api/detections/export.csv")
def export_csv(token: typing.Optional[str] = None):
    if not token:
        raise HTTPException(401, "Export token required")
    user_id = decode_user_token(token)

    conn = db()
    rows = conn.execute(
        """
        SELECT x.id,d.name AS drone,x.label,x.confidence,x.latitude,x.longitude,
               x.altitude,x.detected_at,x.status,x.reported_at
        FROM detections x JOIN drones d ON d.id=x.drone_id
        WHERE d.user_id=? ORDER BY x.id DESC
        """,
        (user_id,),
    ).fetchall()
    conn.close()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(
        [
            "id",
            "drone",
            "label",
            "confidence",
            "latitude",
            "longitude",
            "altitude",
            "detected_at",
            "status",
            "reported_at",
        ]
    )
    for row in rows:
        writer.writerow(list(row))

    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={
            "Content-Disposition": "attachment; filename=aerotrashdetect-detections.csv"
        },
    )

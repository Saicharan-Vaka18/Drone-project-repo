# AeroTrash Detect

Professional drone-based trash detection dashboard using:
- HTML/CSS/JavaScript frontend
- FastAPI backend
- Ultralytics YOLO
- SQLite
- OpenCV camera streaming
- GPS/battery/altitude/speed telemetry

## Your trained model

The backend is configured for your trained file:

`my_model.pt`

Current configured Windows path:

`C:\Users\chara\OneDrive\Desktop\training_yolo_model\second_try\my_model\my_model.pt`

This is stored in:

`backend/.env`

You can also copy the model into:

`backend/models/my_model.pt`

and set:

`YOLO_MODEL_PATH=./models/my_model.pt`

## Run

Open a terminal in the project root:

```powershell
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Check the model:

`http://127.0.0.1:8000/api/model/status`

Expected when the path is correct:

```json
{
  "name": "my_model.pt",
  "exists": true
}
```

The model is loaded when the first inference request is made. This avoids loading the potentially large model unnecessarily during server startup.

## Test the model

Open:

`http://127.0.0.1:8000/docs`

Use `/api/auth/register` or `/api/auth/login`, authorize, then upload a test image to `/api/model/test`.

## Live Server frontend

Open:

`01_HTML/index.html`

with VS Code Live Server.

For actual YOLO inference and live camera processing, the Python backend must also be running.

## Telemetry

Battery, GPS, altitude and speed must be supplied by the drone/onboard computer. The camera feed alone cannot reliably provide them.

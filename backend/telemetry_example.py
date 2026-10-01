import requests

API = "http://YOUR_BACKEND_IP:8000/api/telemetry"
DRONE_API_KEY = "PASTE_THE_KEY_RETURNED_WHEN_YOU_CREATE_THE_DRONE"

payload = {
    "api_key": DRONE_API_KEY,
    "battery": 82,
    "latitude": 16.3067,
    "longitude": 80.4365,
    "altitude": 64.2,
    "speed": 8.4,
    "connected": True
}

requests.post(API, json=payload, timeout=5)

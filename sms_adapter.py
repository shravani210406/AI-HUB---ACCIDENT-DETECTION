"""TextBee SMS integration for the AI Hub accident-detection web app.

Only this file talks to TextBee. Keep API credentials outside source code.
"""

import os
from typing import Any

import requests

TEXTBEE_URL = "https://api.textbee.dev/api/v1/gateway/send-sms"


def _required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing {name}. Fill it in .env before running the app.")
    return value


def build_accident_message(video_filename: str, timestamp: str | None, confidence: Any) -> str:
    """Create the SMS from the information returned by the detector."""
    time_text = timestamp or "Not available"
    confidence_text = f"{confidence}%" if confidence is not None else "Not available"

    # Keep the message short and easy to read on a phone.
    return (
        "ACCIDENT DETECTED\n"
        f"Video: {video_filename}\n"
        f"Time: {time_text}\n"
        f"Confidence: {confidence_text}"
    )


def send_accident_sms(video_filename: str, timestamp: str | None = None, confidence: Any = None) -> dict:
    """Send one accident alert through TextBee.

    TextBee expects:
      x-api-key header
      recipients: [phone number(s)]
      message: SMS text
      optional deviceId
    """
    api_key = _required("TEXTBEE_API_KEY")
    recipient = _required("TEXTBEE_RECIPIENT")
    device_id = os.getenv("TEXTBEE_DEVICE_ID", "").strip()

    payload = {
        "recipients": [recipient],
        "message": build_accident_message(video_filename, timestamp, confidence),
    }
    if device_id:
        payload["deviceId"] = device_id

    response = requests.post(
        TEXTBEE_URL,
        headers={
            "x-api-key": api_key,
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=30,
    )

    try:
        data = response.json()
    except ValueError:
        data = {"raw": response.text}

    if not response.ok:
        raise RuntimeError(f"TextBee HTTP {response.status_code}: {data}")

    result = data.get("data", data)
    return {
        "ok": bool(result.get("success", True)),
        "message": result.get("message", "SMS accepted by TextBee"),
        "sms_batch_id": result.get("smsBatchId"),
        "response": data,
    }

from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from pathlib import Path
import os
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from detector_adapter import detect_accident
from sms_adapter import send_accident_sms

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "change-this-secret-key")

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
SNAPSHOT_DIR = UPLOAD_DIR / "snapshots"
UPLOAD_DIR.mkdir(exist_ok=True)
SNAPSHOT_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/detect", methods=["POST"])
def detect():
    if "video" not in request.files:
        flash("Please select a video.")
        return redirect(url_for("index"))

    video = request.files["video"]
    if not video.filename:
        flash("Please select a video.")
        return redirect(url_for("index"))
    if not allowed_file(video.filename):
        flash("Unsupported video format.")
        return redirect(url_for("index"))

    filename = secure_filename(video.filename)
    video_path = UPLOAD_DIR / filename
    video.save(video_path)

    try:
        result = detect_accident(str(video_path))

        sms_status = "Not required"
        if result.get("accident"):
            try:
                sms = send_accident_sms(
                    video_filename=filename,
                    timestamp=result.get("timestamp"),
                    confidence=result.get("confidence"),
                )
                sms_status = sms.get("message", "SMS sent")
            except Exception as exc:
                sms_status = f"SMS error: {exc}"

        return render_template(
            "result.html",
            filename=filename,
            video_url=url_for("uploaded_video", filename=filename),
            result=result,
            sms_status=sms_status,
            error=None,
        )
    except Exception as exc:
        return render_template(
            "result.html",
            filename=filename,
            video_url=url_for("uploaded_video", filename=filename),
            result=None,
            sms_status=None,
            error=f"Detection pipeline error: {exc}",
        )

@app.route("/uploads/<path:filename>")
def uploaded_video(filename):
    return send_from_directory(UPLOAD_DIR, filename)

@app.route("/snapshots/<path:filename>")
def snapshot(filename):
    return send_from_directory(SNAPSHOT_DIR, filename)

if __name__ == "__main__":
    app.run(debug=True)

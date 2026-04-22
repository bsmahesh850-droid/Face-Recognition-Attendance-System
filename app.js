from flask import Flask, render_template, request, redirect, url_for, send_file
import cv2
import os
import csv
from datetime import datetime
from utils.face_utils import load_known_faces, recognize_faces

app = Flask(__name__)

KNOWN_FACES_DIR = "known_faces"
ATTENDANCE_FILE = "database/attendance.csv"

# Ensure folders exist
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
os.makedirs("database", exist_ok=True)

# Initialize CSV
if not os.path.exists(ATTENDANCE_FILE):
    with open(ATTENDANCE_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Date", "Time"])

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form["name"]
        file = request.files["image"]

        if name and file:
            path = os.path.join(KNOWN_FACES_DIR, f"{name}.jpg")
            file.save(path)

        return redirect(url_for("index"))

    return render_template("register.html")

@app.route("/attendance")
def attendance():
    records = []
    with open(ATTENDANCE_FILE, "r") as f:
        reader = csv.reader(f)
        next(reader)
        records = list(reader)
    return render_template("attendance.html", records=records)

@app.route("/download")
def download():
    return send_file(ATTENDANCE_FILE, as_attachment=True)

@app.route("/start")
def start():
    known_encodings, known_names = load_known_faces(KNOWN_FACES_DIR)

    cap = cv2.VideoCapture(0)

    marked_today = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        names = recognize_faces(frame, known_encodings, known_names)

        for name in names:
            if name != "Unknown":
                today = datetime.now().strftime("%Y-%m-%d")
                if (name, today) not in marked_today:
                    time_now = datetime.now().strftime("%H:%M:%S")

                    with open(ATTENDANCE_FILE, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([name, today, time_now])

                    marked_today.add((name, today))

        cv2.imshow("Attendance", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)

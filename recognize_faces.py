import cv2
import face_recognition
import pickle
import os
import csv
from datetime import datetime

# Load encodings
encodings_path = "encodings.pickle"
if not os.path.exists(encodings_path):
    print("[ERROR] encodings.pickle not found! Run encode_faces.py first.")
    exit()

print("[INFO] Loading encodings...")
with open(encodings_path, "rb") as f:
    data = pickle.load(f)

# Prepare CSV file for attendance
date_str = datetime.now().strftime("%Y-%m-%d")
csv_file = f"attendance_{date_str}.csv"
attendance = {}

if not os.path.exists(csv_file):
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Time"])

print("[INFO] Starting camera...")
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb)
    encs = face_recognition.face_encodings(rgb, boxes)

    for (top, right, bottom, left), encoding in zip(boxes, encs):
        matches = face_recognition.compare_faces(data["encodings"], encoding)
        name = "Unknown"

        if True in matches:
            matched_idxs = [i for i, b in enumerate(matches) if b]
            counts = {}
            for i in matched_idxs:
                n = data["names"][i]
                counts[n] = counts.get(n, 0) + 1
            name = max(counts, key=counts.get)

            # Mark attendance if not already marked
            if name not in attendance:
                time_str = datetime.now().strftime("%H:%M:%S")
                with open(csv_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([name, time_str])
                attendance[name] = time_str
                print(f"[ATTENDANCE] {name} marked present at {time_str}")

        # Draw box + name on screen
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
print("[INFO] Attendance saved in", csv_file)

import os
import sys
import cv2
import pickle
import face_recognition

dataset_path = "dataset"
encodings_file = "encodings.pickle"

print("[INFO] Current working directory:", os.getcwd())
print("[INFO] Checking dataset path:", os.path.abspath(dataset_path))

if not os.path.exists(dataset_path):
    print("[ERROR] Dataset folder not found:", os.path.abspath(dataset_path))
    print("👉 Create a folder named 'dataset' and put student subfolders with images inside it.")
    sys.exit(1)

known_encodings = []
known_names = []
total_images = 0

for student in sorted(os.listdir(dataset_path)):
    student_folder = os.path.join(dataset_path, student)
    if not os.path.isdir(student_folder):
        continue
    print(f"[INFO] Processing student: {student}")
    imgs = sorted(os.listdir(student_folder))
    print(f"       Images found: {len(imgs)}")

    for image_file in imgs:
        image_path = os.path.join(student_folder, image_file)
        image = cv2.imread(image_path)
        if image is None:
            print("       [WARN] Could not read:", image_path)
            continue

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb, model="hog")
        encs = face_recognition.face_encodings(rgb, boxes)

        for e in encs:
            known_encodings.append(e)
            known_names.append(student)

        total_images += 1

print("[INFO] Total student folders processed:", len(set(known_names)))
print("[INFO] Total images scanned:", total_images)
print("[INFO] Total encodings collected:", len(known_encodings))

# Save encodings to pickle file
print("[INFO] Saving encodings to:", os.path.abspath(encodings_file))
with open(encodings_file, "wb") as f:
    pickle.dump({"encodings": known_encodings, "names": known_names}, f)

print("[DONE] encodings.pickle created successfully ✅")

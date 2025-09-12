import os
import cv2
import face_recognition
import pickle

# Path to dataset
dataset_path = "dataset"
encodings_file = "encodings.pickle"

known_encodings = []
known_names = []

print("[INFO] Encoding faces...")

# Loop through each student folder
for student in os.listdir(dataset_path):
    student_folder = os.path.join(dataset_path, student)
    if not os.path.isdir(student_folder):
        continue

    # Loop through images in student's folder
    for image_file in os.listdir(student_folder):
        image_path = os.path.join(student_folder, image_file)

        # Load image
        image = cv2.imread(image_path)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect faces
        boxes = face_recognition.face_locations(rgb, model="hog")
        encodings = face_recognition.face_encodings(rgb, boxes)

        for encoding in encodings:
            known_encodings.append(encoding)
            known_names.append(student)

#

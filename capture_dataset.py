import cv2
import os

# Ask student name
student_name = input("Enter student name: ")

# Path for saving images
dataset_path = "dataset"
student_path = os.path.join(dataset_path, student_name)

# Create folder if it doesn't exist
if not os.path.exists(student_path):
    os.makedirs(student_path)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Load face detector (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

count = 0
print("Press 'q' to quit capturing...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        count += 1
        face = frame[y:y+h, x:x+w]

        # Save the face image
        file_name = os.path.join(student_path, f"{student_name}_{count}.jpg")
        cv2.imwrite(file_name, face)

        # Draw rectangle on face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.putText(frame, f"Images: {count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Capturing Faces", frame)

    # Stop if 'q' pressed or 50 images captured
    if cv2.waitKey(1) & 0xFF == ord('q') or count >= 50:
        break

cap.release()
cv2.destroyAllWindows()

print(f"✅ Saved {count} images to {student_path}")

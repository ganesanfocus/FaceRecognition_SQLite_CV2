import cv2
import numpy as np
import os
import sqlite3
import time

facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("recognizer/trainingdata.yml")

# Cache for profiles -> avoid DB hit every frame
profile_cache = {}

def get_profile(id):
    # If already loaded, return from cache
    if id in profile_cache:
        return profile_cache[id]

    conn = sqlite3.connect("sqlite.db")
    cursor = conn.execute("SELECT * FROM STUDENT WHERE Id = ?", (id,))
    profile = None
    for row in cursor:
        profile = row
    conn.close()

    if profile is not None:
        profile_cache[id] = profile
    return profile

# For keeping box stable even if 1–2 frames miss detection
last_box = None          # (x, y, w, h)
last_profile = None
miss_frames = 0
MAX_MISS_FRAMES = 10     # show last box up to 10 frames after loss


while True:
    ret, img = cam.read()
    if not ret:
        break

    # OPTIONAL: make detection faster by resizing frame
    # comment these 3 lines if you don't want resize
    frame = cv2.resize(img, (0, 0), fx=0.7, fy=0.7)
    scale_x = img.shape[1] / frame.shape[1]
    scale_y = img.shape[0] / frame.shape[0]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = facedetect.detectMultiScale(
        gray,
        scaleFactor=1.3,   # you can try 1.2 or 1.3
        minNeighbors=5,
        minSize=(80, 80)   # ignore very small faces
    )

    if len(faces) > 0:
        # Take the first detected face (or loop over all if you want)
        (x, y, w, h) = faces[0]

        # Map coords back to original image size (because of resize)
        x_full = int(x * scale_x)
        y_full = int(y * scale_y)
        w_full = int(w * scale_x)
        h_full = int(h * scale_y)

        roi_gray = gray[y:y+h, x:x+w]
        id, conf = recognizer.predict(roi_gray)

        profile = get_profile(id)

        # Store as last seen
        last_box = (x_full, y_full, w_full, h_full)
        last_profile = profile
        miss_frames = 0

    else:
        # No face detected this frame
        if miss_frames < MAX_MISS_FRAMES:
            miss_frames += 1
        else:
            last_box = None
            last_profile = None

    # Draw box & text using last known detection
    if last_box is not None and last_profile is not None:
        x, y, w, h = last_box
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        name = str(last_profile[1])
        age = str(last_profile[2])

        cv2.putText(img, "Name: " + name, (x, y + h + 20),
                    cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 127), 2)
        cv2.putText(img, "Age: " + age, (x, y + h + 45),
                    cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 127), 2)

    cv2.imshow("FACE:", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()

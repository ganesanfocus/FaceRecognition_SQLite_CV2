# detector.py
import cv2
import numpy as np
import os
import sqlite3

# --- SSD model config ---
SSD_PROTO = "models/deploy.prototxt.txt"
SSD_MODEL = "models/res10_300x300_ssd_iter_140000.caffemodel"
CONF_THRESHOLD = 0.5

# --- Load SSD face detector ---

def load_ssd_face_detector():
    if not os.path.exists(SSD_PROTO) or not os.path.exists(SSD_MODEL):
        raise FileNotFoundError(
            f"SSD model files not found.\n"
            f"Expected:\n  {SSD_PROTO}\n  {SSD_MODEL}"
        )
    print("Loading SSD face detector...")
    net = cv2.dnn.readNetFromCaffe(SSD_PROTO, SSD_MODEL)
    print("SSD face detector loaded.")
    return net

def detect_faces_ssd(net, frame, conf_threshold=CONF_THRESHOLD):
    """
    Detect faces using SSD.
    Returns list of bounding boxes: [(x1, y1, x2, y2), ...]
    """
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)),
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0)
    )
    net.setInput(blob)
    detections = net.forward()

    boxes = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < conf_threshold:
            continue

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (x1, y1, x2, y2) = box.astype("int")

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w - 1, x2)
        y2 = min(h - 1, y2)

        boxes.append((x1, y1, x2, y2))

    return boxes

# --- DB lookup ---

def getprofile(id):
    conn = sqlite3.connect("database.db")
    cursor = conn.execute("SELECT * FROM STUDENTS WHERE Id = ?", (id,))
    profile = None
    for row in cursor:
        profile = row
    conn.close()
    return profile

# --- Main ---

def main():
    # Load SSD
    net = load_ssd_face_detector()

    # Load LBPH recognizer (already trained)
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("recognizer/trainingdata.yml")

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        ret, img = cam.read()
        if not ret:
            print("Error: Failed to read frame from camera.")
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # SSD detection
        boxes = detect_faces_ssd(net, img, CONF_THRESHOLD)

        for (x1, y1, x2, y2) in boxes:
            w = x2 - x1
            h = y2 - y1

            # Draw rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # ROI for recognition (grayscale)
            face_gray = gray[y1:y2, x1:x2]

            if face_gray.size == 0:
                continue

            id, conf = recognizer.predict(face_gray)
            profile = getprofile(id)
            print("ID:", id, "Conf:", conf, "Profile:", profile)

            THRESHOLD = 70  # lower = stricter, higher = more lenient

            if profile is not None and conf < THRESHOLD:
                name = str(profile[1])
                age = str(profile[2])

                cv2.putText(img, "Name: " + name,
                            (x1, y2 + 20),
                            cv2.FONT_HERSHEY_COMPLEX,
                            0.8, (0, 255, 127), 2)
                cv2.putText(img, "Age: " + age,
                            (x1, y2 + 45),
                            cv2.FONT_HERSHEY_COMPLEX,
                            0.8, (0, 255, 127), 2)
            else:
                cv2.putText(img, "Unknown",
                            (x1, y2 + 20),
                            cv2.FONT_HERSHEY_COMPLEX,
                            0.8, (0, 0, 255), 2)


        cv2.imshow("FACE (SSD + LBPH):", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

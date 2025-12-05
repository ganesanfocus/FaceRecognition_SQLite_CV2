# facenet_detector.py
import cv2
import numpy as np
import os
import sqlite3
from deepface import DeepFace

# ---- SSD model config ----
SSD_PROTO = "models/deploy.prototxt.txt"
SSD_MODEL = "models/res10_300x300_ssd_iter_140000.caffemodel"
CONF_THRESHOLD = 0.5

# ---- FaceNet embeddings path ----
EMBEDDINGS_PATH = "recognizer/facenet_embeddings.npz"

# ---- Load SSD face detector ----
def load_ssd_face_detector():
    if not os.path.exists(SSD_PROTO) or not os.path.exists(SSD_MODEL):
        raise FileNotFoundError(
            f"SSD model files not found.\nExpected:\n  {SSD_PROTO}\n  {SSD_MODEL}"
        )
    print("[INFO] Loading SSD face detector...")
    net = cv2.dnn.readNetFromCaffe(SSD_PROTO, SSD_MODEL)
    print("[INFO] SSD face detector loaded.")
    return net

def detect_faces_ssd(net, frame, conf_threshold=CONF_THRESHOLD):
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

# ---- DB lookup ----
def getprofile(id):
    conn = sqlite3.connect("database.db")
    cursor = conn.execute("SELECT * FROM STUDENTS WHERE Id = ?", (id,))
    profile = None
    for row in cursor:
        profile = row
    conn.close()
    return profile

# ---- Load embeddings ----
def load_embeddings():
    if not os.path.exists(EMBEDDINGS_PATH):
        raise FileNotFoundError(
            f"Embeddings file not found: {EMBEDDINGS_PATH}\n"
            f"Run facenet_trainer.py first."
        )
    data = np.load(EMBEDDINGS_PATH)
    embeddings = data["embeddings"]
    labels = data["labels"]
    print(f"[INFO] Loaded {len(labels)} embeddings from {EMBEDDINGS_PATH}")
    return embeddings, labels

# ---- Main ----
def main():
    # Load models
    net = load_ssd_face_detector()

    print("[INFO] Loading FaceNet model...")
    facenet_model = DeepFace.build_model("Facenet")
    print("[INFO] FaceNet model loaded.")

    embeddings_db, labels_db = load_embeddings()

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("[ERROR] Could not open camera.")
        return

    # Threshold for FaceNet distance (tune if needed)
    # Smaller = stricter, Larger = more lenient
    FACENET_THRESHOLD = 10.0  

    while True:
        ret, frame = cam.read()
        if not ret:
            print("[ERROR] Failed to read frame from camera.")
            break

        boxes = detect_faces_ssd(net, frame, CONF_THRESHOLD)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            face_rgb = frame_rgb[y1:y2, x1:x2]
            if face_rgb.size == 0:
                continue

            try:
                rep = DeepFace.represent(
                img_path=face_rgb,
                model_name="Facenet",
                detector_backend="skip",
                enforce_detection=False
                )

                if isinstance(rep, list):
                    emb = np.array(rep[0]["embedding"], dtype="float32")
                elif isinstance(rep, dict):
                    emb = np.array(rep["embedding"], dtype="float32")
                else:
                    raise ValueError("Unexpected representation format")

            except Exception as e:
                print(f"[WARN] Failed to compute embedding for face: {e}")
                continue

            # Compute distances to all stored embeddings
            diffs = embeddings_db - emb
            dists = np.linalg.norm(diffs, axis=1)
            min_idx = np.argmin(dists)
            min_dist = dists[min_idx]
            predicted_id = int(labels_db[min_idx])

            print(f"Predicted ID: {predicted_id}, distance: {min_dist:.3f}")

            if min_dist < FACENET_THRESHOLD:
                profile = getprofile(predicted_id)
                if profile is not None:
                    name = str(profile[1])
                    age = str(profile[2])

                    cv2.putText(frame, f"Name: {name}",
                                (x1, y2 + 20),
                                cv2.FONT_HERSHEY_COMPLEX,
                                0.8, (0, 255, 127), 2)
                    cv2.putText(frame, f"Age: {age}",
                                (x1, y2 + 45),
                                cv2.FONT_HERSHEY_COMPLEX,
                                0.8, (0, 255, 127), 2)
                else:
                    cv2.putText(frame, "Unknown (no DB record)",
                                (x1, y2 + 20),
                                cv2.FONT_HERSHEY_COMPLEX,
                                0.8, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "Unknown",
                            (x1, y2 + 20),
                            cv2.FONT_HERSHEY_COMPLEX,
                            0.8, (0, 0, 255), 2)

        cv2.imshow("FaceNet + SSD Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

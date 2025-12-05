import cv2
import numpy as np
import os
import sqlite3
from deepface import DeepFace

# ---------- CONFIG ----------
SSD_PROTO = "models/deploy.prototxt.txt"
SSD_MODEL = "models/res10_300x300_ssd_iter_140000.caffemodel"
CONF_THRESHOLD = 0.5

EMBEDDINGS_PATH = "recognizer/facenet_embeddings.npz"
FACENET_THRESHOLD = 10.0      # ID match threshold (smaller = stricter)

MOTION_THRESHOLD = 2.0        # avg pixel diff; smaller = more sensitive
STATIC_FRAMES_LIMIT = 20      # number of low-motion frames to call it "spoof"

# ---------- MODELS ----------
def load_ssd():
    net = cv2.dnn.readNetFromCaffe(SSD_PROTO, SSD_MODEL)
    return net

def detect_faces_ssd(net, frame):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0)
    )
    net.setInput(blob)
    det = net.forward()

    boxes = []
    for i in range(det.shape[2]):
        conf = det[0, 0, i, 2]
        if conf > CONF_THRESHOLD:
            box = det[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype(int)
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(w - 1, x2); y2 = min(h - 1, y2)
            boxes.append((x1, y1, x2, y2))
    return boxes

def load_embeddings():
    data = np.load(EMBEDDINGS_PATH)
    return data["embeddings"], data["labels"]

def get_profile(pid):
    conn = sqlite3.connect("database.db")
    cur = conn.execute("SELECT * FROM STUDENTS WHERE id=?", (pid,))
    row = cur.fetchone()
    conn.close()
    return row

# ---------- MAIN ----------
def main():
    net = load_ssd()
    db_embeddings, db_labels = load_embeddings()

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Could not open camera")
        return

    prev_face_gray = None
    static_frames = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        boxes = detect_faces_ssd(net, frame)

        if len(boxes) > 0:
            # assume single main face (first box)
            (x1, y1, x2, y2) = boxes[0]
            w = x2 - x1
            h = y2 - y1

            face_gray = gray[y1:y2, x1:x2]
            face_rgb = rgb[y1:y2, x1:x2]

            # ---------- MOTION-BASED LIVENESS ----------
            if face_gray.size != 0:
                face_resized = cv2.resize(face_gray, (100, 100))
                if prev_face_gray is not None and prev_face_gray.shape == face_resized.shape:
                    diff = np.mean(np.abs(face_resized.astype("float32") -
                                          prev_face_gray.astype("float32")))
                    if diff < MOTION_THRESHOLD:
                        static_frames += 1
                    else:
                        static_frames = 0
                prev_face_gray = face_resized.copy()
            else:
                static_frames = 0
                prev_face_gray = None

            # if face is almost not moving for many frames -> treat as spoof/photo
            if static_frames >= STATIC_FRAMES_LIMIT:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, "Spoof / Photo detected",
                            (x1, y2 + 20), cv2.FONT_HERSHEY_COMPLEX,
                            0.7, (0, 0, 255), 2)
                cv2.imshow("FaceNet + SSD + Liveness", frame)
                if cv2.waitKey(1) == ord('q'):
                    break
                continue

            # ---------- FACENET RECOGNITION ----------
            if face_rgb.size == 0:
                cv2.imshow("FaceNet + SSD + Liveness", frame)
                if cv2.waitKey(1) == ord('q'):
                    break
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
                else:
                    emb = np.array(rep["embedding"], dtype="float32")
            except Exception as e:
                print("FaceNet error:", e)
                cv2.imshow("FaceNet + SSD + Liveness", frame)
                if cv2.waitKey(1) == ord('q'):
                    break
                continue

            dists = np.linalg.norm(db_embeddings - emb, axis=1)
            idx = np.argmin(dists)
            dist = dists[idx]
            pid = int(db_labels[idx])

            if dist < FACENET_THRESHOLD:
                profile = get_profile(pid)
                if profile:
                    name = profile[1]
                    age = profile[2]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"Name: {name}",
                                (x1, y2 + 20),
                                cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Age: {age}",
                                (x1, y2 + 45),
                                cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(frame, "Unknown",
                            (x1, y2 + 20), cv2.FONT_HERSHEY_COMPLEX,
                            0.7, (0, 255, 255), 2)

        cv2.imshow("FaceNet + SSD + Liveness", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

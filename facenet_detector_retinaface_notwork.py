import cv2
import numpy as np
import os
import sqlite3
from deepface import DeepFace

# ---- SSD model config ----
SSD_PROTO = "models/deploy.prototxt.txt"
SSD_MODEL = "models/res10_300x300_ssd_iter_140000.caffemodel"
CONF_THRESHOLD = 0.5

# ---- FaceNet embeddings ----
EMBEDDINGS_PATH = "recognizer/facenet_embeddings.npz"

# ---- Load SSD ----
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
            boxes.append((x1, y1, x2, y2))
    return boxes

# ---- DB ----
def get_profile(i):
    conn = sqlite3.connect("database.db")
    cur = conn.execute("SELECT * FROM STUDENTS WHERE id=?", (i,))
    row = cur.fetchone()
    conn.close()
    return row

# ---- Load FaceNet embeddings ----
def load_embeddings():
    data = np.load(EMBEDDINGS_PATH)
    return data["embeddings"], data["labels"]

# ---- IOU helper ----
def iou(boxA, boxB):
    (x1, y1, x2, y2) = boxA
    xA = max(x1, boxB[0])
    yA = max(y1, boxB[1])
    xB = min(x2, boxB[0] + boxB[2])
    yB = min(y2, boxB[1] + boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (x2 - x1) * (y2 - y1)
    areaB = boxB[2] * boxB[3]
    union = areaA + areaB - inter
    if union == 0:
        return 0
    return inter / union

# ---- MAIN ----
def main():
    net = load_ssd()
    facenet_embeddings, labels = load_embeddings()

    cam = cv2.VideoCapture(0)
    FACENET_THRESHOLD = 10.0

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ssd_boxes = detect_faces_ssd(net, frame)

        for (x1, y1, x2, y2) in ssd_boxes:

            # -------- ANTI-SPOOFING (RETINAFACE) --------
            try:
                face_objs = DeepFace.extract_faces(
                    img_path=frame,
                    detector_backend="retinaface",
                    anti_spoofing=True,
                    enforce_detection=False
                )

                best = None
                best_iou = 0

                for fo in face_objs:
                    fa = fo.get("facial_area", {})
                    bx = (fa.get("x", 0), fa.get("y", 0),
                          fa.get("w", 0), fa.get("h", 0))
                    score = iou((x1, y1, x2, y2), bx)
                    if score > best_iou:
                        best_iou = score
                        best = fo

                if best is None:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, "Spoof detected", (x1, y2 + 20),
                                cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2)
                    continue

                is_real = best.get("is_real", False)

                if not is_real:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, "Spoof detected", (x1, y2 + 20),
                                cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2)
                    continue

            except:
                continue

            # -------- FACENET RECOGNITION --------
            face_rgb = rgb[y1:y2, x1:x2]
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
                else:
                    emb = np.array(rep["embedding"], dtype="float32")
            except:
                continue

            dists = np.linalg.norm(facenet_embeddings - emb, axis=1)
            idx = np.argmin(dists)
            dist = dists[idx]
            pid = int(labels[idx])

            if dist < FACENET_THRESHOLD:
                profile = get_profile(pid)
                if profile:
                    name = profile[1]
                    age = profile[2]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"Name: {name}", (x1, y2 + 20),
                                cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.putText(frame, f"Age: {age}", (x1, y2 + 45),
                                cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(frame, "Unknown", (x1, y2 + 20),
                            cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("FaceNet + SSD + AntiSpoof", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

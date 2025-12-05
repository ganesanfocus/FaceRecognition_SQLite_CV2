import cv2
import numpy as np
import os
import sqlite3
from deepface import DeepFace
from db_model import OrderDatabase

# ------------- CONFIG -------------
SSD_PROTO = "models/deploy.prototxt.txt"
SSD_MODEL = "models/res10_300x300_ssd_iter_140000.caffemodel"
CONF_THRESHOLD = 0.5

EMBEDDINGS_PATH = "recognizer/facenet_embeddings.npz"
STUDENT_DB_PATH = "database.db"
FACENET_THRESHOLD = 10.0  # smaller = stricter match


# ------------- MODELS -------------
def load_ssd():
    if not (os.path.exists(SSD_PROTO) and os.path.exists(SSD_MODEL)):
        raise FileNotFoundError(
            f"Missing SSD model files:\n{SSD_PROTO}\n{SSD_MODEL}"
        )
    net = cv2.dnn.readNetFromCaffe(SSD_PROTO, SSD_MODEL)
    return net


def detect_faces_ssd(net, frame):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)),
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0),
    )
    net.setInput(blob)
    detections = net.forward()

    boxes = []
    for i in range(detections.shape[2]):
        conf = detections[0, 0, i, 2]
        if conf < CONF_THRESHOLD:
            continue

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (x1, y1, x2, y2) = box.astype("int")

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w - 1, x2)
        y2 = min(h - 1, y2)

        boxes.append((x1, y1, x2, y2))
    return boxes


def load_embeddings():
    if not os.path.exists(EMBEDDINGS_PATH):
        raise FileNotFoundError(
            f"Embeddings file not found: {EMBEDDINGS_PATH}. Run facenet_trainer.py first."
        )
    data = np.load(EMBEDDINGS_PATH)
    embeddings = data["embeddings"]
    labels = data["labels"]
    return embeddings, labels


def get_profile(pid):
    conn = sqlite3.connect(STUDENT_DB_PATH)
    cur = conn.execute("SELECT * FROM STUDENTS WHERE id = ?", (pid,))
    row = cur.fetchone()
    conn.close()
    return row


# ------------- MAIN -------------
def main():
    net = load_ssd()
    db_embeddings, db_labels = load_embeddings()

    # Orders DB
    orders_db = OrderDatabase()

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Error: Cannot open camera")
        return

    last_shown_id = None  # to avoid printing orders every frame

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Error: Failed to read frame from camera")
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        boxes = detect_faces_ssd(net, frame)

        for (x1, y1, x2, y2) in boxes:
            face_rgb = rgb[y1:y2, x1:x2]

            if face_rgb.size == 0:
                continue

            try:
                rep = DeepFace.represent(
                    img_path=face_rgb,
                    model_name="Facenet",
                    detector_backend="skip",
                    enforce_detection=False,
                )

                if isinstance(rep, list):
                    emb = np.array(rep[0]["embedding"], dtype="float32")
                elif isinstance(rep, dict):
                    emb = np.array(rep["embedding"], dtype="float32")
                else:
                    continue
            except Exception as e:
                print("FaceNet error:", e)
                continue

            dists = np.linalg.norm(db_embeddings - emb, axis=1)
            idx = np.argmin(dists)
            dist = dists[idx]
            pid = int(db_labels[idx])

            if dist < FACENET_THRESHOLD:
                profile = get_profile(pid)
                if profile is not None:
                    name = str(profile[1])
                    age = str(profile[2])

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"Name: {name}",
                        (x1, y2 + 20),
                        cv2.FONT_HERSHEY_COMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                    )
                    cv2.putText(
                        frame,
                        f"Age: {age}",
                        (x1, y2 + 45),
                        cv2.FONT_HERSHEY_COMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                    )

                    # -------- DEMO: SHOW ALL ORDERS ONCE WHEN FACE IDENTIFIED --------
                    if last_shown_id != pid:
                        last_shown_id = pid
                        print("\n==============================")
                        print(f"Face recognized: ID={pid}, Name={name}")
                        print("Fetching ALL orders from online_sales.db ...")
                        print("==============================")
                        orders = orders_db.get_all_orders()
                        for o in orders:
                            print(
                                f"OrderID: {o[0]} | Cust: {o[1]} | Phone: {o[2]} | "
                                f"OrderNo: {o[3]} | Amount: {o[4]} | PayStatus: {o[5]} | "
                                f"OrderStatus: {o[6]} | Delivery: {o[7]} | "
                                f"Expected: {o[8]} | DelayReason: {o[9]}"
                            )
                        print("==============================\n")

            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(
                    frame,
                    "Unknown",
                    (x1, y2 + 20),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )

        cv2.imshow("FaceNet + SSD Face Recognition + Orders Demo", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

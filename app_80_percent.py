# app.py - FIXED AND IMPROVED VERSION

import cv2
import numpy as np
import os
import sqlite3
from deepface import DeepFace
from flask import Flask, render_template, Response, jsonify, request
from db_model import OrderDatabase
from datetime import datetime, timedelta
import random
import base64

# ---------------- CONFIG ----------------
SSD_PROTO = "models/deploy.prototxt.txt"
SSD_MODEL = "models/res10_300x300_ssd_iter_140000.caffemodel"
CONF_THRESHOLD = 0.5

EMBEDDINGS_PATH = "recognizer/facenet_embeddings.npz"
STUDENT_DB_PATH = "database.db"

# Use L2-normalized Euclidean distance (recommended for FaceNet)
FACENET_THRESHOLD = 0.92  # Good starting point: lower = stricter. Tune between 0.8–1.0
USE_FACENET512 = False    # Set to True for much better accuracy (threshold ~1.04)

MODEL_NAME = "Facenet512" if USE_FACENET512 else "Facenet"

# ---------------- INIT MODELS ----------------
def load_ssd():
    if not (os.path.exists(SSD_PROTO) and os.path.exists(SSD_MODEL)):
        raise FileNotFoundError(f"Missing SSD model files:\n{SSD_PROTO}\n{SSD_MODEL}")
    net = cv2.dnn.readNetFromCaffe(SSD_PROTO, SSD_MODEL)
    return net

def detect_faces_ssd(net, frame):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    boxes = []
    for i in range(detections.shape[2]):
        conf = detections[0, 0, i, 2]
        if conf < CONF_THRESHOLD:
            continue
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (x1, y1, x2, y2) = box.astype("int")
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        boxes.append((x1, y1, x2, y2))
    return boxes

def load_embeddings():
    if not os.path.exists(EMBEDDINGS_PATH):
        raise FileNotFoundError(f"Embeddings file not found: {EMBEDDINGS_PATH}. Run facenet_trainer.py first.")
    data = np.load(EMBEDDINGS_PATH)
    embeddings = data["embeddings"].astype("float32")
    labels = data["labels"]

    # Pre-normalize all embeddings (critical for stable distance)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized_embeddings = embeddings / norms

    print(f"[INFO] Loaded {len(labels)} embeddings for {len(np.unique(labels))} customers")
    return normalized_embeddings, labels

def get_profile(pid):
    conn = sqlite3.connect(STUDENT_DB_PATH)
    cur = conn.execute("SELECT * FROM STUDENTS WHERE id = ?", (pid,))
    row = cur.fetchone()
    conn.close()
    return row

def create_sample_orders_for_customer(customer_id, customer_name, phone_number):
    orders_db = OrderDatabase()
    orders_db.connect()
    
    num_orders = random.randint(2, 3)
    base_date = datetime.now()
    
    payment_statuses = ["Paid", "Pending"]
    order_statuses = ["Processing", "Shipped", "Delivered", "Delayed"]
    delay_reasons = ["Weather conditions", "Out of stock", "Courier delay"]
    
    created_orders = []
    
    for i in range(num_orders):
        order_num = f"ORD{base_date.year}{base_date.month:02d}{customer_id:04d}{i+1:02d}"
        amount = round(random.uniform(75, 450), 2)
        payment = random.choice(payment_statuses)
        status = random.choice(order_statuses)
        
        expected_delivery = (base_date + timedelta(days=random.randint(3, 7))).strftime("%Y-%m-%d")
        
        if status == "Delivered":
            delivery = (base_date - timedelta(days=random.randint(1, 3))).strftime("%Y-%m-%d")
            delay = None
        elif status == "Delayed":
            delivery = None
            delay = random.choice(delay_reasons)
        else:
            delivery = None
            delay = None
        
        orders_db.cursor.execute('''
            INSERT INTO orders (cust_name, phone, order_number, amount, payment_status, 
                              order_status, delivery_date, expected_delivery_date, delay_reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (customer_name, phone_number, order_num, amount, payment, status,
              delivery, expected_delivery, delay))
        
        created_orders.append(order_num)
        print(f"  ✅ Created order: {order_num} - ${amount:.2f} - {status}")
    
    orders_db.conn.commit()
    orders_db.close()
    
    print(f"🎉 Created {num_orders} sample orders for {customer_name}")
    return created_orders

def train_model():
    print("[INFO] Training FaceNet embeddings...")
    from deepface import DeepFace
    
    dataset_dir = "dataset"
    image_paths = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir)
                   if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    
    if not image_paths:
        raise Exception("No images found in dataset folder")
    
    embeddings = []
    labels = []
    
    for image_path in image_paths:
        filename = os.path.basename(image_path)
        try:
            person_id = int(filename.split(".")[1])
        except:
            print(f"[WARN] Skipping bad filename: {filename}")
            continue
        
        img = cv2.imread(image_path)
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        try:
            rep = DeepFace.represent(
                img_path=img_rgb,
                model_name=MODEL_NAME,
                detector_backend="skip",
                align=True,
                enforce_detection=False
            )
            emb = rep[0]["embedding"] if isinstance(rep, list) else rep["embedding"]
            emb = np.array(emb, dtype="float32")
            
            # Normalize immediately
            emb = emb / np.linalg.norm(emb)
            
            embeddings.append(emb)
            labels.append(person_id)
            print(f"[INFO] Processed {filename} -> ID {person_id}")
        except Exception as e:
            print(f"[WARN] Embedding failed for {filename}: {e}")
            continue
    
    if not embeddings:
        raise Exception("No valid embeddings generated")
    
    embeddings_array = np.vstack(embeddings)
    labels_array = np.array(labels, dtype="int32")
    
    os.makedirs(os.path.dirname(EMBEDDINGS_PATH), exist_ok=True)
    np.savez(EMBEDDINGS_PATH, embeddings=embeddings_array, labels=labels_array)
    
    print(f"[INFO] Saved {len(labels_array)} normalized embeddings to {EMBEDDINGS_PATH}")
    
    # Reload in app
    global db_embeddings, db_labels
    db_embeddings, db_labels = load_embeddings()

# Load models
ssd_net = load_ssd()
db_embeddings, db_labels = load_embeddings()
orders_db = OrderDatabase()

# Shared state
current_face_id = None
current_face_name = None
current_orders = []
camera_active = False
camera_start_time = None
recognition_status = "idle"

# Registration state
registration_mode = False
registration_id = None
registration_name = None
registration_age = None
registration_phone = None
captured_samples = 0
MAX_SAMPLES = 30

# ---------------- FLASK APP ----------------
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/check_face_during_registration", methods=["POST"])
def check_face_during_registration():
    try:
        data = request.json
        image_data = data.get("image")
        if not image_data:
            return jsonify({"status": "error", "message": "No image"})

        image_bytes = base64.b64decode(image_data.split(',')[1])
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes = detect_faces_ssd(ssd_net, frame)

        if not boxes:
            return jsonify({"status": "no_face", "message": "No face detected"})

        (x1, y1, x2, y2) = boxes[0]
        face_rgb = rgb[y1:y2, x1:x2]
        if face_rgb.size == 0:
            return jsonify({"status": "no_face"})

        rep = DeepFace.represent(
            img_path=face_rgb,
            model_name=MODEL_NAME,
            detector_backend="skip",
            align=True,
            enforce_detection=False
        )
        emb = rep[0]["embedding"] if isinstance(rep, list) else rep["embedding"]
        emb = np.array(emb, dtype="float32")
        emb_norm = emb / np.linalg.norm(emb)

        if len(db_embeddings) == 0:
            return jsonify({"status": "new_face", "message": "No registered users yet"})

        # Compute L2 distances to all DB embeddings (already normalized)
        dists = np.linalg.norm(db_embeddings - emb_norm, axis=1)

        unique_customers = np.unique(db_labels)
        customer_matches = []
        for cid in unique_customers:
            mask = db_labels == cid
            best_dist = np.min(dists[mask])
            customer_matches.append({"customer_id": int(cid), "best_distance": float(best_dist)})

        customer_matches.sort(key=lambda x: x['best_distance'])
        closest = customer_matches[0]

        print(f"\n[REG CHECK] Closest: ID {closest['customer_id']} - Dist {closest['best_distance']:.3f}")

        if closest['best_distance'] < FACENET_THRESHOLD:
            profile = get_profile(closest['customer_id'])
            if profile:
                name = profile[1]
                confidence = max(0, min(100, (1 - closest['best_distance'] / 1.5) * 100))
                return jsonify({
                    "status": "already_registered",
                    "customer_id": closest['customer_id'],
                    "customer_name": name,
                    "confidence": round(confidence, 1),
                    "distance": round(closest['best_distance'], 3)
                })

        return jsonify({
            "status": "new_face",
            "closest_distance": round(closest['best_distance'], 3),
            "threshold": FACENET_THRESHOLD
        })

    except Exception as e:
        print(f"Error in check: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)})

@app.route("/start_registration", methods=["POST"])
def start_registration():
    global registration_mode, camera_active, recognition_status
    global registration_id, registration_name, registration_age, registration_phone, captured_samples

    try:
        data = request.json
        registration_name = data.get("name")
        registration_age = data.get("age")
        phone = data.get("phone") or f"555-{random.randint(1000,9999)}"

        if not registration_name:
            return jsonify({"status": "error", "message": "Name required"})

        # Auto ID
        conn = sqlite3.connect(STUDENT_DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT MAX(id) FROM STUDENTS")
        max_id = cur.fetchone()[0]
        registration_id = (max_id or 0) + 1
        conn.close()

        # Insert to DB
        conn = sqlite3.connect(STUDENT_DB_PATH)
        conn.execute("INSERT INTO STUDENTS (id, Name, age) VALUES (?, ?, ?)",
                     (registration_id, registration_name, registration_age))
        conn.commit()
        conn.close()

        registration_phone = phone
        registration_mode = True
        camera_active = True
        captured_samples = 0
        recognition_status = "registering"

        os.makedirs("dataset", exist_ok=True)

        print(f"Started registration for {registration_name} (ID: {registration_id})")
        return jsonify({"status": "started", "customer_id": registration_id})

    except Exception as e:
        print(f"Reg error: {e}")
        return jsonify({"status": "error", "message": str(e)})

@app.route("/registration_status")
def registration_status():
    return jsonify({
        "active": registration_mode,
        "samples": captured_samples,
        "max_samples": MAX_SAMPLES,
        "status": recognition_status
    })

@app.route("/start_camera")
def start_camera():
    global camera_active, camera_start_time, recognition_status
    camera_active = True
    camera_start_time = cv2.getTickCount() / cv2.getTickFrequency()
    recognition_status = "scanning"
    print("Camera started - scanning...")
    return jsonify({"status": "started"})

@app.route("/stop_camera")
def stop_camera():
    global camera_active, recognition_status
    camera_active = False
    recognition_status = "idle"
    return jsonify({"status": "stopped"})

@app.route("/reload_embeddings")
def reload_embeddings():
    global db_embeddings, db_labels
    try:
        db_embeddings, db_labels = load_embeddings()
        return jsonify({"status": "success", "customers": len(np.unique(db_labels))})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route("/camera_status")
def camera_status():
    return jsonify({
        "active": camera_active,
        "status": recognition_status,
        "face_id": current_face_id,
        "name": current_face_name
    })

def gen_frames():
    global current_face_id, current_face_name, current_orders, camera_active, recognition_status
    global registration_mode, captured_samples, registration_id, registration_name

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Cannot open camera")
        return

    try:
        while camera_active:
            success, frame = cam.read()
            if not success:
                break

            if registration_mode:
                boxes = detect_faces_ssd(ssd_net, frame)
                for (x1, y1, x2, y2) in boxes:
                    if captured_samples >= MAX_SAMPLES:
                        break
                    captured_samples += 1
                    face = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
                    filename = f"dataset/user.{registration_id}.{captured_samples}.jpg"
                    cv2.imwrite(filename, face)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, f"Capturing {captured_samples}/{MAX_SAMPLES}",
                                (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

                if captured_samples >= MAX_SAMPLES:
                    print(f"Registration complete for {registration_name}")
                    recognition_status = "registration_complete"
                    camera_active = False
                    registration_mode = False

                    try:
                        train_model()
                        create_sample_orders_for_customer(registration_id, registration_name, registration_phone)
                    except Exception as e:
                        print(f"Post-reg error: {e}")
                    break

            else:
                current_time = cv2.getTickCount() / cv2.getTickFrequency()
                elapsed = current_time - camera_start_time
                if elapsed > 30 and recognition_status == "scanning":
                    recognition_status = "timeout"
                    camera_active = False
                    print("Timeout")
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                boxes = detect_faces_ssd(ssd_net, frame)

                for (x1, y1, x2, y2) in boxes:
                    face_rgb = rgb[y1:y2, x1:x2]
                    if face_rgb.size == 0:
                        continue

                    try:
                        rep = DeepFace.represent(
                            img_path=face_rgb,
                            model_name=MODEL_NAME,
                            detector_backend="skip",
                            align=True,
                            enforce_detection=False
                        )
                        emb = rep[0]["embedding"] if isinstance(rep, list) else rep["embedding"]
                        emb = np.array(emb, dtype="float32")
                        emb_norm = emb / np.linalg.norm(emb)
                    except:
                        continue

                    dists = np.linalg.norm(db_embeddings - emb_norm, axis=1)

                    unique_customers = np.unique(db_labels)
                    customer_matches = []
                    for cid in unique_customers:
                        mask = db_labels == cid
                        best_dist = np.min(dists[mask])
                        customer_matches.append({"id": int(cid), "dist": float(best_dist)})

                    customer_matches.sort(key=lambda x: x['dist'])
                    best = customer_matches[0]

                    print(f"[RECOG] Best: ID {best['id']} - Dist {best['dist']:.3f}")

                    if best['dist'] < FACENET_THRESHOLD:
                        profile = get_profile(best['id'])
                        if profile:
                            name = profile[1]
                            current_face_id = best['id']
                            current_face_name = name
                    
                            current_orders = orders_db.get_orders_by_customer_name(name)
                            recognition_status = "recognized"

                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                            cv2.putText(frame, f"{name} ({best['dist']:.2f})", (x1, y1-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                            print(f"MATCH: {name} (ID: {best['id']})")
                            camera_active = False
                            break
                    else:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                        cv2.putText(frame, f"Unknown ({best['dist']:.2f})", (x1, y1-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

                time_left = int(30 - elapsed)
                cv2.putText(frame, f"Time: {time_left}s", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    finally:
        cam.release()

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/current_orders")
def current_orders_api():
    orders_list = []
    if current_face_id and current_face_name:
        customer_orders = orders_db.get_orders_by_customer_name(current_face_name)
        for o in customer_orders:
            orders_list.append({
                "order_id": o[0], "cust_name": o[1], "phone": o[2],
                "order_number": o[3], "amount": o[4], "payment_status": o[5],
                "order_status": o[6], "delivery_date": o[7],
                "expected_delivery_date": o[8], "delay_reason": o[9]
            })

    return jsonify({
        "face_id": current_face_id,
        "name": current_face_name,
        "orders": orders_list,
        "status": recognition_status
    })

if __name__ == "__main__":
    app.run(debug=True)
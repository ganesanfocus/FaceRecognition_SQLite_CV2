import cv2
import numpy as np
import os
import sqlite3
from deepface import DeepFace
from flask import Flask, render_template, Response, jsonify
from db_model import OrderDatabase

# ---------------- CONFIG ----------------
SSD_PROTO = "models/deploy.prototxt.txt"
SSD_MODEL = "models/res10_300x300_ssd_iter_140000.caffemodel"
CONF_THRESHOLD = 0.5

EMBEDDINGS_PATH = "recognizer/facenet_embeddings.npz"
STUDENT_DB_PATH = "database.db"
FACENET_THRESHOLD = 2.5  # Updated based on diagnostic: same-person max=2.37, diff-person min=2.06

# ---------------- INIT MODELS ----------------
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


def train_model():
    """Train FaceNet model with all images in dataset folder"""
    from deepface import DeepFace
    
    print("[INFO] Preparing FaceNet embeddings using DeepFace...")
    
    # Get all image files
    dataset_dir = "dataset"
    image_paths = [
        os.path.join(dataset_dir, f)
        for f in os.listdir(dataset_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    
    if not image_paths:
        raise Exception("No images found in dataset folder")
    
    embeddings = []
    labels = []
    
    for image_path in image_paths:
        filename = os.path.basename(image_path)
        
        # Expecting filename: user.<id>.<sample>.jpg
        try:
            id_str = filename.split(".")[1]
            person_id = int(id_str)
        except (IndexError, ValueError):
            print(f"[WARN] Skipping file with bad format: {filename}")
            continue
        
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            print(f"[WARN] Could not read image: {image_path}")
            continue
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        try:
            rep = DeepFace.represent(
                img_path=img_rgb,
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
            print(f"[WARN] Failed to get embedding for {filename}: {e}")
            continue
        
        embeddings.append(emb)
        labels.append(person_id)
        print(f"[INFO] Processed {filename} -> ID {person_id}")
    
    if not embeddings:
        raise Exception("No embeddings were created. Check your dataset.")
    
    embeddings_array = np.vstack(embeddings)
    labels_array = np.array(labels, dtype="int32")
    
    # Save embeddings
    os.makedirs(os.path.dirname(EMBEDDINGS_PATH), exist_ok=True)
    np.savez(EMBEDDINGS_PATH, embeddings=embeddings_array, labels=labels_array)
    
    print(f"[INFO] Saved embeddings to {EMBEDDINGS_PATH}")
    print(f"[INFO] Total embeddings: {len(labels_array)}")
    
    # Reload embeddings in the app
    global db_embeddings, db_labels
    db_embeddings = embeddings_array
    db_labels = labels_array
    
    unique_ids = np.unique(labels_array)
    print(f"[INFO] Loaded {len(labels_array)} embeddings for {len(unique_ids)} customers")
    print(f"[INFO] Customer IDs: {sorted(unique_ids.tolist())}")


# Load once
ssd_net = load_ssd()
db_embeddings, db_labels = load_embeddings()
orders_db = OrderDatabase()
# DON'T keep camera open globally - open/close as needed

# Shared state for web polling
current_face_id = None
current_face_name = None
current_orders = []
camera_active = False
camera_start_time = None
recognition_status = "idle"  # idle, scanning, recognized, timeout

# Registration state
registration_mode = False
registration_id = None
registration_name = None
registration_age = None
captured_samples = 0
MAX_SAMPLES = 30

# ---------------- FLASK APP ----------------
app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/start_registration", methods=["POST"])
def start_registration():
    global registration_mode, camera_active, camera_start_time, recognition_status
    global registration_id, registration_name, registration_age, captured_samples
    
    from flask import request
    
    try:
        data = request.json
        print(f"📝 Registration request received: {data}")
        
        registration_id = data.get("id")
        registration_name = data.get("name")
        registration_age = data.get("age") or None
        
        if not registration_id or not registration_name:
            print("❌ Missing required fields")
            return jsonify({"status": "error", "message": "ID and Name are required"})
        
        # Insert/Update in database
        try:
            conn = sqlite3.connect(STUDENT_DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM STUDENTS WHERE id = ?", (registration_id,))
            exists = cursor.fetchone() is not None
            
            if exists:
                print(f"📝 Updating existing customer ID: {registration_id}")
                conn.execute("UPDATE STUDENTS SET Name=?, age=? WHERE id=?", 
                            (registration_name, registration_age, registration_id))
            else:
                print(f"📝 Creating new customer ID: {registration_id}")
                conn.execute("INSERT INTO STUDENTS (id, Name, age) VALUES (?, ?, ?)", 
                            (registration_id, registration_name, registration_age))
            conn.commit()
            conn.close()
            print(f"✅ Database updated successfully")
        except Exception as e:
            print(f"❌ Database error: {e}")
            return jsonify({"status": "error", "message": f"Database error: {str(e)}"})
        
        # Start registration mode
        registration_mode = True
        camera_active = True
        camera_start_time = cv2.getTickCount() / cv2.getTickFrequency()
        recognition_status = "registering"
        captured_samples = 0
        
        # Create dataset directory if not exists
        os.makedirs("dataset", exist_ok=True)
        
        print(f"📹 Registration started for {registration_name} (ID: {registration_id})")
        return jsonify({"status": "started"})
    
    except Exception as e:
        print(f"❌ Registration error: {e}")
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
    global camera_active, camera_start_time, current_face_id, current_face_name, current_orders, recognition_status
    camera_active = True
    camera_start_time = cv2.getTickCount() / cv2.getTickFrequency()
    recognition_status = "scanning"
    current_face_id = None
    current_face_name = None
    current_orders = []
    print("📹 Camera started - scanning for face...")
    return jsonify({"status": "started"})


@app.route("/stop_camera")
def stop_camera():
    global camera_active, recognition_status
    camera_active = False
    recognition_status = "idle"
    print("🛑 Camera stopped manually")
    return jsonify({"status": "stopped"})


@app.route("/reload_embeddings")
def reload_embeddings():
    global db_embeddings, db_labels
    try:
        db_embeddings, db_labels = load_embeddings()
        unique_ids = np.unique(db_labels)
        print(f"🔄 Embeddings reloaded: {len(db_labels)} samples for {len(unique_ids)} customers")
        print(f"   Customer IDs: {sorted(unique_ids.tolist())}")
        return jsonify({
            "status": "success", 
            "total_samples": len(db_labels),
            "unique_customers": len(unique_ids),
            "customer_ids": sorted(unique_ids.tolist())
        })
    except Exception as e:
        print(f"❌ Failed to reload embeddings: {e}")
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
    global current_face_id, current_face_name, current_orders, camera_active, camera_start_time, recognition_status
    global registration_mode, registration_id, registration_name, captured_samples, MAX_SAMPLES

    # Open camera only when starting
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("❌ Failed to open camera")
        return

    try:
        while camera_active:
            success, frame = cam.read()
            if not success:
                break

            # REGISTRATION MODE
            if registration_mode:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                boxes = detect_faces_ssd(ssd_net, frame)
                
                for (x1, y1, x2, y2) in boxes:
                    if captured_samples >= MAX_SAMPLES:
                        break
                    
                    captured_samples += 1
                    
                    # Save face image
                    face_gray = gray[y1:y2, x1:x2]
                    filename = f"dataset/user.{registration_id}.{captured_samples}.jpg"
                    cv2.imwrite(filename, face_gray)
                    
                    # Draw rectangle
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, f"Capturing: {captured_samples}/{MAX_SAMPLES}", 
                               (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    
                    cv2.waitKey(50)
                
                if captured_samples >= MAX_SAMPLES:
                    print(f"✅ Registration complete: {MAX_SAMPLES} samples captured for {registration_name}")
                    recognition_status = "registration_complete"
                    camera_active = False
                    registration_mode = False
                    
                    # Automatically train the model
                    print("🔄 Starting automatic training...")
                    try:
                        train_model()
                        print("✅ Training completed successfully!")
                    except Exception as e:
                        print(f"❌ Training failed: {e}")
                    
                    break
            
            # RECOGNITION MODE
            else:
                # Check timeout (30 seconds)
                current_time = cv2.getTickCount() / cv2.getTickFrequency()
                elapsed = current_time - camera_start_time
                
                if elapsed > 30 and recognition_status == "scanning":
                    recognition_status = "timeout"
                    camera_active = False
                    print("⏱️ Timeout: No matching customer found in 30 seconds")
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
                            model_name="Facenet",
                            detector_backend="skip",
                            enforce_detection=False,
                        )
                        if isinstance(rep, list):
                            emb = np.array(rep[0]["embedding"], dtype="float32")
                        else:
                            emb = np.array(rep["embedding"], dtype="float32")
                    except:
                        continue

                    dists = np.linalg.norm(db_embeddings - emb, axis=1)
                    idx = np.argmin(dists)
                    dist = dists[idx]
                    pid = int(db_labels[idx])
                    
                    # Debug: Show top 3 matches with their distances
                    sorted_indices = np.argsort(dists)
                    print(f"\n🔍 Face Detection - Top 3 matches:")
                    for i in range(min(3, len(sorted_indices))):
                        match_idx = sorted_indices[i]
                        match_dist = dists[match_idx]
                        match_id = int(db_labels[match_idx])
                        match_profile = get_profile(match_id)
                        match_name = match_profile[1] if match_profile else "Unknown"
                        marker = "✅" if i == 0 else "  "
                        print(f"{marker} #{i+1}: ID={match_id} ({match_name}) - Distance={match_dist:.2f}")
                    
                    print(f"   Threshold: {FACENET_THRESHOLD}")
                    print(f"   Selected: ID={pid} with distance={dist:.2f}")

                    if dist < FACENET_THRESHOLD:
                        profile = get_profile(pid)
                        if profile:
                            name = profile[1]

                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, f"{name} ({dist:.1f})", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                            current_face_id = pid
                            current_face_name = name
                            current_orders = orders_db.get_all_orders()
                            recognition_status = "recognized"

                            print(f"✅ FINAL MATCH: {name} (ID={pid}, Distance={dist:.2f})\n")

                            # Stop camera after recognition
                            camera_active = False
                            break

                    else:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                        cv2.putText(frame, "Unknown", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                # Show countdown
                time_left = int(30 - elapsed)
                cv2.putText(frame, f"Time: {time_left}s", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            ret, buffer = cv2.imencode(".jpg", frame)
            frame_bytes = buffer.tobytes()

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )

    finally:
        # Always release camera when done
        cam.release()
        print("📹 Camera released and stream ended")


@app.route("/video_feed")
def video_feed():
    return Response(
        gen_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/current_orders")
def current_orders_api():
    # Convert orders rows to list of dicts
    orders_list = []
    for o in current_orders:
        orders_list.append(
            {
                "order_id": o[0],
                "cust_name": o[1],
                "phone": o[2],
                "order_number": o[3],
                "amount": o[4],
                "payment_status": o[5],
                "order_status": o[6],
                "delivery_date": o[7],
                "expected_delivery_date": o[8],
                "delay_reason": o[9],
            }
        )
    return jsonify(
        {
            "face_id": current_face_id,
            "name": current_face_name,
            "orders": orders_list,
            "status": recognition_status
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
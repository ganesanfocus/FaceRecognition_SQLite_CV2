#!/usr/bin/env python3
"""
Simple face recognition test - captures your face and checks matches
"""

import cv2
import numpy as np
from deepface import DeepFace
import sqlite3

EMBEDDINGS_PATH = "recognizer/facenet_embeddings.npz"
STUDENT_DB_PATH = "database.db"
SSD_PROTO = "models/deploy.prototxt.txt"
SSD_MODEL = "models/res10_300x300_ssd_iter_140000.caffemodel"
THRESHOLD = 2.5

def get_name(pid):
    conn = sqlite3.connect(STUDENT_DB_PATH)
    cur = conn.execute("SELECT Name FROM STUDENTS WHERE id = ?", (pid,))
    row = cur.fetchone()
    conn.close()
    return row[0] if row else "Unknown"

def detect_faces_ssd(net, frame):
    """Detect faces using SSD"""
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
        if conf < 0.5:
            continue

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (x1, y1, x2, y2) = box.astype("int")
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w - 1, x2)
        y2 = min(h - 1, y2)
        boxes.append((x1, y1, x2, y2))
    
    return boxes

# Load SSD
print("Loading SSD face detector...")
ssd_net = cv2.dnn.readNetFromCaffe(SSD_PROTO, SSD_MODEL)

# Load embeddings
print("Loading embeddings...")
data = np.load(EMBEDDINGS_PATH)
db_embeddings = data["embeddings"]
db_labels = data["labels"]

print(f"✅ Loaded {len(db_labels)} embeddings for {len(np.unique(db_labels))} customers")
print(f"   Customer IDs: {sorted(np.unique(db_labels).tolist())}\n")

# Start camera
print("🎥 Starting camera...")
print("   Position your face in the frame")
print("   Press SPACE to test recognition")
print("   Press Q to quit\n")

cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    if not ret:
        break
    
    # Detect faces for visualization
    boxes = detect_faces_ssd(ssd_net, frame)
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    cv2.putText(frame, "Press SPACE to test, Q to quit", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imshow("Face Recognition Test", frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break
    elif key == ord(' '):
        print("\n" + "="*70)
        print("🔍 TESTING RECOGNITION...")
        print("="*70)
        
        # Get face region
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes = detect_faces_ssd(ssd_net, frame)
        
        if not boxes:
            print("❌ No face detected! Make sure your face is clearly visible.\n")
            continue
        
        # Use first detected face
        (x1, y1, x2, y2) = boxes[0]
        face_rgb = rgb[y1:y2, x1:x2]
        
        if face_rgb.size == 0:
            print("❌ Face region is empty!\n")
            continue
        
        try:
            # Get embedding
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
            
            # Calculate distances
            dists = np.linalg.norm(db_embeddings - emb, axis=1)
            
            # Get top 5 matches
            sorted_indices = np.argsort(dists)[:5]
            
            print("\n📊 TOP 5 MATCHES:")
            print(f"{'Rank':<8} {'ID':<8} {'Name':<20} {'Distance':<12} {'Status'}")
            print("-" * 70)
            
            for i, idx in enumerate(sorted_indices):
                match_id = int(db_labels[idx])
                match_name = get_name(match_id)
                distance = dists[idx]
                
                if distance < THRESHOLD:
                    status = "✅ MATCH"
                else:
                    status = "❌ NO MATCH"
                
                marker = "→" if i == 0 else " "
                print(f"{marker} {i+1:<6} {match_id:<8} {match_name:<20} {distance:<12.2f} {status}")
            
            # Final decision
            best_idx = sorted_indices[0]
            best_dist = dists[best_idx]
            best_id = int(db_labels[best_idx])
            best_name = get_name(best_id)
            
            print("\n" + "="*70)
            print(f"⚙️  THRESHOLD: {THRESHOLD}")
            
            if best_dist < THRESHOLD:
                print(f"✅ RESULT: RECOGNIZED as {best_name} (ID={best_id})")
                print(f"   Distance: {best_dist:.2f} (below threshold)")
            else:
                print(f"❌ RESULT: UNKNOWN PERSON")
                print(f"   Closest match: {best_name} (ID={best_id})")
                print(f"   Distance: {best_dist:.2f} (above threshold {THRESHOLD})")
            
            print("="*70 + "\n")
            
        except Exception as e:
            print(f"❌ Error: {e}\n")

cam.release()
cv2.destroyAllWindows()
print("\n✅ Test complete")
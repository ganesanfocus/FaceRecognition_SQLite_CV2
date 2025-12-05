#!/usr/bin/env python3
"""
Quick test to verify face recognition with current embeddings
"""

import cv2
import numpy as np
from deepface import DeepFace
import sqlite3

EMBEDDINGS_PATH = "recognizer/facenet_embeddings.npz"
STUDENT_DB_PATH = "database.db"
THRESHOLD = 2.5

def get_name(pid):
    conn = sqlite3.connect(STUDENT_DB_PATH)
    cur = conn.execute("SELECT Name FROM STUDENTS WHERE id = ?", (pid,))
    row = cur.fetchone()
    conn.close()
    return row[0] if row else "Unknown"

# Load embeddings
print("Loading embeddings...")
data = np.load(EMBEDDINGS_PATH)
db_embeddings = data["embeddings"]
db_labels = data["labels"]

print(f"Loaded {len(db_labels)} embeddings for {len(np.unique(db_labels))} customers")
print(f"Customer IDs in database: {sorted(np.unique(db_labels).tolist())}")

# Test with live camera
print("\n🎥 Starting camera test...")
print("Press SPACE to test recognition, 'q' to quit\n")

cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    if not ret:
        break
    
    cv2.imshow("Test Recognition - Press SPACE to test, Q to quit", frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break
    elif key == ord(' '):  # Space bar
        print("\n" + "="*60)
        print("🔍 Testing face recognition...")
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        try:
            # Get embedding for current face - use skip detection like main app
            rep = DeepFace.represent(
                img_path=rgb,
                model_name="Facenet",
                detector_backend="skip",  # Changed from "opencv" to "skip"
                enforce_detection=False    # Changed from True to False
            )
            
            if isinstance(rep, list):
                emb = np.array(rep[0]["embedding"], dtype="float32")
            else:
                emb = np.array(rep["embedding"], dtype="float32")
            
            # Calculate distances to all stored embeddings
            dists = np.linalg.norm(db_embeddings - emb, axis=1)
            
            # Get top 5 matches
            sorted_indices = np.argsort(dists)[:5]
            
            print("\n📊 Top 5 matches:")
            print(f"{'Rank':<6} {'ID':<6} {'Name':<15} {'Distance':<10} {'Match?'}")
            print("-" * 60)
            
            for i, idx in enumerate(sorted_indices):
                match_id = int(db_labels[idx])
                match_name = get_name(match_id)
                distance = dists[idx]
                is_match = "✅ YES" if distance < THRESHOLD else "❌ NO"
                
                print(f"{i+1:<6} {match_id:<6} {match_name:<15} {distance:<10.2f} {is_match}")
            
            # Final decision
            best_idx = sorted_indices[0]
            best_dist = dists[best_idx]
            best_id = int(db_labels[best_idx])
            best_name = get_name(best_id)
            
            print(f"\n{'='*60}")
            if best_dist < THRESHOLD:
                print(f"✅ RECOGNIZED: {best_name} (ID={best_id}, Distance={best_dist:.2f})")
            else:
                print(f"❌ UNKNOWN PERSON (closest match: {best_name} at {best_dist:.2f})")
            print(f"{'='*60}\n")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            print("   Make sure a face is clearly visible in the frame\n")

cam.release()
cv2.destroyAllWindows()
print("\n✅ Test complete")
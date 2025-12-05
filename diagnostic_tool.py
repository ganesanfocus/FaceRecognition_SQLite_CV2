#!/usr/bin/env python3
"""
Diagnostic Tool for Face Recognition Issues
This script helps identify why faces are being misrecognized
"""

import numpy as np
import sqlite3
import os

EMBEDDINGS_PATH = "recognizer/facenet_embeddings.npz"
STUDENT_DB_PATH = "database.db"
DATASET_DIR = "dataset"

def check_embeddings():
    """Check the embeddings file"""
    print("=" * 60)
    print("📊 CHECKING EMBEDDINGS FILE")
    print("=" * 60)
    
    if not os.path.exists(EMBEDDINGS_PATH):
        print(f"❌ Embeddings file not found: {EMBEDDINGS_PATH}")
        print("   Run: python facenet_trainer.py")
        return None, None
    
    data = np.load(EMBEDDINGS_PATH)
    embeddings = data["embeddings"]
    labels = data["labels"]
    
    print(f"✅ Embeddings loaded successfully")
    print(f"   Total samples: {len(labels)}")
    print(f"   Embedding dimensions: {embeddings.shape}")
    
    # Count samples per user
    unique_ids, counts = np.unique(labels, return_counts=True)
    print(f"\n📋 Samples per customer:")
    for uid, count in zip(unique_ids, counts):
        print(f"   ID {uid}: {count} samples")
    
    return embeddings, labels


def check_database():
    """Check the database"""
    print("\n" + "=" * 60)
    print("🗄️  CHECKING DATABASE")
    print("=" * 60)
    
    if not os.path.exists(STUDENT_DB_PATH):
        print(f"❌ Database not found: {STUDENT_DB_PATH}")
        return
    
    conn = sqlite3.connect(STUDENT_DB_PATH)
    cursor = conn.execute("SELECT id, Name, age FROM STUDENTS ORDER BY id")
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        print("❌ No customers in database")
        return
    
    print(f"✅ Found {len(rows)} customers:")
    for row in rows:
        print(f"   ID={row[0]}, Name={row[1]}, Age={row[2]}")


def check_dataset():
    """Check the dataset folder"""
    print("\n" + "=" * 60)
    print("📁 CHECKING DATASET FOLDER")
    print("=" * 60)
    
    if not os.path.exists(DATASET_DIR):
        print(f"❌ Dataset folder not found: {DATASET_DIR}")
        return
    
    files = [f for f in os.listdir(DATASET_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    if not files:
        print(f"❌ No images in {DATASET_DIR}")
        return
    
    print(f"✅ Found {len(files)} images")
    
    # Count images per user
    user_counts = {}
    for f in files:
        try:
            # Format: user.<id>.<sample>.jpg
            parts = f.split('.')
            user_id = int(parts[1])
            user_counts[user_id] = user_counts.get(user_id, 0) + 1
        except:
            print(f"   ⚠️  Skipping file with bad format: {f}")
    
    print(f"\n📸 Images per customer:")
    for uid, count in sorted(user_counts.items()):
        print(f"   ID {uid}: {count} images")


def check_distance_threshold():
    """Check if threshold is appropriate"""
    print("\n" + "=" * 60)
    print("🎯 THRESHOLD ANALYSIS")
    print("=" * 60)
    
    if not os.path.exists(EMBEDDINGS_PATH):
        return
    
    data = np.load(EMBEDDINGS_PATH)
    embeddings = data["embeddings"]
    labels = data["labels"]
    
    unique_ids = np.unique(labels)
    
    if len(unique_ids) < 2:
        print("⚠️  Need at least 2 customers to analyze distances")
        return
    
    print("Computing distances between different customers...")
    
    # Calculate average distance between same person
    same_person_dists = []
    for uid in unique_ids:
        uid_embeddings = embeddings[labels == uid]
        if len(uid_embeddings) > 1:
            for i in range(len(uid_embeddings)):
                for j in range(i+1, len(uid_embeddings)):
                    dist = np.linalg.norm(uid_embeddings[i] - uid_embeddings[j])
                    same_person_dists.append(dist)
    
    # Calculate distance between different persons
    different_person_dists = []
    for i, uid1 in enumerate(unique_ids):
        for uid2 in unique_ids[i+1:]:
            emb1 = embeddings[labels == uid1][0]
            emb2 = embeddings[labels == uid2][0]
            dist = np.linalg.norm(emb1 - emb2)
            different_person_dists.append(dist)
    
    if same_person_dists:
        avg_same = np.mean(same_person_dists)
        max_same = np.max(same_person_dists)
        print(f"\n📏 Same person distances:")
        print(f"   Average: {avg_same:.2f}")
        print(f"   Maximum: {max_same:.2f}")
    
    if different_person_dists:
        avg_diff = np.mean(different_person_dists)
        min_diff = np.min(different_person_dists)
        print(f"\n📏 Different person distances:")
        print(f"   Average: {avg_diff:.2f}")
        print(f"   Minimum: {min_diff:.2f}")
    
    # Recommend threshold
    if same_person_dists and different_person_dists:
        recommended = (max_same + min_diff) / 2
        print(f"\n💡 RECOMMENDATIONS:")
        print(f"   Current threshold: 10.0")
        print(f"   Recommended threshold: {recommended:.2f}")
        
        if max_same > 10.0:
            print(f"   ⚠️  WARNING: Some same-person distances exceed threshold!")
            print(f"      Consider increasing threshold to {max_same * 1.2:.2f}")
        
        if min_diff < 10.0:
            print(f"   ⚠️  WARNING: Some different-person distances are below threshold!")
            print(f"      Consider decreasing threshold to {min_diff * 0.8:.2f}")


def main():
    print("\n" + "=" * 60)
    print("🔧 FACE RECOGNITION DIAGNOSTIC TOOL")
    print("=" * 60 + "\n")
    
    embeddings, labels = check_embeddings()
    check_database()
    check_dataset()
    
    if embeddings is not None:
        check_distance_threshold()
    
    print("\n" + "=" * 60)
    print("✅ DIAGNOSTIC COMPLETE")
    print("=" * 60)
    print("\n💡 Next steps:")
    print("   1. If you added new customers, run: python facenet_trainer.py")
    print("   2. Check if recommended threshold differs from 10.0")
    print("   3. Restart Flask app if you updated embeddings")
    print("   4. Or use: http://127.0.0.1:5000/reload_embeddings")
    print()


if __name__ == "__main__":
    main()
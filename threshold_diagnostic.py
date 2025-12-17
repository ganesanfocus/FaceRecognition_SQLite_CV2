"""
Face Recognition Threshold Diagnostic Tool

This script helps you:
1. Test current face matching accuracy
2. Find the optimal threshold
3. Compare faces to see distances
"""

import cv2
import numpy as np
import os
from deepface import DeepFace

EMBEDDINGS_PATH = "recognizer/facenet_embeddings.npz"

def load_embeddings():
    if not os.path.exists(EMBEDDINGS_PATH):
        print(f"❌ Embeddings file not found: {EMBEDDINGS_PATH}")
        return None, None
    data = np.load(EMBEDDINGS_PATH)
    embeddings = data["embeddings"]
    labels = data["labels"]
    return embeddings, labels

def get_embedding_from_image(image_path):
    """Get FaceNet embedding from an image file"""
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"❌ Could not read image: {image_path}")
        return None
    
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
        else:
            emb = np.array(rep["embedding"], dtype="float32")
        
        return emb
    except Exception as e:
        print(f"❌ Failed to get embedding: {e}")
        return None

def analyze_embeddings():
    """Analyze all embeddings to find optimal threshold"""
    embeddings, labels = load_embeddings()
    
    if embeddings is None:
        return
    
    print("\n" + "="*60)
    print("FACE RECOGNITION THRESHOLD ANALYSIS")
    print("="*60)
    
    print(f"\n📊 Total embeddings: {len(embeddings)}")
    unique_ids = np.unique(labels)
    print(f"👥 Unique customers: {len(unique_ids)}")
    print(f"🆔 Customer IDs: {sorted(unique_ids.tolist())}")
    
    # Calculate all pairwise distances
    print("\n🔍 Calculating pairwise distances...")
    
    same_person_distances = []
    diff_person_distances = []
    
    for i in range(len(embeddings)):
        for j in range(i+1, len(embeddings)):
            dist = np.linalg.norm(embeddings[i] - embeddings[j])
            
            if labels[i] == labels[j]:
                same_person_distances.append(dist)
            else:
                diff_person_distances.append(dist)
    
    # Statistics for same person
    if same_person_distances:
        same_min = min(same_person_distances)
        same_max = max(same_person_distances)
        same_mean = np.mean(same_person_distances)
        same_std = np.std(same_person_distances)
        
        print("\n✅ SAME PERSON (should be low distances):")
        print(f"   Min:  {same_min:.3f}")
        print(f"   Max:  {same_max:.3f}")
        print(f"   Mean: {same_mean:.3f}")
        print(f"   Std:  {same_std:.3f}")
    else:
        print("\n⚠️ No same-person comparisons found (need multiple samples per person)")
        same_max = 0
    
    # Statistics for different people
    if diff_person_distances:
        diff_min = min(diff_person_distances)
        diff_max = max(diff_person_distances)
        diff_mean = np.mean(diff_person_distances)
        diff_std = np.std(diff_person_distances)
        
        print("\n❌ DIFFERENT PEOPLE (should be high distances):")
        print(f"   Min:  {diff_min:.3f}")
        print(f"   Max:  {diff_max:.3f}")
        print(f"   Mean: {diff_mean:.3f}")
        print(f"   Std:  {diff_std:.3f}")
    else:
        print("\n⚠️ No different-person comparisons found (only one person registered)")
        diff_min = float('inf')
    
    # Recommend threshold
    print("\n" + "="*60)
    print("THRESHOLD RECOMMENDATIONS")
    print("="*60)
    
    if same_person_distances and diff_person_distances:
        # Optimal threshold is between max(same) and min(diff)
        gap = diff_min - same_max
        
        print(f"\n📏 Distance gap: {gap:.3f}")
        print(f"   (difference between max same-person and min different-person)")
        
        if gap > 0.5:
            # Good separation
            recommended = (same_max + diff_min) / 2
            print(f"\n✅ GOOD SEPARATION!")
            print(f"   Recommended threshold: {recommended:.2f}")
            print(f"   This is midway between {same_max:.2f} and {diff_min:.2f}")
        elif gap > 0:
            # Some separation
            recommended = (same_max + diff_min) / 2
            print(f"\n⚠️ TIGHT SEPARATION")
            print(f"   Recommended threshold: {recommended:.2f}")
            print(f"   Warning: Small gap may cause errors")
        else:
            # Overlap - problem!
            print(f"\n❌ OVERLAP DETECTED!")
            print(f"   Max same-person distance: {same_max:.2f}")
            print(f"   Min different-person distance: {diff_min:.2f}")
            print(f"\n   PROBLEM: These overlap by {abs(gap):.2f}")
            print(f"   This means some different people are closer than same-person samples!")
            
            # Try conservative threshold
            recommended = same_max * 0.8
            print(f"\n   Conservative threshold: {recommended:.2f}")
            print(f"   (80% of max same-person distance)")
    else:
        print("\n⚠️ Need more data to calculate optimal threshold")
        recommended = 1.0
        print(f"   Default recommendation: {recommended:.2f}")
    
    print("\n" + "="*60)
    print("HOW TO UPDATE THRESHOLD IN app.py")
    print("="*60)
    print(f"\nFind this line in app.py:")
    print(f'FACENET_THRESHOLD = 2.5')
    print(f"\nChange it to:")
    print(f'FACENET_THRESHOLD = {recommended:.2f}')
    print("\nThen restart your Flask app.")
    print("="*60 + "\n")

def test_image_against_database(image_path):
    """Test a single image against all registered faces"""
    print("\n" + "="*60)
    print("TEST IMAGE AGAINST DATABASE")
    print("="*60)
    
    embeddings, labels = load_embeddings()
    if embeddings is None:
        return
    
    print(f"\n📸 Testing image: {image_path}")
    
    test_emb = get_embedding_from_image(image_path)
    if test_emb is None:
        return
    
    print("\n🔍 Comparing with all registered faces:")
    print("-" * 60)
    
    distances = []
    for i, (emb, label) in enumerate(zip(embeddings, labels)):
        dist = np.linalg.norm(emb - test_emb)
        distances.append((dist, label, i))
    
    # Sort by distance
    distances.sort()
    
    # Show top 5 matches
    print("\nTop 5 closest matches:")
    for rank, (dist, label, idx) in enumerate(distances[:5], 1):
        print(f"  {rank}. ID={label} | Distance={dist:.3f} | Sample #{idx}")
    
    # Show what would happen with different thresholds
    print("\n" + "-" * 60)
    print("Prediction with different thresholds:")
    print("-" * 60)
    
    best_match_dist, best_match_id, _ = distances[0]
    
    for threshold in [0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]:
        if best_match_dist < threshold:
            print(f"  Threshold {threshold:.1f}: ✅ Match → ID {best_match_id} (dist={best_match_dist:.3f})")
        else:
            print(f"  Threshold {threshold:.1f}: ❌ No match (dist={best_match_dist:.3f})")

if __name__ == "__main__":
    import sys
    
    print("""
╔══════════════════════════════════════════════════════════════╗
║        Face Recognition Threshold Diagnostic Tool            ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    if len(sys.argv) > 1:
        # Test specific image
        image_path = sys.argv[1]
        if os.path.exists(image_path):
            test_image_against_database(image_path)
            print("\nNow running full analysis...\n")
            analyze_embeddings()
        else:
            print(f"❌ Image not found: {image_path}")
    else:
        # Just analyze embeddings
        analyze_embeddings()
        
        print("\n💡 TIP: To test a specific image:")
        print("   python threshold_diagnostic.py path/to/your/image.jpg")
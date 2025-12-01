# facenet_trainer.py
import os
import cv2
import numpy as np
from deepface import DeepFace

DATASET_DIR = "dataset"
EMBEDDINGS_PATH = "recognizer/facenet_embeddings.npz"


def get_image_paths(path):
    return [
        os.path.join(path, f)
        for f in os.listdir(path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]


def main():
    os.makedirs(os.path.dirname(EMBEDDINGS_PATH), exist_ok=True)

    print("[INFO] Preparing FaceNet embeddings using DeepFace...")
    # NOTE: We are NOT passing 'model=' anywhere now.
    # DeepFace will handle model loading internally.

    image_paths = get_image_paths(DATASET_DIR)
    if not image_paths:
        print("[ERROR] No images found in dataset/. Run dataset_creater_ssd.py first.")
        return

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
                detector_backend="skip",   # our dataset images are already cropped faces
                enforce_detection=False
            )

            # DeepFace may return list or dict depending on version
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
        print("[ERROR] No embeddings were created. Check your dataset.")
        return

    embeddings = np.vstack(embeddings)
    labels = np.array(labels, dtype="int32")

    np.savez(EMBEDDINGS_PATH, embeddings=embeddings, labels=labels)
    print(f"[INFO] Saved embeddings to {EMBEDDINGS_PATH}")
    print(f"[INFO] Total embeddings: {len(labels)}")


if __name__ == "__main__":
    main()

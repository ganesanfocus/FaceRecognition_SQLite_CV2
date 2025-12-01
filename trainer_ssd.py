# trainer.py
import cv2
import numpy as np
import os

DATASET_DIR = "dataset"
MODEL_PATH = "recognizer/trainingdata.yml"

# Create recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

def get_images_and_labels(path):
    image_paths = [
        os.path.join(path, f)
        for f in os.listdir(path)
        if f.lower().endswith((".jpg", ".png"))
    ]

    faces = []
    ids = []

    for image_path in image_paths:
        # Read image in grayscale
        img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            print(f"Warning: could not read {image_path}")
            continue

        # Filename format: user.<id>.<sample>.jpg
        try:
            id_str = os.path.split(image_path)[-1].split(".")[1]
            id_num = int(id_str)
        except (IndexError, ValueError):
            print(f"Skipping file with bad name: {image_path}")
            continue

        faces.append(img_gray)
        ids.append(id_num)

    return np.array(ids), faces


if __name__ == "__main__":
    ids, faces = get_images_and_labels(DATASET_DIR)

    if len(faces) == 0:
        print("No images found in dataset/. Run dataset_creater_ssd.py first.")
        exit(0)

    print(f"Training on {len(faces)} images...")
    recognizer.train(faces, ids)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    recognizer.save(MODEL_PATH)
    print(f"Training complete. Model saved to {MODEL_PATH}")

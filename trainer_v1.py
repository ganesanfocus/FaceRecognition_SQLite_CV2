import os
import cv2
import numpy as np
from PIL import Image

recognizer = cv2.LBPHFaceRecognizer_create()  # To recognize face in camera
path = 'dataset'

def get_images_with_id(path):
    images_paths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    ids = []
    for single_image_path in images_paths:
        faceImg = Image.open(single_image_path).convert('L') # L Luminous conver color to gray
        faceNp = np.array(faceImg.np.unit8)
        id = int(os.path.split(single_image_path)[-1].split(".")[1])
        print(id)
        faces.append(faceNp)
        
        cv2.imshow("Training :", faceNp)
        cv2.waitkey(10)
        
    return np.array(ids), faces

ids, faces = get_images_with_id(path)
recognizer.train(faces, ids)
recognizer.save("recognizer/trainingdata.yml")
cv2.destryAllWindows()

        

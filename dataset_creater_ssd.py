# dataset_creator.py
import cv2
import numpy as np
import sqlite3
import os

# --- Configuration ---
DB_PATH = "database.db"
DATASET_DIR = "dataset/"

# SSD model paths
SSD_PROTO = "models/deploy.prototxt.txt"
SSD_MODEL = "models/res10_300x300_ssd_iter_140000.caffemodel"
CONF_THRESHOLD = 0.5  # confidence threshold for face detection

# --- Database Functions ---

def init_db():
    """Initializes the SQLite database and creates the STUDENTS table if it doesn't exist."""
    print("Initializing database...")
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS STUDENTS (
                    id INTEGER NOT NULL PRIMARY KEY,
                    Name TEXT NOT NULL,
                    age INTEGER
                )
            """)
        print("Database initialized successfully.")
    except sqlite3.Error as e:
        print(f"An error occurred during database initialization: {e}")

def inserorupdate(Id, Name, age):
    """Inserts a new student record or updates an existing one."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # 1. Check if record exists
        cmd = "SELECT * FROM STUDENTS WHERE id = ?"
        cursor.execute(cmd, (Id,))
        
        isRecordExist = cursor.fetchone() is not None
        
        # 2. Execute INSERT or UPDATE
        if isRecordExist:
            print(f"Updating record for ID: {Id}")
            conn.execute("UPDATE STUDENTS SET Name=?, age=? WHERE id=?", (Name, age, Id))    
        else:
            print(f"Inserting new record for ID: {Id}")
            conn.execute("INSERT INTO STUDENTS (id, Name, age) VALUES (?, ?, ?)", (Id, Name, age))
        
        conn.commit()
    except sqlite3.Error as e:
        print(f"Database operation failed: {e}")
    finally:
        if conn:
            conn.close()

# --- Load SSD face detector ---

def load_ssd_face_detector():
    if not os.path.exists(SSD_PROTO) or not os.path.exists(SSD_MODEL):
        raise FileNotFoundError(
            f"SSD model files not found.\n"
            f"Expected:\n  {SSD_PROTO}\n  {SSD_MODEL}"
        )
    print("Loading SSD face detector...")
    net = cv2.dnn.readNetFromCaffe(SSD_PROTO, SSD_MODEL)
    print("SSD face detector loaded.")
    return net

def detect_faces_ssd(net, frame, conf_threshold=CONF_THRESHOLD):
    """
    Detect faces using SSD.
    Returns list of bounding boxes: [(x1, y1, x2, y2), ...]
    """
    (h, w) = frame.shape[:2]
    # Create blob (resize to 300x300 as required by this SSD model)
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)),
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0)   # mean subtraction values
    )
    net.setInput(blob)
    detections = net.forward()

    boxes = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < conf_threshold:
            continue

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (x1, y1, x2, y2) = box.astype("int")

        # Clip to frame boundaries
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w - 1, x2)
        y2 = min(h - 1, y2)

        boxes.append((x1, y1, x2, y2))

    return boxes

# --- Main Script Execution ---

# Ensure dataset dir exists
os.makedirs(DATASET_DIR, exist_ok=True)

# Initialize DB
init_db()

# --- User Input ---
print("\n--- Enter Student Details ---")
Id   = input("Enter user id: ")
Name = input("Enter user Name: ")
age  = input("Enter user Age: ")

# Store or update the data in the database
inserorupdate(Id, Name, age)

# --- Face Data Collection ---
print("\nStarting face data collection with SSD face detector. Look directly into the camera.")

try:
    # Load SSD detector
    net = load_ssd_face_detector()

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        raise IOError("Cannot open webcam. Check camera connection or access permissions.")

    sampleNum = 0
    MAX_SAMPLES = 30   # number of images per user

    while True:
        ret, img = cam.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Detect faces using SSD
        boxes = detect_faces_ssd(net, img, CONF_THRESHOLD)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        for (x1, y1, x2, y2) in boxes:
            if sampleNum >= MAX_SAMPLES:
                break

            sampleNum += 1

            # Crop face region and save (LBPH expects grayscale)
            face_gray = gray[y1:y2, x1:x2]
            filename = f"{DATASET_DIR}user.{Id}.{sampleNum}.jpg"
            cv2.imwrite(filename, face_gray)

            # Draw rectangle on color image
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.putText(
                img,
                f"Samples: {sampleNum}/{MAX_SAMPLES}",
                (x1, max(0, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            cv2.waitKey(50)

        cv2.imshow("Face Detector - Collecting Samples (SSD)", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if sampleNum >= MAX_SAMPLES:
            print(f"Successfully collected {MAX_SAMPLES} samples.")
            break

except FileNotFoundError as e:
    print(f"Error: {e}")
except IOError as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

# --- Cleanup ---
if 'cam' in locals() and cam.isOpened():
    cam.release()

cv2.destroyAllWindows()
print("Camera stream closed and windows destroyed.")

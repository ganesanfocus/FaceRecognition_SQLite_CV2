import cv2
import numpy as np
import sqlite3

# --- Configuration ---
DB_PATH = "database.db"
HAARCASCADE_PATH = 'haarcascade_frontalface_default.xml'
DATASET_DIR = "dataset/"

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
        
        # 1. Check if record exists (Using parameterized query for security)
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

# --- Main Script Execution ---

# **CRITICAL FIX:** Call init_db() to ensure the table exists before use
init_db()

# --- User Input ---
# Note: Input values are stored as strings. SQLite handles the conversion for the database.
print("\n--- Enter Student Details ---")
Id   = input("Enter user id: ")
Name = input("Enter user Name: ")
age  = input("Enter user Age: ")

# Store or update the data in the database
inserorupdate(Id, Name, age)

# --- Face Data Collection ---
print("\nStarting face data collection. Look directly into the camera.")

try:
    # Initialize face detection and camera
    faceDetect = cv2.CascadeClassifier(HAARCASCADE_PATH)
    cam = cv2.VideoCapture(0)

    if not cam.isOpened():
        raise IOError("Cannot open webcam. Check camera connection or access permissions.")

    # detect face in the web camera
    sampleNum = 0
    MAX_SAMPLES = 30 # Increased count for better training data
    
    while(True):
        ret, img = cam.read()
        
        # Check if frame was read successfully
        if not ret:
            print("Failed to grab frame.")
            break
            
        # FIX: Correct OpenCV function call: cv2.cvtColor
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = faceDetect.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            # Increment sample number and check if we are still collecting
            if sampleNum < MAX_SAMPLES:
                sampleNum += 1
                
                # Save the face image to the dataset directory
                # Filename format: dataset/user.[Id].[sampleNum].jpg
                cv2.imwrite(f"{DATASET_DIR}user.{Id}.{sampleNum}.jpg", gray[y:y+h, x:x+w])
                
                # Draw a rectangle around the face
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # FIX: Correct capitalization: cv2.waitKey
                # Short delay to see the rectangle drawn
                cv2.waitKey(50) 
            else:
                break # Break out of the for loop if enough samples are collected

        # Display the video feed
        cv2.imshow("Face Detector - Collecting Samples", img)
        
        # FIX: Correct capitalization: cv2.waitKey (1ms delay)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break # Exit on 'q' press
            
        # Break the main loop after collecting enough samples
        if (sampleNum >= MAX_SAMPLES):
            print(f"Successfully collected {MAX_SAMPLES} samples.")
            break
            
except FileNotFoundError:
    print(f"Error: Could not find '{HAARCASCADE_PATH}'. Ensure the file is in the script directory.")
except IOError as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    
# --- Cleanup (CRITICAL FIX: These lines must be OUTSIDE the while loop) ---
if 'cam' in locals() and cam.isOpened():
    cam.release()
    
cv2.destroyAllWindows()
print("Camera stream closed and windows destroyed.")
import cv2
import numpy as np
import sqlite3


DB_PATH = "sqlite.db"

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
                CREATE TABLE IF NOT EXISTS STUDENTS (
                    id INTEGER NOT NULL ,
                    Name TEXT NOT NULL,
                    age INTEGER
                )
            """)

faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # To detect the face in camera

cam = cv2.VideoCapture(0)

def inserorupdate(Id, Name, age):
    conn = sqlite3.connect('database.db')
    cmd = "SELECT * FROM STUDENTS WHERE ID ="+str(Id)+ " "
    cursor = conn.execute(cmd)
    
    isRecordExist = 0
    for row in cursor:
        isRecordExist = 1
    
    if isRecordExist == 1:
        conn.execute("UPDATE STUDENTS SET Name=?, age=? where Id=?", (Name, age, Id))    
    else:
        conn.execute("INSERT INTO STUDENTS (Id, Name, age) values(?, ?, ?)", (Id, Name, age))
    
    conn.commit()
    conn.close()
Id   = input("Enter user id ")
Name = input("Enter user Name")
age  = input("Enter user Age")

inserorupdate(Id, Name, age)

# detect face in the web camera
sampleNum = 0
while(True):
    ret, img = cam.read()
    gray = cv2.cvColor(img.cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3, 5)
    for (x,y, w,h) in faces:
        sampleNum = sampleNum + 1
        cv2.imwrite("dataset/user."+str(Id)+str(sampleNum)+".jpg", gray[y:y+h, x:x+w])
        cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 2)
        cv2.waitkey(100)
    cv2.imshow("Face", img)
    cv2.waitkey(1)
    
    if (sampleNum > 20 ):
        break
    cam.release()
    cv2.destroyAllWindows()
    
    
            
    
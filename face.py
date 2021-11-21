import cv2
import face_recognition
import numpy as np
import os
from datetime import datetime


cap = cv2.VideoCapture(0)
img_directory = "Images"
lst = os.listdir(img_directory)
imgs = []
class_names = []

for cl in lst:
    img = cv2.imread(os.path.join(img_directory, cl))
    imgs.append(img)
    class_names.append(os.path.splitext(cl)[0])
print(class_names)

def find_encodings(imgs):
    encode_lst = []
    for img in imgs:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encode_lst.append(encode)
    return encode_lst

encode_lst = find_encodings(imgs)
p_time, c_time = 0, 0

def markAttendance(name):
    with open('Attendance.csv','r+',encoding="Latin-1") as f:
        myDataList=f.readlines()
        nameList=[]
        for line in myDataList:
            entry=line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now=datetime.now()
            dtString=now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')
encode_lst = find_encodings(imgs)
print('Enconding Complete')
cap=cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgs = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)

    faces_location = face_recognition.face_locations(imgs)
    encode_faces = face_recognition.face_encodings(imgs, faces_location)

    for face_location, encode_face in zip(faces_location, encode_faces):
        results = face_recognition.compare_faces(encode_lst, encode_face)
        distance = face_recognition.face_distance(encode_lst, encode_face)
        print (results)
        matchIndex = np.argmin(distance)
        
        if results[matchIndex]:
            name = class_names[matchIndex].upper()
            
        y1, x2, y2, x1 = face_location
        y1, x2, y2, x1 =  y1 * 4, x2 * 4, y2 * 4, x1 * 4

        cv2.rectangle(img, (x1, y1), (x2 , y2), (255, 0, 0), 3)
        max_class = np.argmin(distance)
        person = class_names[max_class].upper()
        cv2.putText(img, person, (x1 - 10, y1 - 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        markAttendance(name)

    #c_time = datetime.time()
    #fps = 1 / (c_time - p_time)
    #p_time = c_time
    #cv2.putText(img, f"FPS: {int(fps)}", (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    cv2.imshow("Video", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    
    
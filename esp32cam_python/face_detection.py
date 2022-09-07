import pandas as pd
import cv2
import urllib.request
import numpy as np
import os
from datetime import datetime
from pathlib import Path
from statistics import mean as average
# import face_recognition


# path = r'C:\Users\Matias\Desktop\esp32_faceRecogniton\image_folder'
BASE_DIR = Path(__file__).absolute().parent
path_to_images = os.path.join(BASE_DIR, "image_folder")
path_to_csv = os.path.join(BASE_DIR)
path_to_caffemodel = os.path.join(BASE_DIR, "res10_300x300_ssd_iter_140000_fp16.caffemodel")
path_to_deploy = os.path.join(BASE_DIR, "deploy.prototxt")

url = 'http://ip/cam-hi.jpg'
'''cam.bmp / cam-lo.jpg /cam-hi.jpg / cam.mjpeg '''


def face_detect(images):
    # for caffemodel detect face on images
    in_width = 120
    in_height = 120
    mean = [104, 117, 123]
    conf_threshold = 0.7
    confs = []
    count = 0
    net = cv2.dnn.readNetFromCaffe(path_to_deploy, path_to_caffemodel)

    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GARAY)
        if count > 0:
            average_conf = average(confs)
            if average_conf > 0.96:
                return True

        blob = cv2.dnn.blobFromImage(img, 1.0, (in_width, in_height), mean, swapRB=False, crop=False)
        net.setInput(blob)
        detections = net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                confs.append(confidence)
                count += 1


        # encode = face_recognition.face_encodings(img)[0]
        # encodeList.append(encode)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

if 'Attendance.csv' in os.listdir(path_to_csv):
    print("there iss..")
    os.remove("Attendance.csv")
else:
    df=pd.DataFrame(list())
    df.to_csv("Attendance.csv")

images = []
classNames = []
myList = os.listdir(path_to_images)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path_to_images}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def markAttendance(name):
    with open("Attendance.csv", 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
            if name not in nameList:
                now = datetime.now()
                dtString = now.strftime('%H:%M:%S')
                f.writelines(f'\n{name},{dtString}')


# encodeListKnown = findEncodings(images)

print('Encoding Complete')
 
#cap = cv2.VideoCapture(0)

while True:
    #success, img = cap.read()
    img_resp=urllib.request.urlopen(url)
    imgnp=np.array(bytearray(img_resp.read()),dtype=np.uint8)
    img=cv2.imdecode(imgnp,-1)
# img = captureScreen()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
 
    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
# print(faceDis)
        matchIndex = np.argmin(faceDis)
 
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
# print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            # markAttendance(name)
 
    cv2.imshow('Webcam', img)
    key=cv2.waitKey(5)
    if key==ord('q'):
        break
cv2.destroyAllWindows()
cv2.imread
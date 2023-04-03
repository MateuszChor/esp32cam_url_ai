import cv2
import urllib.request
import numpy as np
import os
from pathlib import Path
from statistics import mean as average


# path = r'C:\Users\Matias\Desktop\esp32_faceRecogniton\image_folder'
BASE_DIR = Path(__file__).absolute().parent
path_to_images = os.path.join(BASE_DIR, "image_folder")
path_to_csv = os.path.join(BASE_DIR)
path_to_caffemodel = os.path.join(BASE_DIR, "../modules/res10_300x300_ssd_iter_140000_fp16.caffemodel")
path_to_deploy = os.path.join(BASE_DIR, "../modules/deploy.prototxt")

url = 'http://192.168.8.116/cam-hi.jpg'
'''cam.bmp / cam-lo.jpg /cam-hi.jpg / cam.mjpeg '''



while True:
    #success, img = cap.read()
    img_resp=urllib.request.urlopen(url)
    imgnp=np.array(bytearray(img_resp.read()),dtype=np.uint8)
    img=cv2.imdecode(imgnp,-1)
    # img = captureScreen()
    imgResize = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    # imgS = cv2.cvtColor(imgResize, cv2.COLOR_BGR2RGB)
    imgGray = cv2.cvtColor(imgResize, cv2.COLOR_BGR2GRAY)

    
    # found = stop_data.detectMultiScale(imgGray, minSize =(20, 20))
    # found = cars_data.detectMultiScale(imgGray,1.1, 1)

    # Don't do anything if there's  
    # no sign
    # amount_found = len(found)

    # print(found)


    net = cv2.dnn.readNetFromCaffe(path_to_deploy, path_to_caffemodel)
    h, w = imgResize.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(imgResize, (300, 300)), 1.0,
    (300, 300), (104.0, 117.0, 123.0))
    net.setInput(blob)
    faces = net.forward()
    #to draw faces on image
    for i in range(faces.shape[2]):
        confidence = faces[0, 0, i, 2]
        if confidence > 0.7:
            box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x1, y1) = box.astype("int")
            cv2.rectangle(imgResize, (x, y), (x1, y1), (0, 0, 255), 2)

            print(faces)
    
    # # There may be more than one
    # # sign in the image
    # for (x, y, width, height) in found:
        
    #     # We draw a green rectangle around
    #     # every recognized sign
    #     cv2.rectangle(imgResize, (x, y), 
    #                 (x + height, y + width), 
    #                 (0, 255, 0), 5)

    cv2.imshow('Webcam', imgResize)
    key=cv2.waitKey(5)
    if key==ord('q'):
        break

cv2.destroyAllWindows()
cv2.imread
import cv2
import urllib.request
import numpy as np
import os
from pathlib import Path

BASE_DIR = Path(__file__).absolute().parent
path_to_images = os.path.join(BASE_DIR, "image_folder")

path_to_stop_xml = os.path.join(BASE_DIR, "../modules/stop_data.xml")
path_to_cars_xml = os.path.join(BASE_DIR, "../modules/cars.xml")

path_to_caffemodel = os.path.join(BASE_DIR, "../modules/res10_300x300_ssd_iter_140000_fp16.caffemodel")
path_to_deploy = os.path.join(BASE_DIR, "../modules/deploy.prototxt")

url = 'http://192.168.8.116/cam-hi.jpg'

stop_data = cv2.CascadeClassifier(path_to_stop_xml)
cars_data = cv2.CascadeClassifier(path_to_cars_xml)

while True:

    # success, img = cap.read()
    img_resp = urllib.request.urlopen(url)

    img_np = np.array(bytearray(img_resp.read()), dtype=np.uint8)
    img = cv2.imdecode(img_np, -1)

    # img = captureScreen()
    imgResize = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    # imgS = cv2.cvtColor(imgResize, cv2.COLOR_BGR2RGB)
    imgGray = cv2.cvtColor(imgResize, cv2.COLOR_BGR2GRAY)

    found = stop_data.detectMultiScale(imgGray, minSize=(20, 20))
    # found = cars_data.detectMultiScale(imgGray, 1.1, 1)
    # found = cars_data.detectMultiScale(imgGray, minSize=(20, 20))

    # net.setInput(blob)

    # There may be more than one
    # sign in the image
    for (x, y, width, height) in found:
        # We draw a green rectangle around
        # every recognized sign
        cv2.rectangle(imgResize, (x, y),
                      (x + height, y + width),
                      (0, 255, 0), 5)

    cv2.imshow('Esp32Cam', imgResize)
    key = cv2.waitKey(5)
    if key == ord('q'):
        break

cv2.destroyAllWindows()

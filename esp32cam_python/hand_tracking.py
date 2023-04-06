import cv2
import urllib.request
import numpy as np
import os
from pathlib import Path
import mediapipe as mp

BASE_DIR = Path(__file__).absolute().parent
path_to_modules = os.path.join(BASE_DIR, "modules")

url = 'http://192.168.8.116/cam-hi.jpg'

mpHands = mp.solutions.hands
hands = mpHands.Hands()  # default parameters of model
mpDraw = mp.solutions.drawing_utils  # drawing line tool


while True:
    # success, img = cap.read() # read from  camera hardware
    url_img = urllib.request.urlopen(url)  # open address url with url to camera

    img_np = np.array(bytearray(url_img.read()), dtype=np.uint8)  # on matrix
    img = cv2.imdecode(img_np, -1)  # cv2 flag -1 encoding original format


    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    # results.multi_hand_landmarks # <-- landmarks
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms)


    cv2.imshow('Esp32Cam', img)
    key = cv2.waitKey(5)
    if key == ord('q'):
        break

cv2.destroyAllWindows()

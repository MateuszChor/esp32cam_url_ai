import cv2
import urllib.request
import numpy as np
import os
from pathlib import Path
import mediapipe as mp
import time

BASE_DIR = Path(__file__).absolute().parent
path_to_modules = os.path.join(BASE_DIR, "modules")

url = 'http://192.168.8.116/cam-hi.jpg'

mpHands = mp.solutions.hands
hands = mpHands.Hands()  # default parameters of model
mpDraw = mp.solutions.drawing_utils  # drawing line tool

pTime = 0
cTime = 0

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
            for id_number, lm, in enumerate(handLms.landmark):
                # print(id_number, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y*h)
                print(id_number, cx, cy)
                if id_number == 4:  # draw circle for id landmark
                    cv2.circle(img, (cx, cy), 25, (255, 0, 255), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    # calculate fps
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 255), 3)

    cv2.imshow('Esp32Cam', img)
    key = cv2.waitKey(5)
    if key == ord('q'):
        break

cv2.destroyAllWindows()

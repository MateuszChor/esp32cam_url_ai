import cv2
import urllib.request
import numpy as np
import os
from pathlib import Path
import mediapipe as mp
import time

BASE_DIR = Path(__file__).absolute().parent
path_to_modules = os.path.join(BASE_DIR, "modules")


class handDetector:
    # hand landmarks doc https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
    def __init__(self, mode=False, maxHands=2, model_complexity=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.model_complexity = model_complexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.model_complexity,
                                        self.detectionCon, self.trackCon)  # parameters of model
        self.mpDraw = mp.solutions.drawing_utils  # drawing line tool

    def findHands(self, img, draw=True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        # results.multi_hand_landmarks # <-- landmarks
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNumber=0, draw=True):

        lmList = []
        if self.results.multi_hand_landmarks:

            myHand = self.results.multi_hand_landmarks[handNumber]
            for id_number, lm, in enumerate(myHand.landmark):
                # print(id_number, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(id_number, cx, cy)
                lmList.append([id_number, cx, cy])
                if id_number == 4:  # for get to specific landmarks
                    # print(id_number, lm)
                    if draw:
                        cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        return lmList

def get_video_from_esp32(url):
    pTime = 0
    cTime = 0

    detector = handDetector()

    while True:
        # success, img = cap.read() # read from  camera hardware
        url_img = urllib.request.urlopen(url)  # open address url with url to camera

        img_np = np.array(bytearray(url_img.read()), dtype=np.uint8)  # on matrix
        img = cv2.imdecode(img_np, -1)  # cv2 flag -1 encoding original format

        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[4])

        # calculate fps
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 255), 3)

        cv2.imshow('Esp32Cam', img)
        key = cv2.waitKey(5)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()


url = 'http://192.168.8.116/cam-hi.jpg'

get_video_from_esp32(url)

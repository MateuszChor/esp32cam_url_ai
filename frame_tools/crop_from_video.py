import cv2
from pathlib import Path
import os
import imageio


BASE_DIR = Path(__file__).absolute().parent
path_to_video = "C:\\Users\\apollo\\Videos\\Captures\\Esp32Cam 2023-04-06 16-58-22.mp4"
# path_to_videos = os.path.join(BASE_DIR, "")


def save_cropped_vid_mp4(path_to_video):

    capture = cv2.VideoCapture(path_to_video)

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print('Counted frames = ', total_frames)

    x = 20
    y = 10
    w = 780
    h = 590

    # save parameters
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('cropped_video.mp4', fourcc, 30.0, (w, h))

    for i in range(total_frames):
        ret, frame = capture.read()

        cropped_frame = frame[y:y + h, x:x + w]

        out.write(cropped_frame)

        cv2.imshow('cropped frame', cropped_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    out.release()

    cv2.destroyAllWindows()

# save_cropped_vid_mp4(path_to_video)

def save_cropped_video_gif(path_to_video):

    capture = cv2.VideoCapture(path_to_video)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print('Counted frames = ', total_frames)

    x = 20
    y = 10
    w = 780
    h = 590

    frames = []

    # for i in range(total_frames):
    #     ret, frame = capture.read()
    #     cropped_frame = frame[y:y + h, x:x + w]
    #     frames.append(cropped_frame)

    # cut frames
    start = 400
    stop = 460

    for i in range(stop):
        ret, frame = capture.read()
        cropped_frame = frame[y:y + h, x:x + w]
        if i > start:
            frames.append(cropped_frame)



    capture.release()
    imageio.mimsave('cropped_video.gif', frames, fps=30)
    print("Gif video saved")


save_cropped_video_gif(path_to_video)

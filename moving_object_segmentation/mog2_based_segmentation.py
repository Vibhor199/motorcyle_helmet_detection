import cv2 as cv
import numpy as np
import pafy


def detect_motion_in_youtube(url: str):
    v_pafy = pafy.new(url)
    play = v_pafy.getbestvideo(preftype="webm")

    background_subtract = cv.createBackgroundSubtractorMOG2(history = 2,varThreshold = 16, detectShadows = True)
    # backSub = cv.createBackgroundSubtractorKNN()

    capture = cv.VideoCapture(play.url)
    # capture = cv.VideoCapture(0)

    while True:
        ret, frame = capture.read()
        img = cv.GaussianBlur(frame, (5, 5), 0)
        # frame = cv.flip(frame   , 1)

        fg_mask = background_subtract.apply(img)
        kernel = np.ones((11, 11), np.uint8)
        fg_mask = cv.dilate(fg_mask, kernel, iterations = 1)
        fg_mask = cv.morphologyEx(fg_mask, cv.MORPH_CLOSE, kernel)
        fg_mask = cv.morphologyEx(fg_mask, cv.MORPH_OPEN, kernel)

        res = cv.bitwise_and(frame, frame, mask=fg_mask)

        cv.imshow('Frame', frame)
        cv.imshow('FG Mask', res)
        keyboard = cv.waitKey(30)
        if keyboard == 'q' or keyboard == 27:
            break
    # When everything done, release the capture
    capture.release()
    cv.destroyAllWindows()

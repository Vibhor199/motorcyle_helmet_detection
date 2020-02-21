import cv2 as cv
import numpy as np
import pafy


def detect_motion_in_youtube(url: str):
    vPafy = pafy.new(url)
    play = vPafy.getbestvideo(preftype="webm")

    backSub = cv.createBackgroundSubtractorMOG2(history = 2,varThreshold = 16, detectShadows = True)
    # backSub = cv.createBackgroundSubtractorKNN()

    capture = cv.VideoCapture(play.url)
    # capture = cv.VideoCapture(0)

    while True:
        ret, frame = capture.read()
        img = cv.GaussianBlur(frame,(5,5),0)
        # frame = cv.flip(frame   , 1)

        fgMask = backSub.apply(img)
        kernel = np.ones((11,11),np.uint8)
        fgMask = cv.dilate(fgMask,kernel,iterations = 1)
        fgMask = cv.morphologyEx(fgMask, cv.MORPH_CLOSE, kernel)
        fgMask = cv.morphologyEx(fgMask, cv.MORPH_OPEN, kernel)

        res = cv.bitwise_and(frame,frame,mask = fgMask)

        cv.imshow('Frame', frame)
        cv.imshow('FG Mask', res)
        keyboard = cv.waitKey(30)
        if keyboard == 'q' or keyboard == 27:
            break
    # When everything done, release the capture
    capture.release()
    cv.destroyAllWindows()

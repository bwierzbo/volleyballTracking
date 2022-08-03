import cv2 as cv
import numpy as np

videoCapture = cv.VideoCapture('test_vid.avi')

backSub = cv.createBackgroundSubtractorKNN()

while True:
    ret, frame = videoCapture.read()
    if not ret: break
    
    mask = backSub.apply(frame)


    mask = cv.GaussianBlur(mask, (13, 13),0)
    ret,mask = cv.threshold(mask,0,255,cv.THRESH_BINARY | cv.THRESH_OTSU)

    

    cv.imshow("movingMask", mask)

    if cv.waitKey(10) & 0xFF == ord('q'): break

videoCapture.release()
cv.destroyAllWindows()
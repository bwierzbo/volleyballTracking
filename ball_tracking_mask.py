import cv2 as cv
import numpy as np

videoCapture = cv.VideoCapture('test_vid.avi')
prevCircle = None
dist = lambda x1,y1,x2,y2: (x1-x2)**2+(y1-y2)**2

backSub = cv.createBackgroundSubtractorKNN()

while True:
    ret, frame = videoCapture.read()
    if not ret: break
    
    mask = backSub.apply(frame)


    mask = cv.GaussianBlur(mask, (13, 13),0)
    ret,mask = cv.threshold(mask,0,255,cv.THRESH_BINARY | cv.THRESH_OTSU)

    circles = cv.HoughCircles(mask, cv.HOUGH_GRADIENT_ALT, 1.5, 100, param1=300, param2= 0.8, minRadius=8, maxRadius=50)

    if circles is not None:
            circles = np.uint16(np.around(circles))
            chosen = None
            for i in circles[0, :]:
                if chosen is None: chosen = i
                if prevCircle is not None:
                    if dist(chosen[0],chosen[1],prevCircle[0],prevCircle[1]) <= dist(i[0],i[1],prevCircle[0],prevCircle[1]):
                        chosen = i
            cv.circle(frame, (chosen[0], chosen[1]), 1, (0,100,100), 3)
            cv.circle(frame, (chosen[0], chosen[1]), chosen[2], (255,0,255), 3)

            prevCircle = chosen

    

    cv.imshow("movingMask", frame)

    if cv.waitKey(10) & 0xFF == ord('q'): break

videoCapture.release()
cv.destroyAllWindows()
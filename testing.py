import cv2
import os


cam = cv2.VideoCapture("video/V1.mp4")
name="V1"


frameno = 0
while(True):
    ret,frame = cam.read()
    if ret:
       
        cv2.imwrite(name+str(frameno), frame)
        frameno += 1
    else:
        break

cam.release()
cv2.destroyAllWindows()

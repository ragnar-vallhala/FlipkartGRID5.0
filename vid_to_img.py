import cv2
import os

for i in range(12,23):
   cam = cv2.VideoCapture('video/V' + str(i)+ '.mp4')



   frameno = 0
   while(True):
      ret,frame = cam.read()
      if ret:
         # if video is still left continue creating images
         name ='imagestore\Side\V'+str(i) + str(frameno) + '.jpg'
         
         
         print ('new frame captured...' + name)

         cv2.imwrite(name, frame)
         frameno += 1
      else:
         break

   cam.release()
   cv2.destroyAllWindows()

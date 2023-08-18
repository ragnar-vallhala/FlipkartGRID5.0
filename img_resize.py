import cv2 as cv
import os

path = "C:/Users/IIT JAMMU/Documents/Flipkart/imagestore/Side"
files = os.listdir(path)

for i in files:
    mg = cv.imread(path + '/' + i,1)
    mg =  cv.resize(mg,(128,128), interpolation = cv.INTER_AREA)
    cv.imwrite(path + '/' + i, mg)
    cv.imshow("image", mg)
    cv.waitKey(1)
        
cv.destroyAllWindows()
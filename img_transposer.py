import cv2 as cv
import os

path = "C:/Users/IIT JAMMU/Documents/Flipkart/imagestore/Side"
files = os.listdir(path)

for i in files:
    mg = cv.imread(path + '/' + i,1)
    if (mg.shape[0] > mg.shape[1]):
        mg = cv.transpose(mg)
        cv.imwrite(path + '/' + i, mg)
        cv.imshow("image", mg)
        cv.waitKey(1)
        
cv.destroyAllWindows()


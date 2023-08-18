import cv2 as cv


mg = cv.imread("imagestore\Top\V10.jpg",1)

cv.imshow("image", mg)
cv.waitKey(0)
cv.destroyAllWindows()
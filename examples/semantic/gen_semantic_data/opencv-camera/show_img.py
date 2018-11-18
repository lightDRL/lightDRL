# import numpy as np
import cv2


print('cv2.__file__')
import sys

print(sys.path)

# Load an color image in grayscale
img = cv2.imread('img_00110.jpg',1)

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
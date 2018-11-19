import numpy as np
import cv2
import time

cap = cv2.VideoCapture('output.avi')

while(cap.isOpened()):
    ret, frame = cap.read()

    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    time.sleep(0.1)

cap.release()
cv2.destroyAllWindows()
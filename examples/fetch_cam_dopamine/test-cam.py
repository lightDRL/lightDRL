import cv2
import getch
import time

cap = cv2.VideoCapture(0)
# print('CV_CAP_PROP_BUFFERSIZE=', cap.get(cv2.CV_CAP_PROP_BUFFERSIZE) ) # internal buffer will now store only 3 frames
# cap.set(cv2.CV_CAP_PROP_BUFFERSIZE, 3)
print(cap.isOpened())

SAVE_COUNT = 0


def save_pic():
  global SAVE_COUNT
  SAVE_COUNT+=1
  pic_path = 'cap_%03d.jpg' % SAVE_COUNT
  print('save picture to ->', pic_path)
  
  cv2.imwrite(pic_path, frame)


while(True):
  
  print('wait 2')
  
  time.sleep(2)
  print('ready')
  for i in range(5):
      cap.grab()

  ret, frame = cap.read()

  cv2.imshow('frame', frame)
  keycode = cv2.waitKey(1)
  print(keycode)

  print('wait')
  time.sleep(3)
  
  # char = getch.getch() 
    
  if keycode== ord('q'):
    break

  if keycode==ord('c'):
    save_pic()

    
cap.release()

cv2.destroyAllWindows()
import cv2
import getch


cap = cv2.VideoCapture(0)

print(cap.isOpened())

SAVE_COUNT = 0


def save_pic():
  global SAVE_COUNT
  SAVE_COUNT+=1
  pic_path = 'cap_%03d.jpg' % SAVE_COUNT
  print('save picture to ->', pic_path)
  
  cv2.imwrite(pic_path, frame)


while(True):

  ret, frame = cap.read()

  cv2.imshow('frame', frame)


  keycode = cv2.waitKey(1)
  print(keycode)
  # char = getch.getch() 
    
  if keycode== ord('q'):
    break

  if keycode==ord('c'):
    save_pic()

    
cap.release()

cv2.destroyAllWindows()
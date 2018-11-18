import cv2
import numpy as np

cap = cv2.VideoCapture(0)



def nothing(x):
    pass

h,s,v = 100,100,100
cv2.namedWindow('result')
cv2.createTrackbar('h', 'result',0,179,nothing)
cv2.createTrackbar('s', 'result',211,255,nothing)
cv2.createTrackbar('v', 'result',100,255,nothing)

cv2.createTrackbar('h_u', 'result',5,179,nothing)
cv2.createTrackbar('s_u', 'result',255,255,nothing)
cv2.createTrackbar('v_u', 'result',255,255,nothing)


def one_hsv_2_bgr(h, s,v ):
    hsv_one_color = np.uint8([[[h,s,v ]]]) 
    hsv2bgr = cv2.cvtColor(hsv_one_color, cv2.COLOR_HSV2BGR)
    return hsv2bgr[0][0]

def one_hsv_2_bgr_show(title, h, s,v ):
    bgr = one_hsv_2_bgr(h, s , v)
    rgb = bgr[::-1]
    rgb_ratio = rgb/255
    print(title, ' h,s,v = (%d, %d, %d)' %(h,s,v) ,' hsv2bgr = ', bgr,', rgb=',rgb,',rgb_ratio=', rgb_ratio)
    

def hsv_and_mask(img,lower_hsv, upper_hsv,  show_id):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask)

    cv2.imshow('frame%02' % show_id, img)
    cv2.imshow('mask%02' % show_id,mask)
    cv2.imshow('res%02' % show_id,res)

while(1):

    # get info from track bar and appy to result
    h = cv2.getTrackbarPos('h','result')
    s = cv2.getTrackbarPos('s','result')
    v = cv2.getTrackbarPos('v','result')

    # get info from track bar and appy to result
    h_u = cv2.getTrackbarPos('h_u','result')
    s_u = cv2.getTrackbarPos('s_u','result')
    v_u = cv2.getTrackbarPos('v_u','result')

    # Take each frame
    # _, frame = cap.read()
    frame   = cv2.imread('sim_pic/001.jpg')
    frame_2 = cv2.imread('sim_pic/028.jpg')

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Normal masking algorithm
    lower_hsv = np.array([h,s,v])
    upper_hsv = np.array([h_u,s_u,v_u])

    print('h: [{}-{}], s: [{}-{}], v: [{}-{}]'.format(h,h_u, s, s_u, v,v_u))

    # h_2, s_2, v_2 = h+ (h_u-h)* 0.5, s+ (s_u-s)* 0.5, v+ (v_u-v)* 0.5
    # print('lower h,s,v = (%d, %d, %d)' %(h,s,v) ,', hsv2bgr = ', one_hsv_2_bgr(h, s, v))
    # print('upper h,s,v = (%d, %d, %d)' %(h_u,s_u,v_u) ,' hsv2bgr = ', one_hsv_2_bgr(h_u, s_u , v_u))
    # print('half h,s,v = (%d, %d, %d)' %(h_2,s_2,v_2) ,' hsv2bgr = ', one_hsv_2_bgr(h_2, s_2 , v_2))
    # one_hsv_2_bgr_show('half',h_2,s_2,v_2)
    # hsv2bgr =  [[[14  7 41]]]

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask)

    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
# import numpy as np
import cv2


print('cv2.__file__')
import sys

print(sys.path)

to_gray = True
IMG_W_H = 84
# Load an color image in grayscale
# img = cv2.imread('sim/camenv_run_pic_00/010.jpg',1)

# dir_name='real/tmp_img_path'
# f_name = '1542085723.8116562'

# dir_name='sim/camenv_run_pic_0.7'
# f_name = '016'

dir_name='sim/sim_01'
f_name = '006'

img = cv2.imread('%s/%s.jpg' % (dir_name, f_name))
resize_img = cv2.resize(img, (IMG_W_H, IMG_W_H), interpolation=cv2.INTER_AREA)
cv2.imwrite("%s.jpg" % f_name, resize_img)
if to_gray:
    gray_img = cv2.cvtColor(resize_img, cv2.COLOR_RGB2GRAY)
else:
    gray_img = resize_img

print('img.shpae = ', img.shape)
print('resize_img.shpae = ', resize_img.shape)
print('gray_img.shpae = ', gray_img.shape)


def to_digit(gray_img, f_head_name, use_threshold = True):
    fo = open("%s.txt" % f_head_name, "w")
    for i in range(gray_img.shape[0]):
        fo.write('%02d:' % i )
        for j in range(gray_img.shape[1]):
            data = 0 if gray_img[i][j]>100 and use_threshold else gray_img[i][j] 
            fo.write('%03d ' % data )

        fo.write('\n' )
    fo.write( str(gray_img) )
    fo.close()

to_digit(gray_img , f_name, False)
to_digit(gray_img , f_name + '_threshold', True)


cv2.imwrite("%s_gray.jpg" % f_name, gray_img)
cv2.imshow('image',gray_img)
c = cv2.waitKey(0)

cv2.destroyAllWindows()
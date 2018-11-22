def to_digit(gray_img, f_head_name):
    fo = open("%s.txt" % f_head_name, "w")
    for i in range(gray_img.shape[0]):
        fo.write('%02d:' % i )
        for j in range(gray_img.shape[1]):
            if len(gray_img.shape)==3:
                fo.write('[%3d,%3d,%3d] ' % ( gray_img[i][j][0], gray_img[i][j][1], gray_img[i][j][2]) )
            else:
                fo.write('%2d ' % gray_img[i][j] )

        fo.write('\n' )
    fo.write( str(gray_img) )
    fo.close()


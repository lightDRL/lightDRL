from multiprocessing import Process, Value, Lock, Manager, Pipe
import sys, os
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__) )+ '/../'))
from semantic.segnet_label import SegnetLabel
import time
import cv2
import numpy as np

USE_TIEM_SUFFIX_DIR =  False

def semantic_process_func(child_conn, d):
    abs_dir = os.path.abspath(os.path.dirname(__file__))
    weight_path = os.path.abspath( abs_dir + '/../semantic/weights/3obj')
    print('Load weight_path  ->', weight_path)
    # s = SegnetLabel( n_classes=2, input_height=224, input_width=224, save_weights_path=weight_path, epoch_number=10 )
    s = SegnetLabel( n_classes=4, input_height=224, input_width=224, save_weights_path=weight_path, epoch_number=5 )
    

    suffix = '_' + time.strftime("%y%b%d_%H%M%S") if USE_TIEM_SUFFIX_DIR else ''
    predict_dir = 'semantic_annotate' + suffix
    predict_dir  = os.path.abspath(predict_dir)
    if os.path.exists(predict_dir):
        import shutil
        shutil.rmtree(predict_dir)
    print("Make dir: " + predict_dir)
    os.makedirs(predict_dir)
    

    img_predict_id = 0

    child_conn.send('ready')

    while True:
        input_path = child_conn.recv()
        # output = s.predict(d['input_path'])
        output = s.predict(input_path)

        # with lock:
        seg_img = output*255.0
        output_prefix = '{}/{:04d}'.format(predict_dir, img_predict_id)
        output_path = output_prefix + '.png' 
        # print('Annotate finish, save to ' + output_path)

        cv2.imwrite( output_path, output )
        cv2.imwrite( output_prefix + '_show.jpg' , seg_img )

        child_conn.send( output_path)

        img_predict_id = img_predict_id + 1 if img_predict_id<1000 else 0


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

if __name__ == '__main__':
    import argparse
    import cv2
    parent_conn, child_conn = Pipe()
    manager = Manager()

    p = Process(target=semantic_process_func, args=(child_conn, manager.dict()   )   )
    p.daemon = True
    p.start()

    predict_sum_time = 0
    if parent_conn.recv()=='ready':
        for i in range(1000):
            #-----test.png----#
            img_path = 'test.png' 
            img_path = os.path.abspath(img_path)

            # s_time = time.time()
            parent_conn.send(img_path)
            get_path = parent_conn.recv()
            annot_img = cv2.imread(get_path, 1)
            # predict_use_time = time.time()-s_time
            # print('Use time:  {}'.format( predict_use_time ) )
            # predict_sum_time+= predict_use_tim
            # cv2.imshow('annot', annot_img*255.0 )
            to_digit(annot_img , get_path + '_ori_digit')

            reshape_annot_img =  annot_img[:,:,0]#np.reshape(annot_img, (annot_img.shape[0], annot_img.shape[1]))
            to_digit(reshape_annot_img , get_path + '_reshape_digit')
            
            cv2.imwrite(  'test_annot.png',  annot_img*255.0 )

            # cv2.waitKey(1000)


    # print('predict_sum_time = ', predict_sum_time)
    # print('predict_sum_time average  = ', predict_sum_time/1000)
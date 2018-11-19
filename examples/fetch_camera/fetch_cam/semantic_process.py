from multiprocessing import Process, Value, Lock, Manager, Pipe
import sys, os
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__) )+ '/../'))
from semantic.segnet_label import SegnetLabel
import time

def process_func(child_conn, d):
    abs_dir = os.path.abspath(os.path.dirname(__file__))
    weight_path = os.path.abspath( abs_dir + '/../semantic/weights/ex1')
    print('weight_path  ->', weight_path)
    s = SegnetLabel( n_classes=2, input_height=224, input_width=224, save_weights_path=weight_path, epoch_number=10 )
    
    suffix = time.strftime("%y%b%d_%H%M%S")
    predict_dir = 'semantic_annotate_' + suffix
    predict_dir  = os.path.abspath(predict_dir)
    os.makedirs(predict_dir)
    print("Make dir: " + predict_dir)

    img_predict_id = 0

    child_conn.send('ready')

    while True:
        input_path = child_conn.recv()
        # output = s.predict(d['input_path'])
        output = s.predict(input_path)

        # with lock:
        seg_img = output*255.0
        output_prefix = '{}/{:03d}'.format(predict_dir, img_predict_id)
        output_path = output_prefix + '.png' 
        print('Annotate finish, save to ' + output_path)

        cv2.imwrite( output_prefix + '.png' , output )
        cv2.imwrite( output_prefix + '_show.jpg' , seg_img )

        child_conn.send( output_path)

        img_predict_id+=1


if __name__ == '__main__':
    import argparse
    import cv2
    parent_conn, child_conn = Pipe()
    manager = Manager()

    p = Process(target=process_func, args=(child_conn, manager.dict()   )   )
    p.daemon = True
    p.start()

    if parent_conn.recv()=='ready':
        #-----test.png----#
        img_path = 'test.png' 
        img_path = os.path.abspath(img_path)
        parent_conn.send(img_path)
        get_path = parent_conn.recv()

        annot_img = cv2.imread(get_path, 1)
        cv2.imshow('annot', annot_img*255.0 )
        cv2.imwrite(  'test_annot.png',  annot_img*255.0 )

        cv2.waitKey(1000)
import os
import cv2
import numpy as np 

DEFALUT_IMG_W_H = 84

class IMG_TYPE:
    RAW = 0 
    GRAY = 1
    BIN = 2
    SEMANTIC = 3 
    RGB = 4

class IMG_SHOW:
    HIDE=0
    RAW=1
    PROCESS=2
    RAW_PROCESS=3

class ImgProcess:
    _show_type = IMG_SHOW.HIDE
    _img_w_h = DEFALUT_IMG_W_H
    def __init__(self,img_type = IMG_TYPE.RGB, flip = True):
        self.img_type = img_type
        self.flip = flip

        if  self.img_type==IMG_TYPE.SEMANTIC:
            self.semantic_process_init()
 
    @property
    def show_type(self):
        return self._show_type

    @show_type.setter
    def show_type(self, show_type):
        self._show_type = show_type

    def semantic_process_init(self):
        from fetch_cam.semantic_process import semantic_process_func, USE_TIEM_SUFFIX_DIR
        from multiprocessing import Process, Manager, Pipe
        import time
        self.parent_conn, child_conn = Pipe()
        manager = Manager()

        p = Process(target=semantic_process_func, args=(child_conn, manager.dict()   )   )
        p.daemon = True
        p.start()

        # wait init finish, NOTE: BLOCK here!!
        if self.parent_conn.recv()=='ready':
            # prepare dir for image save
            suffix = '_' + time.strftime("%y%b%d_%H%M%S") if USE_TIEM_SUFFIX_DIR else ''
            img_dir = 'semantic_raw_img' + suffix
            self.semantic_img_dir  = os.path.abspath(img_dir)
            if os.path.exists(self.semantic_img_dir):
                import shutil
                shutil.rmtree(self.semantic_img_dir)
            os.makedirs(self.semantic_img_dir )

            self.semantic_img_id = 0
        else:
            print("Strange recv() return")

    def preprocess(self, rgb_gripper):
        show_type = self._show_type
        if self.flip:
            rgb_gripper = cv2.flip(rgb_gripper,0)
        if show_type==IMG_SHOW.RAW or show_type==IMG_SHOW.RAW_PROCESS:
            # show_img = cv2.flip(rgb_gripper,0)
            cv2.imshow('Raw Image',rgb_gripper)
            cv2.waitKey(10)

        
        if self.img_type == IMG_TYPE.RGB or self.img_type == IMG_TYPE.RAW:
            process_img =  rgb_gripper
        elif self.img_type == IMG_TYPE.GRAY:
            process_img =  cv2.cvtColor(rgb_gripper, cv2.COLOR_RGB2GRAY)
        elif self.img_type== IMG_TYPE.BIN:
            lower_hsv = np.array([0,211,100])
            upper_hsv = np.array([5,255,255])
            hsv = cv2.cvtColor(rgb_gripper, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
            mask[mask < 200] = 0
            mask[mask >= 200] = 1
            # print('mask.shape = ', mask.shape)
            # res = cv2.bitwise_and(rgb_gripper,rgb_gripper, mask= mask)
            # print(res)
            process_img = mask
        elif self.img_type== IMG_TYPE.SEMANTIC:
            img_path = '{}/{:04d}.png'.format(self.semantic_img_dir, self.semantic_img_id)
            cv2.imwrite(img_path, rgb_gripper)
            self.parent_conn.send(img_path)
            # print('img_process img_raw = ', img_path)
            # NOTE: block here
            get_annot_path = self.parent_conn.recv()
            # print('img_process get_annot_path = ', img_path)

            self.semantic_img_id = self.semantic_img_id + 1  if self.semantic_img_id < 1000 else 0
            process_img = cv2.imread(get_annot_path, 1)
            process_img= process_img[:,:,0]

            # print('process_img.shape = ', process_img.shape)
        
        else:
            print("FetchDiscreteCamEnv step() image process say STRANGE img_type =  ", self.img_type)

        process_img = cv2.resize(process_img, (self._img_w_h, self._img_w_h), interpolation=cv2.INTER_AREA)

        
        if show_type==IMG_SHOW.PROCESS or show_type==IMG_SHOW.RAW_PROCESS:
            if self.img_type== IMG_TYPE.BIN or self.img_type== IMG_TYPE.SEMANTIC:
                show_img = process_img*255.0
            else:
                show_img = process_img
            # show_img = cv2.flip(show_img,0)
            cv2.imshow('Process Image',show_img)
            cv2.waitKey(10)
        
        # print('process_img.shape = ', process_img.shape)
        if self.img_type == IMG_TYPE.RAW: 
            return rgb_gripper
        else:
            return process_img
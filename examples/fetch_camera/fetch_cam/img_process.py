import os
import cv2
import numpy as np 
import time

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


BoarderSize = 3
BoarderColor = [200., 200., 200.]  # [B, G, R]

class MergeImage:
    def __init__(self, height, width):
        self.tar_img = np.zeros((height,width,3), np.uint8)
        self.tar_img[:,:] = BoarderColor    # set background

        # video
        video_name = time.strftime("%y%b%d_%H%M%S") + '.avi'
        print("Save to video: ", video_name)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out = cv2.VideoWriter(video_name,fourcc, 20.0, (width,height))

        # img 5
        self.img_5 = np.zeros((200,200,3), np.uint8)
        self.img_5[:,:] = [255, 255,255]    # set background
        self.pre_load_arrow_pic()


    def place_img(self,src_img, x, y, resize_h, resize_w, board_h = BoarderSize, board_w = BoarderSize):
        # print('src_img.shape = ', src_img.shape)
        tar_img = self.tar_img
        if len(src_img.shape) < 3:
            # brocast to 3 channles
            place_img = cv2.cvtColor(src_img, cv2.COLOR_GRAY2BGR)        
        else:
            place_img = src_img
        resize_img = cv2.resize(place_img,  (resize_h, resize_w), interpolation=cv2.INTER_AREA)       
        x_end = x+resize_h
        y_end = y+resize_w

        tar_img[x:x_end, y:y_end] = resize_img

        # set boarder line
        tar_img[x:x+board_h        ,y:y_end] = BoarderColor
        tar_img[x_end-board_h:x_end,y:y_end] = BoarderColor            
        tar_img[x:x_end,y:y+board_w]         = BoarderColor
        tar_img[x:x_end,y_end-board_w:y_end] = BoarderColor 
    
    def merge(self, img_1, img_2,img_3, img_4, img_5 = None):
        self.place_img(img_1, 0  , 0   , 300,300  )
        self.place_img(img_2, 300, 0   , 300,300  )
        self.place_img(img_3, 0  , 350 , 200,200)
        self.place_img(img_4, 200, 350 , 200,200)

        if img_5==None:
            self.img_5[:,:] = [255, 255,255] 
            img_5 = self.img_5
        
        self.place_img(img_5, 400, 350 , 200,200)

    def show(self, wait_time = 20):
        cv2.imshow('MergeWindow', self.tar_img)
        cv2.waitKey(wait_time)

    def save_2_video(self):
        self.out.write(self.tar_img)

    def release(self):
        self.out.release()

    def pre_load_arrow_pic(self):
        arrow_pic_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'arrow_pic')
        self.arrow_up   = cv2.imread(arrow_pic_dir + '/up.png')
        self.arrow_left = cv2.imread(arrow_pic_dir + '/left.png')
        self.arrow_down = cv2.imread(arrow_pic_dir + '/down.png')
        self.arrow_right= cv2.imread(arrow_pic_dir + '/right.png')
        self.arrow_pick = cv2.imread(arrow_pic_dir + '/pick.png')


    def show_arrow(self, action, wait_time=20):
        if action==0:
            img_5 = self.arrow_up
        elif action==1:
            img_5 = self.arrow_left
        elif action==2:
            img_5 = self.arrow_down
        elif action==3:
            img_5 = self.arrow_right
        elif action==4:
            img_5 = self.arrow_pick

        self.place_img(img_5, 400, 350 , 200,200)
        
        self.show(wait_time)

    
# merge_img = MergeImage(height = 600, width = 600)

if __name__ == '__main__':
    cap0 = cv2.VideoCapture(0)
    cap1 = cv2.VideoCapture(1)

    merge_img = MergeImage(height = 600, width = 600)

    while(True):

        
        ret, frame1 = cap1.read()
        ret, frame0 = cap0.read()

        img_1 = frame0
        img_2 = frame1
        img_3 = cv2.cvtColor(frame0, cv2.COLOR_RGB2GRAY)  # gray
        img_4 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)  # gray
        # img_5 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)  # gray

        
        merge_img.merge(img_1, img_2, img_3, img_4)
        merge_img.save_2_video()
        merge_img.show(10)

        keycode = cv2.waitKey(10)

        randint = np.random.randint(0,5)
        merge_img.show_arrow(randint)
        # print(keycode)
        # char = getch.getch() 

        if keycode== ord('q'):
            break

    # Release everything if job is finished
    merge_img.release()
    cap0.release()
    cap1.release()
    cv2.destroyAllWindows()
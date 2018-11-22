import cv2
import numpy as np
import os,sys

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__) )+ '/../'))
from fetch_cam.img_process import ImgProcess, IMG_TYPE, IMG_SHOW

IMG_PATH = os.path.abspath(os.path.dirname(__file__))

# because thread bloack the image catch (maybe), so create the shell class 
class FeedImgEnv:
    def __init__(self, step_pixel=5, img_type = IMG_TYPE.BIN):
        self.step_pixel =step_pixel
        # Load an color image

        
        self.table = cv2.imread(IMG_PATH + '/table.png')
        self.cube = cv2.imread(IMG_PATH + '/cube_shadow.png')

        self.imp = ImgProcess(img_type)
        self.imp.show_type = IMG_SHOW.RAW_PROCESS

        print('!!!!!!!!!!------FeedImgEnv init fniish----------!!!!!!!!!!!')
        # print("self.table.shape = ", self.table.shape)
        # print("self.cube.shape = ", self.cube.shape)

    def reset(self):
        print('in reset')
        # self.state = cv2.imread('img_00110.jpg')
        self.state = self.table.copy()
        self.cube_img_x, self.cube_img_y = np.random.randint(50,400), np.random.randint(50,400)
        self.rotate_deg =  np.random.randint(0, 90)
        self.scale = np.random.uniform(0.8,1.2)
        # for i in range(self.state.shape[0]):
        #     for j in range(self.state.shape[1]):
        #         self.state[i][j] = 
        # self.tmp = self.state[self.cube_img_x:self.cube_img_x+self.cube.shape[0], self.cube_img_y:self.cube_img_y+self.cube.shape[1], :] 
        # print('self.tmp.shape = ', self.tmp.shape)
        # self.tmp = self.cube
        # self.state[self.cube_img_x:self.cube_img_x+self.cube.shape[0], self.cube_img_y:self.cube_img_y+self.cube.shape[1], :]  = self.cube[:,:,:] 
        self.paste_cube(self.cube_img_x, self.cube_img_y)

        s = self.imp.preprocess(self.state)

        # if self.is_render:
        #     self.render()
        return s

    # def paste_cube(self, cube_img_x, cube_img_y):
        
    #     cube_h = self.cube.shape[0]
    #     cube_w = self.cube.shape[1]
    #     self.state = self.table.copy()
    #     if cube_img_x >= self.table.shape[0] or cube_img_y >= self.table.shape[1] or \
    #        (cube_img_x+cube_h) < 0 or  (cube_img_y+cube_w) < 0 :
    #         return 

        
    #     paste_low_h =  -cube_img_x  if cube_img_x < 0 else 0
    #     paste_low_w =  -cube_img_y  if cube_img_y < 0 else 0
    #     cube_img_x = 0 if cube_img_x < 0 else cube_img_x
    #     cube_img_y = 0 if cube_img_y < 0 else cube_img_y
    #     paste_h = self.cube.shape[0] if (cube_img_x +  self.cube.shape[0]) <= self.table.shape[0] else ( self.table.shape[0]- cube_img_x)  
    #     paste_w = self.cube.shape[1] if (cube_img_y +  self.cube.shape[1]) <= self.table.shape[1] else ( self.table.shape[1]- cube_img_y)  

    #     # paste_h -= paste_low_h
    #     # paste_w -= paste_low_w
    #     # print('--------------')
    #     # print('self.table.shape =', self.table.shape)
    #     # print('self.cube.shape =', self.cube.shape)

    #     # print('paste cube_img_x = {}, cube_img_y={}'.format(cube_img_x, cube_img_y))
        

    #     # print('paste paste_h = {}, paste_w={}'.format(paste_h, paste_w))
    #     # print('paste paste_low_h = {}, paste_low_w={}'.format(paste_low_h, paste_low_w))

        
    #     self.state[cube_img_x:cube_img_x+(paste_h-paste_low_h), cube_img_y:cube_img_y + (paste_w-paste_low_w), :]  = self.cube[paste_low_h:paste_h,paste_low_w:paste_w,:] #self.cube[:,:,:] 
    def paste_cube(self, cube_img_x, cube_img_y, rotate_deg= 30, scale= 1):
        
        cube_new = self.cube.copy()

        cube_new = cv2.resize(cube_new, \
                            (int(cube_new.shape[0]*scale),int(cube_new.shape[1]*scale)), \
                            interpolation=cv2.INTER_AREA)        
        rows = cube_new.shape[0]
        cols = cube_new.shape[1]
        M = cv2.getRotationMatrix2D((cols/2,rows/2),rotate_deg,1)
        cube_new = cv2.warpAffine(cube_new,M,(cols,rows))

        
        # print('self.cube.shape = ', self.cube.shape)
        # print('cube_new.shape = ', cube_new.shape)

        cube_h = cube_new.shape[0]
        cube_w = cube_new.shape[1]
        self.state = self.table.copy()
        if cube_img_x >= self.table.shape[0] or cube_img_y >= self.table.shape[1] or \
           (cube_img_x+cube_h) < 0 or  (cube_img_y+cube_w) < 0 :
            return 
        
        paste_low_h =  -cube_img_x  if cube_img_x < 0 else 0
        paste_low_w =  -cube_img_y  if cube_img_y < 0 else 0
        cube_img_x = 0 if cube_img_x < 0 else cube_img_x
        cube_img_y = 0 if cube_img_y < 0 else cube_img_y
        paste_h = cube_new.shape[0] if (cube_img_x +  cube_new.shape[0]) <= self.table.shape[0] else ( self.table.shape[0]- cube_img_x)  
        paste_w = cube_new.shape[1] if (cube_img_y +  cube_new.shape[1]) <= self.table.shape[1] else ( self.table.shape[1]- cube_img_y)  

        s_min_x = cube_img_x
        s_max_x = cube_img_x+(paste_h-paste_low_h)
        s_min_y = cube_img_y
        s_max_y = cube_img_y + (paste_w-paste_low_w)
        self.state[s_min_x:s_max_x, s_min_y:s_max_y, :]  = cube_new[paste_low_h:paste_h,paste_low_w:paste_w,:] #self.cube[:,:,:] 

        for i in range(s_min_x, s_max_x):
            for j in range(s_min_y, s_max_y):
                if sum(self.state[i][j]) == 0:
                    self.state[i,j,:] = self.table[i,j,:] 

        # self.label = self.table_label.copy()
        # self.label[cube_img_x:cube_img_x+(paste_h-paste_low_h), cube_img_y:cube_img_y + (paste_w-paste_low_w), :]  =self.cube_label[paste_low_h:paste_h,paste_low_w:paste_w,:] #self.cube[:,:,:] 
        

    def step(self,action):
        d = False
        r = 0

        # print('Use action: ', action)
        if action==4:
            print("-------------USE 4-----------------")
        if action ==0:
            self.cube_img_x += self.step_pixel
        elif action == 1:
            self.cube_img_y += self.step_pixel
        elif action ==2:
            self.cube_img_x -= self.step_pixel
        elif action == 3:
            self.cube_img_y -= self.step_pixel
        elif action == 4:
            d = True
        
        # print('cube_img_x = {}, cube_img_y={}'.format(self.cube_img_x, self.cube_img_y))
        
        self.paste_cube(self.cube_img_x, self.cube_img_y, self.rotate_deg, self.scale)

        s = self.imp.preprocess(self.state)
        # if self.is_render:
        #     self.render()
        return s, r, d, None

    def render(self):
        pass
        # # rgb_img = cv2.cvtColor(gripper_img, cv2.COLOR_BGR2RGB)
        # cv2.imshow('Feed Image Env',self.state)
        # cv2.waitKey(50)


if __name__ == '__main__':
    env = FeedImgEnv(img_type=IMG_TYPE.BIN) #BIN)
    env.reset()
    for i in range(60):
        s, r, d , _ = env.step(0)
        # print('s.shape = ', s.shape)
        # cv2.imshow('Feed Image Env', s)
        # cv2.waitKey(50)

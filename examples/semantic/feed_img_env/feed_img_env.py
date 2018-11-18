import cv2
import numpy as np
import os

IMG_PATH = os.path.abspath(os.path.dirname(__file__))
IMG_W_H = 84
# because thread bloack the image catch (maybe), so create the shell class 
class FeedImgEnv:
    def __init__(self, step_ds=0.005, gray_img = True, is_render = False):
        self.gray_img = gray_img
        self.is_render = is_render
        self.step_ds =step_ds
        # Load an color image

        
        self.table = cv2.imread(IMG_PATH + '/table.png')
        self.cube = cv2.imread(IMG_PATH + '/cube_shadow.png')

        # print("self.table.shape = ", self.table.shape)
        # print("self.cube.shape = ", self.cube.shape)

    def reset(self):
        # self.state = cv2.imread('img_00110.jpg')
        self.state = self.table.copy()
        self.cube_img_x, self.cube_img_y = 200, 200
        # for i in range(self.state.shape[0]):
        #     for j in range(self.state.shape[1]):
        #         self.state[i][j] = 
        # self.tmp = self.state[self.cube_img_x:self.cube_img_x+self.cube.shape[0], self.cube_img_y:self.cube_img_y+self.cube.shape[1], :] 
        # print('self.tmp.shape = ', self.tmp.shape)
        # self.tmp = self.cube
        # self.state[self.cube_img_x:self.cube_img_x+self.cube.shape[0], self.cube_img_y:self.cube_img_y+self.cube.shape[1], :]  = self.cube[:,:,:] 
        self.paste_cube(self.cube_img_x, self.cube_img_y)

        s = self.state_preprocess(self.state)

        if self.is_render:
            self.render()
        return s

    def paste_cube(self, cube_img_x, cube_img_y):
        
        cube_h = self.cube.shape[0]
        cube_w = self.cube.shape[1]
        self.state = self.table.copy()
        if cube_img_x >= self.table.shape[0] or cube_img_y >= self.table.shape[1] or \
           (cube_img_x+cube_h) < 0 or  (cube_img_y+cube_w) < 0 :
            return 

        
        paste_low_h =  -cube_img_x  if cube_img_x < 0 else 0
        paste_low_w =  -cube_img_y  if cube_img_y < 0 else 0
        cube_img_x = 0 if cube_img_x < 0 else cube_img_x
        cube_img_y = 0 if cube_img_y < 0 else cube_img_y
        paste_h = self.cube.shape[0] if (cube_img_x +  self.cube.shape[0]) <= self.table.shape[0] else ( self.table.shape[0]- cube_img_x)  
        paste_w = self.cube.shape[1] if (cube_img_y +  self.cube.shape[1]) <= self.table.shape[1] else ( self.table.shape[1]- cube_img_y)  

        # paste_h -= paste_low_h
        # paste_w -= paste_low_w
        # print('--------------')
        # print('self.table.shape =', self.table.shape)
        # print('self.cube.shape =', self.cube.shape)

        # print('paste cube_img_x = {}, cube_img_y={}'.format(cube_img_x, cube_img_y))
        

        # print('paste paste_h = {}, paste_w={}'.format(paste_h, paste_w))
        # print('paste paste_low_h = {}, paste_low_w={}'.format(paste_low_h, paste_low_w))

        
        self.state[cube_img_x:cube_img_x+(paste_h-paste_low_h), cube_img_y:cube_img_y + (paste_w-paste_low_w), :]  = self.cube[paste_low_h:paste_h,paste_low_w:paste_w,:] #self.cube[:,:,:] 


    # -> IMG_W_H*IMG_W_H and to gray
    def state_preprocess(self, img):
        if self.gray_img:
            resize_img = cv2.resize(img, (IMG_W_H, IMG_W_H), interpolation=cv2.INTER_AREA)
            gray_img = cv2.cvtColor(resize_img, cv2.COLOR_RGB2GRAY)
            return np.reshape(gray_img,(IMG_W_H,IMG_W_H,1))
        else:
            return img
        

    def step(self,action):
        d = False
        r = 0
        print('Use action: ', action)
        if action ==0:
            self.cube_img_x += 10
        elif action == 1:
            self.cube_img_y += 10
        elif action ==2:
            self.cube_img_x -= 10
        elif action == 3:
            self.cube_img_y -= 10
        elif action == 4:
            d = True
        
        # print('cube_img_x = {}, cube_img_y={}'.format(self.cube_img_x, self.cube_img_y))
        self.paste_cube(self.cube_img_x, self.cube_img_y)

        s = self.state_preprocess(self.state)
        if self.is_render:
            self.render()
        return s, r, d, None

    def render(self):
        # rgb_img = cv2.cvtColor(gripper_img, cv2.COLOR_BGR2RGB)
        cv2.imshow('Feed Image Env',self.state)
        cv2.waitKey(50)


if __name__ == '__main__':
    env = FeedImgEnv(gray_img=False)
    env.reset()
    for i in range(60):
        s, r, d , _ = env.step(3)
        # print('s.shape = ', s.shape)
        cv2.imshow('Feed Image Env', s)
        cv2.waitKey(50)


from fetch_cam import FetchDiscreteEnv
import cv2
import numpy as np
import time
import tensorflow as tf
# because thread bloack the image catch (maybe), so create the shell class 
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
from matplotlib import pyplot as plt

IMG_W_H = 84
class FetchDiscreteCamSiamenseEnv:
    def __init__(self, dis_tolerance = 0.001, step_ds=0.005, gray_img = True, hsv_color = False, is_render=False):
        self.env = FetchDiscreteEnv(dis_tolerance = 0.001, step_ds=0.005, is_render = is_render)
        self.gray_img = gray_img

        self.target_pic = None
        self.hsv_color = hsv_color
        self.is_render = is_render

    def img_preprocess(self, img):
        # not support gray image
        
        if self.hsv_color:
            resize_img = cv2.resize(img, (IMG_W_H, IMG_W_H), interpolation=cv2.INTER_AREA)
            resize_img = resize_img/255.0
            # print('resize_img = ', resize_img)
            hsv_img = rgb_to_hsv(resize_img)
            return hsv_img
        else:            
            resize_img = cv2.resize(img, (IMG_W_H, IMG_W_H), interpolation=cv2.INTER_AREA)
            return resize_img
        
    def step(self,action):
        # print('i action = ', action)
        a_one_hot = np.zeros(5)
        a_one_hot[action] = 1
        s_old, r, d, _ = self.env.step(a_one_hot)

        # no use, but you need preserve it; otherwise, you will get error image
        rgb_external = self.env.sim.render(width=256, height=256, camera_name="external_camera_0", depth=False,
                    mode='offscreen', device_id=-1)
        rgb_gripper = self.env.sim.render(width=256, height=256, camera_name="gripper_camera_rgb", depth=False,
            mode='offscreen', device_id=-1)

        if self.is_render:
            self.render_gripper_img(rgb_gripper)
            # rgb_img = cv2.cvtColor(rgb_gripper, cv2.COLOR_BGR2RGB)
            # cv2.imshow('Gripper Image',rgb_img)
            # cv2.waitKey(50)
            

        s = [self.img_preprocess(rgb_gripper), self.target_pic]
        return s, r, d, None


    @property
    def pos(self):
        return self.env.pos

    @property
    def obj_pos(self):
        return self.env.obj_pos

    def get_obj_pos(self, obj_id):
        return self.env.get_obj_pos(obj_id)

    @property
    def gripper_state(self):
        return self.env.gripper_state


    def take_only_obj0_pic(self):
        
        try:
            self.env.rand_obj0_hide_obj1_obj2()
            self.env.render()
            # time.sleep(2)
            rgb_external = self.env.sim.render(width=256, height=256, camera_name="external_camera_0", depth=False,
                        mode='offscreen', device_id=-1)
            rgb_gripper = self.env.sim.render(width=256, height=256, camera_name="gripper_camera_rgb", depth=False,
                mode='offscreen', device_id=-1)

            if self.is_render :
                self.render_target_img(rgb_gripper)
                self.render_gripper_img(rgb_gripper)
                
                time.sleep(2)

            # time.sleep(2)
            # resize_img = cv2.resize(rgb_gripper, (IMG_W_H, IMG_W_H), interpolation=cv2.INTER_AREA)
            # self.target_pic = resize_img.copy()
            self.target_pic =  self.img_preprocess(rgb_gripper)

            self.env.recover_obj0_obj1_obj2_pos()
            self.env.render()
            # time.sleep(2)
        except Exception as e:
            print(' Exception e -> ', e )
            pass
            # print(' Exception e -> ', e )
        

    def reset(self):
        if self.hsv_color:
            self.env.rand_objs_hsv()
        else:
            self.env.rand_objs_color()

        self.env.reset()
        self.take_only_obj0_pic()
        

        self.env.render()
        rgb_external = self.env.sim.render(width=256, height=256, camera_name="external_camera_0", depth=False,
                    mode='offscreen', device_id=-1)
        rgb_gripper = self.env.sim.render(width=256, height=256, camera_name="gripper_camera_rgb", depth=False,
            mode='offscreen', device_id=-1)
    
        if self.is_render :
            self.render_gripper_img(rgb_gripper)
            time.sleep(1)


        s = [self.img_preprocess(rgb_gripper), self.target_pic]
        return s
        

    def render(self):
        self.env.render()

    def render_target_img(self, img):
        # if self.is_render:
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imshow('Target Image',rgb_img)
        cv2.waitKey(50)

    def render_gripper_img(self, gripper_img):
        # if self.is_render:
        rgb_img = cv2.cvtColor(gripper_img, cv2.COLOR_BGR2RGB)
        cv2.imshow('Gripper Image',rgb_img)
        cv2.waitKey(50)


        # plt.figure(1)
        # plt.imshow(rgb_gripper)
        # plt.show(block=False)
        # plt.pause(0.001)
        # time.sleep(2)


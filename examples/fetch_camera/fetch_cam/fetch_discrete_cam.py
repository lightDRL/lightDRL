
from fetch_cam import FetchDiscreteEnv
import cv2
import numpy as np


IMG_W_H = 84

class IMG_TYPE:
    RGB = 0 
    GRAY = 1
    BIN = 2
    SEMANTIC = 3 
# because thread bloack the image catch (maybe), so create the shell class 
class FetchDiscreteCamEnv:
    def __init__(self, dis_tolerance = 0.001, step_ds=0.005, img_type = IMG_TYPE.RGB, use_tray = True, is_render = False, only_show_obj0=False):
        self.env = FetchDiscreteEnv(dis_tolerance = 0.001, step_ds=0.005, use_tray=use_tray, is_render = is_render)
        self.img_type = img_type
        self.is_render = is_render
        self.only_show_obj0 = only_show_obj0
        
    def state_preprocess(self, rgb_gripper):
        rgb_gripper =  cv2.cvtColor(rgb_gripper, cv2.COLOR_BGR2RGB)

        if self.img_type == IMG_TYPE.RGB:
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
        else:
            print("FetchDiscreteCamEnv step() image process say STRANGE img_type ")

        # print('process_img.shape = ', process_img.shape)

        return process_img

    def color_state_preprocess(self, img):
        imgNorm='sub_mean'
        if imgNorm == "sub_and_divide":
            img = np.float32(cv2.resize(img, ( IMG_W_H , IMG_W_H ))) / 127.5 - 1
        elif imgNorm == "sub_mean":
            img = cv2.resize(img, ( IMG_W_H , IMG_W_H ))
            img = img.astype(np.float32)
            img[:,:,0] -= 103.939
            img[:,:,1] -= 116.779
            img[:,:,2] -= 123.68
        elif imgNorm == "divide":
            img = cv2.resize(img, ( IMG_W_H , IMG_W_H ))
            img = img.astype(np.float32)
            img = img/255.0

        return img

    def step(self,action):
        # print('i action = ', action)
        a_one_hot = np.zeros(6)
        a_one_hot[action] = 1
        s, r, d, _ = self.env.step(a_one_hot)

        # no use, but you need preserve it; otherwise, you will get error image
        rgb_external = self.env.sim.render(width=256, height=256, camera_name="external_camera_0", depth=False,
                    mode='offscreen', device_id=-1)
        rgb_gripper = self.env.sim.render(width=256, height=256, camera_name="gripper_camera_rgb", depth=False,
            mode='offscreen', device_id=-1)
        
        process_img = self.state_preprocess(rgb_gripper)
        # RESIZE
        resize_img = cv2.resize(process_img, (IMG_W_H, IMG_W_H), interpolation=cv2.INTER_AREA)

        if self.is_render:
            if self.img_type== IMG_TYPE.BIN:
                process_img = process_img*255.0
            self.render_gripper_img(process_img)

        return resize_img, r, d, cv2.cvtColor(rgb_gripper, cv2.COLOR_BGR2RGB)


    @property
    def pos(self):
        return self.env.pos

    @property
    def obj_pos(self):
        return self.env.obj_pos

    @property
    def red_tray_pos(self):
        return self.env.red_tray_pos

    @property
    def gripper_state(self):
        return self.env.gripper_state

    @property
    def is_gripper_close(self):
        return self.env.is_gripper_close

    def reset(self):
        # self.env.rand_objs_color(exclude_obj0 = True)
        # self.env.rand_red_or_not(obj_name='object0', use_red_color=True)
        self.env.rand_red_or_not(obj_name='object1', use_red_color=False)
        self.env.rand_red_or_not(obj_name='object2', use_red_color=False)
        self.env.reset()
        if self.only_show_obj0:
            self.env.hide_obj1_obj2()
        self.env.render()
        rgb_external = self.env.sim.render(width=256, height=256, camera_name="external_camera_0", depth=False,
                    mode='offscreen', device_id=-1)
        rgb_gripper = self.env.sim.render(width=256, height=256, camera_name="gripper_camera_rgb", depth=False,
            mode='offscreen', device_id=-1)
    
        process_img = self.state_preprocess(rgb_gripper)
        # RESIZE
        resize_img = cv2.resize(process_img, (IMG_W_H, IMG_W_H), interpolation=cv2.INTER_AREA)

        if self.is_render:
            self.render_gripper_img(process_img)

        return resize_img

    def render(self):
        self.env.render()

    
    def render_gripper_img(self, img):
        # if self.is_render:
        # rgb_img = cv2.cvtColor(gripper_img, cv2.COLOR_BGR2RGB)
        cv2.imshow('Gripper Image',img)
        cv2.waitKey(50)

from fetch_cam import FetchDiscreteEnv
import cv2
import numpy as np

# because thread bloack the image catch (maybe), so create the shell class 
class FetchDiscreteCamEnv:
    def __init__(self, dis_tolerance = 0.001, step_ds=0.005):
        self.env = FetchDiscreteEnv(dis_tolerance = 0.001, step_ds=0.005)


    def state_preprocess(self, img):
        resize_img = cv2.resize(img, (84, 84), interpolation=cv2.INTER_AREA)
        gray_img = cv2.cvtColor(resize_img, cv2.COLOR_RGB2GRAY)
        # ret, state = cv2.threshold(state,1,255,cv2.THRESH_BINARY)

        # cv2.imwrite('img.jpg', img)
        # cv2.imwrite('resize_img.jpg', resize_img)
        # cv2.imwrite('gray_img.jpg', gray_img)
    
        # print('state',gray_img )
        return np.reshape(gray_img,(84,84,1))


    def step(self,action):
        # print('i action = ', action)
        a_one_hot = np.zeros(5)
        a_one_hot[action] = 1
        s, r, d, _ = self.env.step(a_one_hot)

        # no use, but you need preserve it; otherwise, you will get error image
        rgb_external = self.env.sim.render(width=256, height=256, camera_name="external_camera_0", depth=False,
                    mode='offscreen', device_id=-1)
        rgb_gripper = self.env.sim.render(width=256, height=256, camera_name="gripper_camera_rgb", depth=False,
            mode='offscreen', device_id=-1)

        s = self.state_preprocess(rgb_gripper)


        return s, r, d, None

    @property
    def pos(self):
        return self.env.pos

    @property
    def obj_pos(self):
        return self.env.obj_pos

    @property
    def gripper_state(self):
        return self.env.gripper_state

    def reset(self):
        self.env.reset()
        self.env.render()
        rgb_external = self.env.sim.render(width=256, height=256, camera_name="external_camera_0", depth=False,
                    mode='offscreen', device_id=-1)
        rgb_gripper = self.env.sim.render(width=256, height=256, camera_name="gripper_camera_rgb", depth=False,
            mode='offscreen', device_id=-1)

        s = self.state_preprocess(rgb_gripper)

        return s

    def render(self):
        self.env.render()


#   python maze.py DQN.yaml
#   python maze.py Q-learning.yaml
#   
# Author:  kbehouse  <https://github.com/kbehouse/>
# modify from https://github.com/yenchenlin/DeepLearningFlappyBird/blob/master/deep_q_network.py and https://github.com/floodsung/DRL-FlappyBird/blob/master/FlappyBirdDQN.py        

import sys, os
import time
import numpy as np
import cv2
import sys

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)),'../../'))
from standalone import Standalone
from config import load_config_from_arg, get_yaml_name_from_arg
sys.path.append("./")
from fetch_cam import FetchDiscreteEnv


print('FetchDiscreteEnv.__file__=', FetchDiscreteEnv.__file__)
# state preprocess to 80*80 gray image
def state_preprocess(state):
    state = cv2.cvtColor(cv2.resize(state, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, state = cv2.threshold(state,1,255,cv2.THRESH_BINARY)
    return np.reshape(state,(80,80,1))

class Fetch_Cam_Standalone(Standalone):

    def log_time(self, s):
        if hasattr(self, 'ts'):
            print(s + ' use time = ' + str( time.time() - self.ts  ))
        self.ts = time.time()

    def gripper_pic(self):
        rgb_external = self.env.sim.render(width=256, height=256, camera_name="external_camera_0", depth=False,
                    mode='offscreen', device_id=-1)
        rgb_gripper = self.env.sim.render(width=256, height=256, camera_name="gripper_camera_rgb", depth=False,
            mode='offscreen', device_id=-1)
        return rgb_external

    def env_init(self):
        self.env = FetchDiscreteEnv(dis_tolerance = 0.001, step_ds=0.005)
        self.env.render()
        # init_state = self.gripper_pic()
        # init_state = cv2.cvtColor(cv2.resize(init_state, (80, 80)), cv2.COLOR_RGB2GRAY)
        # ret, init_state = cv2.threshold(init_state,1,255,cv2.THRESH_BINARY)
        # self.state = np.stack((init_state, init_state, init_state, init_state), axis = 2)

    def env_reset(self):
        self.env.reset()
        self.env.gripper_close(False)

        init_state = self.gripper_pic()
        init_state = cv2.cvtColor(cv2.resize(init_state, (80, 80)), cv2.COLOR_RGB2GRAY)
        ret, init_state = cv2.threshold(init_state,1,255,cv2.THRESH_BINARY)
        self.state = np.stack((init_state, init_state, init_state, init_state), axis = 2)

        return self.state

    def on_action_response(self, action):
        next_state, reward, done, _ = self.env.step(action)
        pic = self.gripper_pic()
        next_state = state_preprocess(pic)
        next_state_4pic = np.append(self.state[:,:,1:], next_state,axis = 2)
        # self.log_time('env step ')
        now_state_4pic = self.state
        self.state = next_state_4pic
    
        return now_state_4pic, reward, done, next_state_4pic

if __name__ == '__main__':
    cfg = load_config_from_arg()
    # print(cfg)
    Fetch_Cam_Standalone(cfg, project_name='fetch_cam-' + get_yaml_name_from_arg()).run()

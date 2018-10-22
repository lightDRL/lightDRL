#   python fetch_cam_standalone_evaluation.py DQN.yaml
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
sys.path.append("fetch_cam/")
from fetch_cam import FetchDiscreteEnv


# print('FetchDiscreteEnv.__file__=', FetchDiscreteEnv.__file__)

# state preprocess to 80*80 gray image
def state_preprocess(state):
    state = cv2.cvtColor(cv2.resize(state, (80, 80)), cv2.COLOR_RGB2GRAY)
    # ret, state = cv2.threshold(state,1,255,cv2.THRESH_BINARY)
    return np.reshape(state,(80,80,1))


def create_dir(dir_name):
    import os
    if os.path.exists(dir_name):
       import shutil
       shutil.rmtree(dir_name) 
    os.makedirs(dir_name)

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
        return rgb_gripper

    def env_init(self):
        self.env = FetchDiscreteEnv(dis_tolerance = 0.001, step_ds=0.005, is_render=False)
        self.env.render()
        # init_state = self.gripper_pic()
        # init_state = cv2.cvtColor(cv2.resize(init_state, (80, 80)), cv2.COLOR_RGB2GRAY)
        # ret, init_state = cv2.threshold(init_state,1,255,cv2.THRESH_BINARY)
        # self.state = np.stack((init_state, init_state, init_state, init_state), axis = 2)

    def env_reset(self):
        self.env.reset()
        self.env.hide_obj1_obj2()
        self.env.gripper_close(False)

        init_state = self.gripper_pic()
        gray_init_state = cv2.cvtColor(cv2.resize(init_state, (80, 80)), cv2.COLOR_RGB2GRAY)
        # ret, gray_init_state = cv2.threshold(init_state,1,255,cv2.THRESH_BINARY)
        self.state = np.stack((gray_init_state, gray_init_state, gray_init_state, gray_init_state), axis = 2)

        # self.save_dir = 'ep_pic/ep_%03d' % self.ep
        # create_dir(self.save_dir)
        # cv2.imwrite(self.save_dir + '/%03d_r%3.2f_init.jpg' % (self.ep_use_step,0 ), cv2.cvtColor(self.gripper_pic(), cv2.COLOR_RGB2BGR))
        # cv2.imwrite(self.save_dir + '/%03d_r%3.2f_init_gray.jpg' % (self.ep_use_step,0 ), gray_init_state )
        

        return self.state

    def on_action_response(self, action):
        # print('action = ', action)
        a_one_hot = np.zeros(5)
        a_one_hot[action] = 1
        _ , reward, done, _ = self.env.step(a_one_hot)
        pic = self.gripper_pic()
        next_state = state_preprocess(pic)
        next_state_4pic = np.append(self.state[:,:,1:], next_state,axis = 2)
        # self.log_time('env step ')
        now_state_4pic = self.state
        self.state = next_state_4pic

        if self.ep %100 == 0:
            self.env.render()
        # cv2.imwrite(self.save_dir + '/%03d_r%3.2f.jpg' % (self.ep_use_step,0 ), cv2.cvtColor(pic, cv2.COLOR_RGB2BGR))
        # cv2.imwrite(self.save_dir + '/%03d_r%3.2f_gray.jpg' % (self.ep_use_step,0 ), next_state[:,:,0] )#np.squeeze(next_state) )
        
        # print('%03d: r=%.2f' % (self.ep, reward), ',done=', done)
        return now_state_4pic, reward, done, next_state_4pic

    

    # def ep_done_cb(self, ep_reward, ep = None,  all_ep_sum_reward = None, ep_use_steps=None, terminal_reward = None):
    def evaluation_done_cb( self, ep = None, terminal_reward = None, ep_use_steps = None):
        if not hasattr(self, 'EP_Success'):
            self.EP_Success = 0
            self.EP_Overstep = 0
            self.EP_Fail = 0
            self.sum_steps = 0
        
        # print('in Fetch_Cam_Standalone() ep_done_cb()!, terminal_reward = ', terminal_reward)
        if terminal_reward >0.2:
            self.EP_Success+=1
        elif terminal_reward ==0:
            self.EP_Overstep+=1
        elif terminal_reward ==-1:
            self.EP_Fail+=1
        else:
            print('Strange Reward!! terminal_reward = ', terminal_reward)

        self.sum_steps+=ep_use_steps

        t = time.time() - self.start_time
        hms = '%02dh%02dm%02ds' % (t/3600, t/60 % 60  , t%60)
        avg_step_per_second = self.sum_steps/int(t)
        sum_count = self.EP_Success+self.EP_Overstep+self.EP_Fail
        print(f'EP={ep:5d}, EP_Success={self.EP_Success:5d}, EP_Fail={self.EP_Fail:5d}, EP_Overstep={self.EP_Overstep:5d},sum_count={sum_count:5d}, t = {hms}, step/s={avg_step_per_second:5.2f}')




if __name__ == '__main__':

    cfg = load_config_from_arg()
    cfg['misc']['evaluation'] = True
    cfg['misc']['evaluation_ep'] = 10000
    # print(cfg)
    Fetch_Cam_Standalone(cfg, project_name='fetch_cam-' + get_yaml_name_from_arg()).run()

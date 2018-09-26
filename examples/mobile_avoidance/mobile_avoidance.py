# Run with DDPG
#   python ../../server.py
#   python mobile_avoidance.py DDPG.yaml
#   
# Author:  allengun2000  <https://github.com/allengun2000/>
#          

import sys, os
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)),'../../'))
from client import Client, EnvSpace
from car_env import CarEnv
import time
from config import cfg, get_yaml_name
import numpy as np

TRAIN_MODE = True
MAX_EPISODES = 1000
EP_MAXSTEP = 100
RENDER = True

class MobileAvoidance(EnvSpace):
    def env_init(self):
        self.env = CarEnv()
        self.state = self.env.reset()
        self.send_state_get_action(self.state)

        self.var = 1

    def on_predict_response(self, action):
        self.var = self.var * 0.9995 if self.ep_use_step > cfg['DDPG']['memory_capacity'] else self.var
        a = np.clip(np.random.normal(action, self.var), *self.env.action_bound) 
        next_state, reward, done, _ = self.env.step(action)
        # print(next_state)
        done = True if self.ep_use_step >= EP_MAXSTEP else done
        self.send_train_get_action(self.state, action, reward , done, next_state)
        self.state = next_state
    
        # print('self.env_name=',self.env_name)
        if self.cfg['misc']['render'] and self.ep>= self.cfg['misc']['render_after_ep'] :
            self.env.render()   
        if done:
            self.state =  self.env.reset()
            self.send_state_get_action(self.state)

if __name__ == '__main__':
    c = Client(MobileAvoidance, project_name='mobile_avoidance-'+get_yaml_name())

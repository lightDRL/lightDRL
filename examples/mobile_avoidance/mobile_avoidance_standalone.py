#   python maze_standalone.py DQN.yaml
#   python maze.py Q-learning.yaml
#   
# Author:  kbehouse  <https://github.com/kbehouse/>
#          

import sys, os
import time
import numpy as np
from car_env import CarEnv
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)),'../../'))
from config import load_config_from_arg, get_yaml_name_from_arg
from standalone import Standalone, standalone_switch

MAX_EP_STEPS = 100

class AvoidanceStandalone(Standalone):
    def env_init(self):
        self.env = CarEnv()
        self.a_list = []
        self.r_list =[]
    
    def env_reset(self):
        self.state = self.env.reset()/400
        return self.state

    def on_action_response(self, action):
        # print('on_action_response a = ', action)
        next_state, reward, done, _ = self.env.step(action)
        now_state = self.state
        self.state = next_state
        

        self.a_list.append(action)
        self.r_list.append(reward)
        # print(f' r={reward}, d={done}')
        # print(f' s = {now_state}, r={reward}, d={done}, s_={next_state}')
        # print(f' r={reward}, d={done}')
        if self.ep>=50:
            self.env.render()
        if self.ep_use_step >= (MAX_EP_STEPS-1): 
            done = True   
            # print('a_list = ', self.a_list)
            # print('r_list = ', self.r_list)

        return now_state, reward, done, next_state


if __name__ == '__main__':
    s = standalone_switch(AvoidanceStandalone, load_config_from_arg(), project_name='avoidance-' + get_yaml_name_from_arg())
    s.run()
    # c = MazeStandalone(load_config_from_arg(), project_name='maze-' + get_yaml_name_from_arg())
    # c.set_success(threshold_r = 1, threshold_successvie_count = 20)
    # c.run()
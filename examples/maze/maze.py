#   python maze.py DQN.yaml
#   python maze.py Q-learning.yaml
#   
# Author:  kbehouse  <https://github.com/kbehouse/>
#          

import sys, os
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)),'../../'))
from client_standalone import Client, EnvSpace
from maze_env import Maze
import time
from config import cfg, get_yaml_name
import numpy as np

class GymBasic(EnvSpace):
    def env_init(self):
        self.env = Maze()    
        # self.env.seed(self.cfg['misc']['random_seed'])
        
        self.state = self.env.reset()
        self.send_state_get_action(self.state)

    def on_action_response(self, action):
        # print('client use action = ', action)
        next_state, reward, done, _ = self.env.step(action)
        self.send_train_get_action(self.state, action, reward , done, next_state)
        self.state = next_state
    
        if self.cfg['misc']['render'] and self.ep > self.cfg['misc']['render_after_ep']:
            self.env.render()   
        if done:
            self.state =  self.env.reset()
            self.send_state_get_action(self.state)

            if self.ep > self.cfg['misc']['max_ep']:
                return 

def maze_cfg(cfg):
    # of course, you colud set following in .yaml
    cfg['RL']['state_discrete'] = False     # very special for continous x, y 
    cfg['RL']['state_shape']  = (2,)        # (x, y)
    #cfg['RL']['state_discrete_n'] = env.observation_space.n
    
    # action
    cfg['RL']['action_discrete'] = True 
    cfg['RL']['action_shape'] = (4,)
    cfg['RL']['action_discrete_n'] = 4
   
    return cfg

if __name__ == '__main__':
    c = Client(GymBasic, i_cfg = maze_cfg(cfg) , project_name='maze-' + get_yaml_name() ,retrain_model= True).run()
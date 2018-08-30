#   python maze.py DQN.yaml
#   python maze.py Q-learning.yaml
#   
# Author:  kbehouse  <https://github.com/kbehouse/>
# modify from https://github.com/yenchenlin/DeepLearningFlappyBird/blob/master/deep_q_network.py and https://github.com/floodsung/DRL-FlappyBird/blob/master/FlappyBirdDQN.py        

import sys, os
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)),'../../'))
from client_nons import Client, EnvBase
import time
from config import cfg, get_yaml_name
import numpy as np
import cv2
import sys
sys.path.append("flappybird_env/")
import wrapped_flappy_bird as game
import time

# state preprocess to 80*80 gray image
def state_preprocess(state):
    state = cv2.cvtColor(cv2.resize(state, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, state = cv2.threshold(state,1,255,cv2.THRESH_BINARY)
    return np.reshape(state,(80,80,1))

class FlappyBirdBasic(EnvBase):

    def log_time(self, s):
        if hasattr(self, 'ts'):
            print(s + ' use time = ' + str( time.time() - self.ts  ) + ', time=' + str(time.time()))
        self.ts = time.time()

    def env_init(self):
        self.env = game.GameState()
        self.log_time('before step')
        init_state, reward, done,  _ = self.env.step(0)  # do nothing for get image
        self.log_time('env step')
        init_state = cv2.cvtColor(cv2.resize(init_state, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, init_state = cv2.threshold(init_state,1,255,cv2.THRESH_BINARY)
        self.state = np.stack((init_state, init_state, init_state, init_state), axis = 2)
        self.log_time('preprocess and stack')
        self.send_state_get_action(self.state)

    def on_action_response(self, action):
        # print('client use action = ', action)
        self.log_time('get action ')
        next_state, reward, done, _ = self.env.step(action)
        next_state = state_preprocess(next_state)
        next_state_4pic = np.append(self.state[:,:,1:], next_state,axis = 2)
        self.log_time('env step ')
        # time.sleep(3)
        # self.log_time('after sleep')
        self.send_train_get_action(self.state, action, reward , done, next_state_4pic)
        self.log_time('-----after send train get action------')
        self.state = next_state_4pic
    
        # if self.cfg['misc']['render'] and self.ep > self.cfg['misc']['render_after_ep']:
        #self.env.render()   
        if done:
            if self.ep > self.cfg['misc']['max_ep']:
                return 
            #self.state =  self.env.reset()
            self.send_state_get_action(self.state)

            
def flappybird_cfg(cfg):
    # of course, you colud set following in .yaml
    cfg['RL']['state_discrete'] = True     
    cfg['RL']['state_shape']  = (80,80,4)         
    # action
    cfg['RL']['action_discrete'] = True 
    cfg['RL']['action_shape'] = (2,)
   
    return cfg

if __name__ == '__main__':
    c = Client(FlappyBirdBasic, project_name='conn_flappybird-'+ '-' + get_yaml_name(), i_cfg = flappybird_cfg(cfg), retrain_model= True)

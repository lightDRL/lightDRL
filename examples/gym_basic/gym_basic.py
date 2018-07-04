# Run with DDPG
#   python ../../server.py
#   python gym_basic.py DDPG.yaml
#   
# Author:  kbehouse  <https://github.com/kbehouse/>
#          

import sys, os
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)),'../../'))
from client_standalone import Client, EnvSpace
import gym
import time
from config import cfg, get_yaml_name
import numpy as np
import threading

MAX_EPISODES = 1000
RENDER = False

ENV_NAME = cfg['misc']['gym_env'] #'CartPole-v0' 

class GymBasic(EnvSpace):
    def env_init(self):
        self.env = gym.make(ENV_NAME)
        self.env.seed(cfg['misc']['random_seed'])

        self.state = self.env.reset()
        self.send_state_get_action(self.state)

    def on_action_response(self, action):
        # print('client use action = ', action)
        next_state, reward, done, _ = self.env.step(action)
        self.send_train_get_action(self.state, action, reward , done, next_state)
        self.state = next_state
    
        if self.ep>=100 and RENDER:
            self.env.render()   
        if done:
            self.state =  self.env.reset()
            self.send_state_get_action(self.state)

            if self.ep>=MAX_EPISODES:
                exit()

        

def modify_cfg():
    env = gym.make(ENV_NAME)
    env = env.unwrapped
    # state shape 
    if type(env.observation_space)== gym.spaces.discrete.Discrete:
        # discrete state shape
        cfg['RL']['state_shape']  = np.array(1)
    elif type(env.observation_space)==gym.spaces.tuple_space.Tuple:
        state_num = len(env.observation_space.spaces)
        cfg['RL']['state_shape']  = np.array(state_num)
    else:
        cfg['RL']['state_shape'] = env.observation_space.shape
    
    # action
    cfg['RL']['action_discrete'] = True if type(env.action_space) == gym.spaces.discrete.Discrete else False
    
    print("cfg['RL']['action_discrete'] = ", cfg['RL']['action_discrete'])

    if cfg['RL']['action_discrete']:
        cfg['RL']['action_num'] = env.action_space.n
    else:
        assert len(env.action_space.shape) == 1, 'NOT support >= 2D action,  len(env.action_space.shape)=%d' %  len(env.action_space.shape)
        assert (env.action_space.high == -env.action_space.low), 'NOT support action high low, only support high=-low'        
        cfg['RL']['action_num'] = env.action_space.shape[0]
        cfg['RL']['action_bound'] = env.action_space.high 
        # print("cfg['RL']['action_num'] = ", cfg['RL']['action_num'])

    env.close()
    print('{} close! Because get parameter done.'.format(ENV_NAME))
    return cfg

if __name__ == '__main__':
    c = Client(GymBasic, project_name='gym-' + get_yaml_name(), i_cfg = modify_cfg(), retrain_model= True)

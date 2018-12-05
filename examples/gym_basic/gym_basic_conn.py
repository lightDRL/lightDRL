# Run with DDPG
#   python ../../server.py
#   python gym_basic_conn.py DDPG_CartPole-v0.yaml
#   python gym_basic_conn.py DDPG_MountainCarContinuous-v0.yaml
# Author:  kbehouse  <https://github.com/kbehouse/>
#          

import sys, os
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)),'../../'))
from client import Client #, EnvSpace
import gym
import time
from config import load_config_from_arg
import numpy as np
# from gym_basic import gym_cfg

class GymClient(Client):
    def env_init(self):
        self.env = gym.make(self.cfg['misc']['gym_env'])
        self.env.seed(self.cfg['misc']['random_seed'])

    def env_reset(self):
        self.state = self.env.reset()
        print('[%03d] ' % self.ep, 's_%d: [%4.2f,%4.2f,%4.2f,%4.2f]' % (self.ep_use_step, self.state[0], self.state[1],self.state[2],self.state[3]) )
        return self.state

    def on_action_response(self, action):

        print('[%03d] ' % self.ep,'a_%d: ' % self.ep_use_step, action)
        next_state, reward, done, _ = self.env.step(action)
        now_state = self.state
        self.state = next_state
        if self.cfg['misc']['render'] and self.ep > self.cfg['misc']['render_after_ep']:
            self.env.render()   

        print('[%03d] ' % self.ep, 's_%d: [%4.2f,%4.2f,%4.2f,%4.2f]' % (self.ep_use_step, now_state[0], now_state[1],now_state[2],now_state[3]),
            ', r_%d:' % self.ep_use_step , reward,  
            ', d_%d:' % self.ep_use_step , done, 
            ', s_%d [%4.2f,%4.2f,%4.2f,%4.2f]' % (self.ep_use_step+1, next_state[0], next_state[1],next_state[2], next_state[3]) )
        return now_state, reward, done, next_state


def gym_cfg(cfg):
    env = gym.make( cfg['misc']['gym_env'])
    env = env.unwrapped

    cfg['RL']['state_discrete'] = True if type(env.observation_space) == gym.spaces.discrete.Discrete else False
    # state shape 
    if cfg['RL']['state_discrete']:
        # print('inininii gym.spaces.discrete.Discrete:')
        # discrete state shape
        cfg['RL']['state_shape']  = (1,)
        cfg['RL']['state_discrete_n'] = env.observation_space.n
    elif type(env.observation_space)==gym.spaces.tuple_space.Tuple:
        state_num = len(env.observation_space.spaces)
        cfg['RL']['state_shape']  = np.array(state_num)
    else:
        cfg['RL']['state_shape'] = env.observation_space.shape
    
    print("cfg['RL']['state_shape'] = ", cfg['RL']['state_shape'])

    # action
    cfg['RL']['action_discrete'] = True if type(env.action_space) == gym.spaces.discrete.Discrete else False
    
    # print("cfg['RL']['action_discrete'] = ", cfg['RL']['action_discrete'])

    if cfg['RL']['action_discrete']:
        cfg['RL']['action_shape'] = (env.action_space.n,)
        #cfg['RL']['action_discrete_n'] = env.action_space.n
    else:
        assert len(env.action_space.shape) == 1, 'NOT support >= 2D action,  len(env.action_space.shape)=%d' %  len(env.action_space.shape)
        assert (env.action_space.high == -env.action_space.low), 'NOT support action high low, only support high=-low'        
        cfg['RL']['action_bound'] = env.action_space.high 
        cfg['RL']['action_shape'] = env.action_space.shape
    env.close()
    
    print("cfg['RL']['action_shape']= ", cfg['RL']['action_shape'])

    
    print('{} close! Because get parameter done.'.format( cfg['misc']['gym_env']))
    return cfg

if __name__ == '__main__':
    cfg = gym_cfg( load_config_from_arg()  )
    c = GymClient(i_cfg = cfg , project_name='gym-'+ cfg['misc']['gym_env'])
    c.set_success(cfg['misc']['threshold_r'], cfg['misc']['threshold_successvie_count'])
    c.run()
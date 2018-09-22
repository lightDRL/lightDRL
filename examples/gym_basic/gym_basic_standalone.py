# Run with DDPG
#   python gym_basic_standalone.py DDPG_CartPole-v0.yaml
#   
# Author:  kbehouse  <https://github.com/kbehouse/>
#          

import sys, os
import time
import numpy as np
import gym
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)),'../../'))
from config import load_config_from_arg, get_yaml_name_from_arg
from standalone import Standalone

print('gym version:', gym.__version__)
print('gym path = ', gym.__file__)

class GymStandalone(Standalone):
    def env_init(self):
        self.env = gym.make(self.cfg['misc']['gym_env'])
        if self.cfg['misc']['gym_monitor_path'] != None:
            try:
                self.env = gym.wrappers.monitor.Monitor(self.env,self.cfg['misc']['gym_monitor_path'], 
                    force=True, video_callable=lambda episode_id: episode_id % self.cfg['misc']['gym_monitor_episode']==0 ) 
            except:
                self.env = gym.wrappers.monitoring.Monitor(self.env,self.cfg['misc']['gym_monitor_path'], 
                    force=True, video_callable=lambda episode_id: episode_id % self.cfg['misc']['gym_monitor_episode']==0 ) 
                
        self.env.seed(self.cfg['misc']['random_seed'])

    def env_reset(self):
        self.state = self.env.reset()
        return self.state

    def on_action_response(self, action):

        next_state, reward, done, _ = self.env.step(action)
        now_state = self.state
        self.state = next_state
        if self.cfg['misc']['render'] and self.ep > self.cfg['misc']['render_after_ep']:
            self.env.render()   

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
    print('{} close! Because get parameter done.'.format( cfg['misc']['gym_env']))
    return cfg

if __name__ == '__main__':
    cfg = gym_cfg( load_config_from_arg()  )
    c = GymStandalone(cfg , project_name='gym-' + get_yaml_name_from_arg())
    c.set_success(threshold_r = 200, threshold_successvie_count = 20)
    c.run()
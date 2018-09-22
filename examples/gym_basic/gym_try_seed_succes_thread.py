# Run gym with different seed
#   ex: 
#       python gym_try_seed.py DDPG_CartPole-v0.yaml
#       
# Author:  kbehouse  <https://github.com/kbehouse/>
#          

import sys, os
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)),'../../'))
from gym_basic_standalone import GymStandalone, gym_cfg
import gym
import time
from config import set_gym_monitor_path, load_config
import numpy as np
from threading import Thread
import copy

class ThreadWithReturnValue(Thread):
    # from: https://stackoverflow.com/questions/6893968/how-to-get-the-return-value-from-a-thread-in-python
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None
    def run(self):
        print(type(self._target))
        if self._target is not None:
            self._return = self._target(*self._args,
                                                **self._kwargs)
    def join(self, *args):
        Thread.join(self, *args)
        return self._return

def gym_thread(seed, yaml_path):
    # print('thread get path = ', yaml_path)
    cfg_copy = gym_cfg( load_config(yaml_path)  )
    # print('cfg_copy of seed=', seed)
    
    cfg_copy['misc']['random_seed'] = seed
    cfg_copy['misc']['render'] = True
    cfg_copy['misc']['gpu_memory_ratio'] = 0.70     # all share same ratio
    cfg_copy['misc']['max_ep'] = 600
    cfg_copy['misc']['worker_nickname'] = 'w_%03d' % (seed)
    prj_name ='gym-%s_seed_%04d' % ( cfg_copy['RL']['method'], seed)
    
    cfg_copy['misc']['gym_monitor_path'] = None
    # s = Server(GymBasic, i_cfg = cfg_copy , project_name=prj_name,retrain_model= True)
    c = GymStandalone(cfg_copy , project_name=prj_name)
    # c.set_success(threshold_r = 200, threshold_successvie_count = 20)
    c.run()

    avg_r = c.all_ep_reward
    return avg_r

if __name__ == '__main__':
    yaml_path = sys.argv[1]
    print('Main thread yaml_paht = ', yaml_path)
    all_t =[]
    for seed in range(70,86):  #39, 118
        t = ThreadWithReturnValue(target=gym_thread, args=(seed,yaml_path, ), name='t_seed_%03d' % seed )
        t.start()
        all_t.append(t)
        
    max_avg_ep_r = -999999
    min_avg_ep_r =  999999
    max_t = None
    min_t = None
    for t in all_t:
        avg_ep_r = t.join()

        if avg_ep_r > max_avg_ep_r:
            max_t = t
            max_avg_ep_r = avg_ep_r

        if avg_ep_r < min_avg_ep_r:
            min_t = t
            min_avg_ep_r = avg_ep_r
        
    print('%s get max reward = %6.2f' % (max_t.name, max_avg_ep_r))
    

'''
(w_009-asyc-003) EP  150 | all_ep_reward: 13660.000000
(w_009-asyc-002) EP  149 | EP_Step:    45 | EP_Reward:    45.00 | MAX_R: 1.00  | EP_Time:   0m 0s | All_Time:   0h 8m11s
(w_009-asyc-002) EP  150 | all_ep_reward: 13614.000000
('Best ep avg reward = ', 91.06666666666666)
t_seed_009 get max reward =  91.07

('Best ep avg reward = ', 91.06666666666666)
t_seed_009 get max reward =  60
'''
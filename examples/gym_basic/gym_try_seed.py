# Run gym with different seed
#   ex: 
#       python gym_try_seed.py DDPG_CartPole-v0.yaml
#       
# Author:  kbehouse  <https://github.com/kbehouse/>
#          

import sys, os
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)),'../../'))
from client_standalone import Client, EnvSpace
import gym
import time
from config import cfg, get_yaml_name, set_gym_monitor_path
import numpy as np
from gym_basic import gym_cfg, GymBasic
from threading import Thread, Lock
import copy

class ThreadWithReturnValue(Thread):
    # from: https://stackoverflow.com/questions/6893968/how-to-get-the-return-value-from-a-thread-in-python
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs, Verbose)
        self._return = None
        self.name = name
    def run(self):
        if self._Thread__target is not None:
            self._return = self._Thread__target(*self._Thread__args,
                                                **self._Thread__kwargs)
    def join(self):
        Thread.join(self)
        return self._return

threadLock = Lock()

def gym_thread(seed):
    
    threadLock.acquire()
    # print('seed = ', seed)
    cfg_copy = copy.deepcopy(cfg) #cfg.deepcopy()
    threadLock.release()
    cfg_copy = gym_cfg(cfg_copy)
    # print('cfg_copy of seed=', seed)
    
    cfg_copy['misc']['random_seed'] = seed
    cfg_copy['misc']['render'] = False
    cfg_copy['misc']['gpu_memory_ratio'] = 0.1
    cfg_copy['misc']['max_ep'] = 1000
    cfg_copy['misc']['worker_nickname'] = 'w_' + str(seed)
    prj_name ='gym-%s_seed_%04d' % ( get_yaml_name(), seed)
    
    cfg_copy['misc']['gym_monitor_path'] = None
    # cfg_copy['misc']['gym_monitor_path'] = set_gym_monitor_path(cfg_copy['misc']['gym_monitor_path_origin'], i_project_name=prj_name)
    # print("cfg['misc']['gym_monitor_path'] = ", cfg_copy['misc']['gym_monitor_path'])    

    c = Client(GymBasic, project_name=prj_name, i_cfg = cfg_copy, retrain_model= True)
    c.run()
    avg_r = c.env_space.worker.avg_ep_reward()
    return avg_r

if __name__ == '__main__':
    all_t =[]
    for seed in range(1,100):  #39, 118
        s = seed
        t = ThreadWithReturnValue(target=gym_thread, args=(s,), name='t_'+ str(seed), )
        t.start()
        all_t.append(t)
        
    max_avg_ep_r = -9999
    max_t = None
    for t in all_t:
        avg_ep_r = t.join()

        if avg_ep_r > max_avg_ep_r:
            max_t = t
            max_avg_ep_r = avg_ep_r
        
    print('%s get max reward = %6.2f' % (max_t.name, max_avg_ep_r))
    

'''
moutaincar
(w_33) EP   78 | EP_Reward:    84.89 | MAX_R: 99.88 | EP_Time:   0m19s | All_Time:   2h20m16s
(w_91) EP   75 | EP_Reward:    91.64 | MAX_R: 99.90 | EP_Time:   0m 9s | All_Time:   2h20m15s
(w_34) EP   26 | EP_Reward:   -20.34 | MAX_R: -0.00 | EP_Time:   2m20s | All_Time:   2h20m21s
(w_62) EP   26 | EP_Reward:   -24.68 | MAX_R: -0.00 | EP_Time:   2m21s | All_Time:   2h20m23s
(w_92) EP   59 | EP_Reward:    88.35 | MAX_R: 99.95 | EP_Time:   0m12s | All_Time:   2h20m23s
(w_8) EP   37 | EP_Reward:    89.27 | MAX_R: 100.00 | EP_Time:   0m11s | All_Time:   2h20m25s
(w_35) EP   26 | EP_Reward:   -19.47 | MAX_R: -0.00 | EP_Time:   2m19s | All_Time:   2h20m24s
t_44 get max reward =  78.59
'''
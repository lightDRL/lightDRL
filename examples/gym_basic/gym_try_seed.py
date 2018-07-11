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
    cfg_copy['misc']['gpu_memory_ratio'] = 0.76
    cfg_copy['misc']['max_ep'] = 120
    cfg_copy['misc']['worker_nickname'] = 'w_%03d' % (seed)
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


Cartpole DDPG
(w_25) EP  200 | EP_Step:   200 | EP_Reward:   200.00 | MAX_R: 1.00  | EP_Time:   0m 3s | All_Time:   0h42m 0s
(w_39) EP  200 | EP_Step:   200 | EP_Reward:   200.00 | MAX_R: 1.00  | EP_Time:   0m 2s | All_Time:   0h42m 4s
(w_38) EP  200 | EP_Step:   200 | EP_Reward:   200.00 | MAX_R: 1.00  | EP_Time:   0m 1s | All_Time:   0h42m16s

(w_28) EP  196 | EP_Step:   200 | EP_Reward:   200.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 0s | All_Time:   0h17m 9s
(w_28) EP  197 | EP_Step:   200 | EP_Reward:   200.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 0s | All_Time:   0h17m10s
(w_28) EP  198 | EP_Step:    87 | EP_Reward:    87.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 0s | All_Time:   0h17m10s
(w_28) EP  199 | EP_Step:    25 | EP_Reward:    25.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 0s | All_Time:   0h17m10s
(w_28) EP  200 | EP_Step:    10 | EP_Reward:    10.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 0s | All_Time:   0h17m10s
t_28 get max reward =  66.04

DQN Cartpole
(w_079) EP  191 | EP_Step:   167 | EP_Reward:   167.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 0s | All_Time:   0h17m40s
(w_079) EP  192 | EP_Step:   141 | EP_Reward:   141.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 0s | All_Time:   0h17m40s
(w_079) EP  193 | EP_Step:    40 | EP_Reward:    40.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 0s | All_Time:   0h17m40s
(w_079) EP  194 | EP_Step:   144 | EP_Reward:   144.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 0s | All_Time:   0h17m41s
(w_079) EP  195 | EP_Step:   159 | EP_Reward:   159.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 0s | All_Time:   0h17m41s
(w_079) EP  196 | EP_Step:   120 | EP_Reward:   120.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 0s | All_Time:   0h17m42s
(w_079) EP  197 | EP_Step:   144 | EP_Reward:   144.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 0s | All_Time:   0h17m42s
(w_079) EP  198 | EP_Step:   107 | EP_Reward:   107.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 0s | All_Time:   0h17m42s
(w_079) EP  199 | EP_Step:    94 | EP_Reward:    94.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 0s | All_Time:   0h17m43s
(w_079) EP  200 | EP_Step:    52 | EP_Reward:    52.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 0s | All_Time:   0h17m43s
t_79 get max reward =  72.8


(w_043) EP  109 | EP_Step:    75 | EP_Reward:    75.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 4s | All_Time:   0h 9m 1s
(w_026) EP   88 | EP_Step:    56 | EP_Reward:    56.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 3s | All_Time:   0h 9m 1s
(w_061) EP  115 | EP_Step:    34 | EP_Reward:    34.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 2s | All_Time:   0h 9m 1s
(w_014) EP   92 | EP_Step:    11 | EP_Reward:    11.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 0s | All_Time:   0h 9m 1s
(w_006) EP  116 | EP_Step:   109 | EP_Reward:   109.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 7s | All_Time:   0h 9m 1s
(w_057) EP   93 | EP_Step:    73 | EP_Reward:    73.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 4s | All_Time:   0h 9m 2s
(w_060) EP  113 | EP_Step:    96 | EP_Reward:    96.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 6s | All_Time:   0h 9m 4s
(w_085) EP  118 | EP_Step:    60 | EP_Reward:    60.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 3s | All_Time:   0h 9m 3s
(w_044) EP  115 | EP_Step:    58 | EP_Reward:    58.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 3s | All_Time:   0h 9m 3s
(w_061) EP  116 | EP_Step:    37 | EP_Reward:    37.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 2s | All_Time:   0h 9m 3s
(w_030) EP  106 | EP_Step:   200 | EP_Reward:   200.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m13s | All_Time:   0h 9m 6s
(w_001) EP  117 | EP_Step:    88 | EP_Reward:    88.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 5s | All_Time:   0h 9m 4s
(w_043) EP  110 | EP_Step:    58 | EP_Reward:    58.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 3s | All_Time:   0h 9m 5s
(w_068) EP  107 | EP_Step:   119 | EP_Reward:   119.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 7s | All_Time:   0h 9m 4s
(w_052) EP  120 | EP_Step:    68 | EP_Reward:    68.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 4s | All_Time:   0h 9m 4s
(w_076) EP  109 | EP_Step:    78 | EP_Reward:    78.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 4s | All_Time:   0h 9m 5s
(w_061) EP  117 | EP_Step:    21 | EP_Reward:    21.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 1s | All_Time:   0h 9m 5s
(w_073) EP  107 | EP_Step:    69 | EP_Reward:    69.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 4s | All_Time:   0h 9m 5s
(w_058) EP  103 | EP_Step:    99 | EP_Reward:    99.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 6s | All_Time:   0h 9m 5s
(w_089) EP  107 | EP_Step:   114 | EP_Reward:   114.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 7s | All_Time:   0h 9m 5s
(w_083) EP   85 | EP_Step:   200 | EP_Reward:   200.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m12s | All_Time:   0h 9m 5s
(w_057) EP   94 | EP_Step:    60 | EP_Reward:    60.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 3s | All_Time:   0h 9m 5s
(w_019) EP  100 | EP_Step:   137 | EP_Reward:   137.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 8s | All_Time:   0h 9m 6s
(w_006) EP  117 | EP_Step:    70 | EP_Reward:    70.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 4s | All_Time:   0h 9m 6s
(w_083) EP   86 | EP_Step:    12 | EP_Reward:    12.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 0s | All_Time:   0h 9m 6s
(w_071) EP   88 | EP_Step:   200 | EP_Reward:   200.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m12s | All_Time:   0h 9m 6s
(w_085) EP  119 | EP_Step:    59 | EP_Reward:    59.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 3s | All_Time:   0h 9m 6s
(w_044) EP  116 | EP_Step:    57 | EP_Reward:    57.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 3s | All_Time:   0h 9m 7s
(w_005) EP  106 | EP_Step:   153 | EP_Reward:   153.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 9s | All_Time:   0h 9m 7s
(w_063) EP   89 | EP_Step:   131 | EP_Reward:   131.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 8s | All_Time:   0h 9m 9s
(w_014) EP   93 | EP_Step:   109 | EP_Reward:   109.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 6s | All_Time:   0h 9m 8s
(w_043) EP  111 | EP_Step:    63 | EP_Reward:    63.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 3s | All_Time:   0h 9m 9s
(w_089) EP  108 | EP_Step:    52 | EP_Reward:    52.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 3s | All_Time:   0h 9m 8s
(w_060) EP  114 | EP_Step:   103 | EP_Reward:   103.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 5s | All_Time:   0h 9m10s
(w_014) EP   94 | EP_Step:    10 | EP_Reward:    10.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 0s | All_Time:   0h 9m 9s
(w_073) EP  108 | EP_Step:    67 | EP_Reward:    67.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 3s | All_Time:   0h 9m 9s
(w_001) EP  118 | EP_Step:    87 | EP_Reward:    87.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 5s | All_Time:   0h 9m 9s
(w_030) EP  107 | EP_Step:    97 | EP_Reward:    97.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 5s | All_Time:   0h 9m12s
(w_006) EP  118 | EP_Step:    56 | EP_Reward:    56.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 3s | All_Time:   0h 9m 9s
(w_026) EP   89 | EP_Step:   144 | EP_Reward:   144.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 8s | All_Time:   0h 9m10s
(w_057) EP   95 | EP_Step:    67 | EP_Reward:    67.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 3s | All_Time:   0h 9m 9s
(w_085) EP  120 | EP_Step:    60 | EP_Reward:    60.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 3s | All_Time:   0h 9m10s
(w_083) EP   87 | EP_Step:    68 | EP_Reward:    68.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 3s | All_Time:   0h 9m10s
(w_063) EP   90 | EP_Step:    37 | EP_Reward:    37.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 2s | All_Time:   0h 9m12s
(w_076) EP  110 | EP_Step:   101 | EP_Reward:   101.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 5s | All_Time:   0h 9m11s
(w_043) EP  112 | EP_Step:    50 | EP_Reward:    50.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 2s | All_Time:   0h 9m12s
(w_019) EP  101 | EP_Step:    94 | EP_Reward:    94.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 5s | All_Time:   0h 9m12s
(w_060) EP  115 | EP_Step:    56 | EP_Reward:    56.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 2s | All_Time:   0h 9m13s
(w_044) EP  117 | EP_Step:    79 | EP_Reward:    79.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 4s | All_Time:   0h 9m11s
(w_058) EP  104 | EP_Step:   112 | EP_Reward:   112.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 6s | All_Time:   0h 9m12s
(w_073) EP  109 | EP_Step:    63 | EP_Reward:    63.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 3s | All_Time:   0h 9m12s
(w_068) EP  108 | EP_Step:   140 | EP_Reward:   140.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 8s | All_Time:   0h 9m12s
(w_094) EP  100 | EP_Step:   200 | EP_Reward:   200.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m11s | All_Time:   0h 9m15s
(w_061) EP  118 | EP_Step:   134 | EP_Reward:   134.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 7s | All_Time:   0h 9m12s
(w_044) EP  118 | EP_Step:    21 | EP_Reward:    21.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 1s | All_Time:   0h 9m13s
(w_073) EP  110 | EP_Step:    18 | EP_Reward:    18.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 0s | All_Time:   0h 9m14s
(w_006) EP  119 | EP_Step:    75 | EP_Reward:    75.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 4s | All_Time:   0h 9m14s
(w_061) EP  119 | EP_Step:    17 | EP_Reward:    17.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 1s | All_Time:   0h 9m14s
(w_044) EP  119 | EP_Step:    17 | EP_Reward:    17.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 0s | All_Time:   0h 9m14s
(w_076) EP  111 | EP_Step:    60 | EP_Reward:    60.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 3s | All_Time:   0h 9m15s
(w_060) EP  116 | EP_Step:    47 | EP_Reward:    47.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 2s | All_Time:   0h 9m15s
(w_001) EP  119 | EP_Step:    97 | EP_Reward:    97.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 5s | All_Time:   0h 9m14s
(w_089) EP  109 | EP_Step:   111 | EP_Reward:   111.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 6s | All_Time:   0h 9m14s
(w_076) EP  112 | EP_Step:     9 | EP_Reward:     9.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 0s | All_Time:   0h 9m15s
(w_076) EP  113 | EP_Step:     9 | EP_Reward:     9.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 0s | All_Time:   0h 9m16s
(w_043) EP  113 | EP_Step:    78 | EP_Reward:    78.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 4s | All_Time:   0h 9m16s
(w_061) EP  120 | EP_Step:    32 | EP_Reward:    32.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 1s | All_Time:   0h 9m15s
(w_076) EP  114 | EP_Step:    10 | EP_Reward:    10.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 0s | All_Time:   0h 9m16s
(w_057) EP   96 | EP_Step:   115 | EP_Reward:   115.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 6s | All_Time:   0h 9m16s
(w_006) EP  120 | EP_Step:    39 | EP_Reward:    39.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 2s | All_Time:   0h 9m16s
(w_063) EP   91 | EP_Step:   105 | EP_Reward:   105.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 5s | All_Time:   0h 9m18s
(w_071) EP   89 | EP_Step:   181 | EP_Reward:   181.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m10s | All_Time:   0h 9m16s
(w_073) EP  111 | EP_Step:    65 | EP_Reward:    65.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 3s | All_Time:   0h 9m17s
(w_094) EP  101 | EP_Step:    84 | EP_Reward:    84.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 4s | All_Time:   0h 9m19s
(w_005) EP  107 | EP_Step:   181 | EP_Reward:   181.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 9s | All_Time:   0h 9m17s
(w_044) EP  120 | EP_Step:    64 | EP_Reward:    64.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 3s | All_Time:   0h 9m17s
(w_043) EP  114 | EP_Step:    42 | EP_Reward:    42.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 2s | All_Time:   0h 9m18s
(w_019) EP  102 | EP_Step:   126 | EP_Reward:   126.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 6s | All_Time:   0h 9m18s
(w_089) EP  110 | EP_Step:    69 | EP_Reward:    69.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 3s | All_Time:   0h 9m18s
(w_043) EP  115 | EP_Step:    13 | EP_Reward:    13.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 0s | All_Time:   0h 9m19s
(w_001) EP  120 | EP_Step:    76 | EP_Reward:    76.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 3s | All_Time:   0h 9m18s
(w_060) EP  117 | EP_Step:    78 | EP_Reward:    78.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 4s | All_Time:   0h 9m19s
(w_073) EP  112 | EP_Step:    34 | EP_Reward:    34.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 1s | All_Time:   0h 9m19s
(w_030) EP  108 | EP_Step:   189 | EP_Reward:   189.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 9s | All_Time:   0h 9m21s
(w_058) EP  105 | EP_Step:   136 | EP_Reward:   136.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 7s | All_Time:   0h 9m19s
(w_014) EP   95 | EP_Step:   200 | EP_Reward:   200.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m10s | All_Time:   0h 9m19s
(w_026) EP   90 | EP_Step:   200 | EP_Reward:   200.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m10s | All_Time:   0h 9m20s
(w_094) EP  102 | EP_Step:    63 | EP_Reward:    63.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 3s | All_Time:   0h 9m22s
(w_083) EP   88 | EP_Step:   200 | EP_Reward:   200.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m10s | All_Time:   0h 9m20s
(w_060) EP  118 | EP_Step:    46 | EP_Reward:    46.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 2s | All_Time:   0h 9m21s
(w_068) EP  109 | EP_Step:   167 | EP_Reward:   167.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 8s | All_Time:   0h 9m20s
(w_043) EP  116 | EP_Step:    54 | EP_Reward:    54.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 2s | All_Time:   0h 9m21s
(w_089) EP  111 | EP_Step:    56 | EP_Reward:    56.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 2s | All_Time:   0h 9m21s
(w_073) EP  113 | EP_Step:    55 | EP_Reward:    55.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 2s | All_Time:   0h 9m21s
(w_019) EP  103 | EP_Step:    70 | EP_Reward:    70.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 3s | All_Time:   0h 9m21s
(w_057) EP   97 | EP_Step:   117 | EP_Reward:   117.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 5s | All_Time:   0h 9m21s
(w_014) EP   96 | EP_Step:    63 | EP_Reward:    63.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 2s | All_Time:   0h 9m22s
(w_089) EP  112 | EP_Step:    34 | EP_Reward:    34.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 1s | All_Time:   0h 9m22s
(w_076) EP  115 | EP_Step:   149 | EP_Reward:   149.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 6s | All_Time:   0h 9m23s
(w_058) EP  106 | EP_Step:    80 | EP_Reward:    80.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 3s | All_Time:   0h 9m23s
(w_005) EP  108 | EP_Step:   119 | EP_Reward:   119.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 5s | All_Time:   0h 9m22s
(w_043) EP  117 | EP_Step:    48 | EP_Reward:    48.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 2s | All_Time:   0h 9m23s
(w_005) EP  109 | EP_Step:    11 | EP_Reward:    11.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 0s | All_Time:   0h 9m23s
(w_073) EP  114 | EP_Step:    62 | EP_Reward:    62.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 2s | All_Time:   0h 9m24s
(w_060) EP  119 | EP_Step:    76 | EP_Reward:    76.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 3s | All_Time:   0h 9m25s
(w_057) EP   98 | EP_Step:    65 | EP_Reward:    65.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 2s | All_Time:   0h 9m24s
(w_005) EP  110 | EP_Step:    24 | EP_Reward:    24.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 1s | All_Time:   0h 9m24s
(w_071) EP   90 | EP_Step:   162 | EP_Reward:   162.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 7s | All_Time:   0h 9m24s
(w_063) EP   92 | EP_Step:   174 | EP_Reward:   174.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 8s | All_Time:   0h 9m26s
(w_005) EP  111 | EP_Step:    10 | EP_Reward:    10.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 0s | All_Time:   0h 9m24s
(w_019) EP  104 | EP_Step:    85 | EP_Reward:    85.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 3s | All_Time:   0h 9m25s
(w_057) EP   99 | EP_Step:    24 | EP_Reward:    24.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 1s | All_Time:   0h 9m25s
(w_043) EP  118 | EP_Step:    52 | EP_Reward:    52.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 2s | All_Time:   0h 9m26s
(w_076) EP  116 | EP_Step:    61 | EP_Reward:    61.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 2s | All_Time:   0h 9m26s
(w_094) EP  103 | EP_Step:   118 | EP_Reward:   118.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 5s | All_Time:   0h 9m28s
(w_043) EP  119 | EP_Step:    11 | EP_Reward:    11.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 0s | All_Time:   0h 9m26s
(w_073) EP  115 | EP_Step:    40 | EP_Reward:    40.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 1s | All_Time:   0h 9m26s
(w_068) EP  110 | EP_Step:   113 | EP_Reward:   113.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 5s | All_Time:   0h 9m25s
(w_043) EP  120 | EP_Step:    10 | EP_Reward:    10.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 0s | All_Time:   0h 9m27s
(w_071) EP   91 | EP_Step:    45 | EP_Reward:    45.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 2s | All_Time:   0h 9m26s
(w_057) EP  100 | EP_Step:    36 | EP_Reward:    36.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 1s | All_Time:   0h 9m26s
(w_076) EP  117 | EP_Step:    28 | EP_Reward:    28.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 1s | All_Time:   0h 9m27s
(w_019) EP  105 | EP_Step:    43 | EP_Reward:    43.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 1s | All_Time:   0h 9m27s
(w_030) EP  109 | EP_Step:   200 | EP_Reward:   200.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 8s | All_Time:   0h 9m30s
(w_060) EP  120 | EP_Step:    99 | EP_Reward:    99.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 4s | All_Time:   0h 9m29s
(w_073) EP  116 | EP_Step:    65 | EP_Reward:    65.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 2s | All_Time:   0h 9m29s
(w_030) EP  110 | EP_Step:    10 | EP_Reward:    10.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 0s | All_Time:   0h 9m31s
(w_026) EP   91 | EP_Step:   200 | EP_Reward:   200.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 8s | All_Time:   0h 9m29s
(w_089) EP  113 | EP_Step:   151 | EP_Reward:   151.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 6s | All_Time:   0h 9m29s
(w_019) EP  106 | EP_Step:    47 | EP_Reward:    47.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 1s | All_Time:   0h 9m29s
(w_058) EP  107 | EP_Step:   142 | EP_Reward:   142.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 6s | All_Time:   0h 9m29s
(w_068) EP  111 | EP_Step:    76 | EP_Reward:    76.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 3s | All_Time:   0h 9m28s
(w_083) EP   89 | EP_Step:   200 | EP_Reward:   200.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 8s | All_Time:   0h 9m29s
(w_057) EP  101 | EP_Step:    63 | EP_Reward:    63.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 2s | All_Time:   0h 9m29s
(w_076) EP  118 | EP_Step:    74 | EP_Reward:    74.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 3s | All_Time:   0h 9m31s
(w_014) EP   97 | EP_Step:   200 | EP_Reward:   200.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 8s | All_Time:   0h 9m30s
(w_030) EP  111 | EP_Step:    76 | EP_Reward:    76.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 2s | All_Time:   0h 9m34s
(w_073) EP  117 | EP_Step:    82 | EP_Reward:    82.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 3s | All_Time:   0h 9m32s
(w_019) EP  107 | EP_Step:    76 | EP_Reward:    76.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 3s | All_Time:   0h 9m32s
(w_058) EP  108 | EP_Step:    80 | EP_Reward:    80.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 3s | All_Time:   0h 9m32s
(w_076) EP  119 | EP_Step:    62 | EP_Reward:    62.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 2s | All_Time:   0h 9m33s
(w_089) EP  114 | EP_Step:    93 | EP_Reward:    93.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 3s | All_Time:   0h 9m32s
(w_063) EP   93 | EP_Step:   195 | EP_Reward:   195.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 7s | All_Time:   0h 9m34s
(w_057) EP  102 | EP_Step:    73 | EP_Reward:    73.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 2s | All_Time:   0h 9m32s
(w_068) EP  112 | EP_Step:    88 | EP_Reward:    88.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 3s | All_Time:   0h 9m32s
(w_094) EP  104 | EP_Step:   175 | EP_Reward:   175.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 7s | All_Time:   0h 9m35s
(w_005) EP  112 | EP_Step:   200 | EP_Reward:   200.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 8s | All_Time:   0h 9m33s
(w_019) EP  108 | EP_Step:    49 | EP_Reward:    49.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 1s | All_Time:   0h 9m34s
(w_057) EP  103 | EP_Step:    27 | EP_Reward:    27.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 1s | All_Time:   0h 9m33s
(w_073) EP  118 | EP_Step:    53 | EP_Reward:    53.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 2s | All_Time:   0h 9m34s
(w_005) EP  113 | EP_Step:    17 | EP_Reward:    17.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 0s | All_Time:   0h 9m33s
(w_071) EP   92 | EP_Step:   200 | EP_Reward:   200.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 7s | All_Time:   0h 9m34s
(w_076) EP  120 | EP_Step:    71 | EP_Reward:    71.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 2s | All_Time:   0h 9m36s
(w_089) EP  115 | EP_Step:    76 | EP_Reward:    76.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 3s | All_Time:   0h 9m35s
(w_068) EP  113 | EP_Step:    87 | EP_Reward:    87.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 3s | All_Time:   0h 9m35s
(w_019) EP  109 | EP_Step:    65 | EP_Reward:    65.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 2s | All_Time:   0h 9m36s
(w_030) EP  112 | EP_Step:   126 | EP_Reward:   126.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 4s | All_Time:   0h 9m39s
(w_026) EP   92 | EP_Step:   200 | EP_Reward:   200.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 7s | All_Time:   0h 9m37s
(w_073) EP  119 | EP_Step:    79 | EP_Reward:    79.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 2s | All_Time:   0h 9m37s
(w_094) EP  105 | EP_Step:   112 | EP_Reward:   112.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 4s | All_Time:   0h 9m39s
(w_083) EP   90 | EP_Step:   200 | EP_Reward:   200.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 7s | All_Time:   0h 9m37s
(w_089) EP  116 | EP_Step:    48 | EP_Reward:    48.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 1s | All_Time:   0h 9m37s
(w_083) EP   91 | EP_Step:    11 | EP_Reward:    11.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 0s | All_Time:   0h 9m37s
(w_063) EP   94 | EP_Step:   146 | EP_Reward:   146.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 5s | All_Time:   0h 9m39s
(w_068) EP  114 | EP_Step:    58 | EP_Reward:    58.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 2s | All_Time:   0h 9m38s
(w_014) EP   98 | EP_Step:   200 | EP_Reward:   200.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 7s | All_Time:   0h 9m38s
(w_083) EP   92 | EP_Step:    23 | EP_Reward:    23.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 0s | All_Time:   0h 9m38s
(w_026) EP   93 | EP_Step:    59 | EP_Reward:    59.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 2s | All_Time:   0h 9m39s
(w_063) EP   95 | EP_Step:    27 | EP_Reward:    27.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 1s | All_Time:   0h 9m40s
(w_089) EP  117 | EP_Step:    64 | EP_Reward:    64.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 2s | All_Time:   0h 9m39s
(w_019) EP  110 | EP_Step:    89 | EP_Reward:    89.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 3s | All_Time:   0h 9m39s
(w_058) EP  109 | EP_Step:   187 | EP_Reward:   187.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 7s | All_Time:   0h 9m39s
(w_005) EP  114 | EP_Step:   150 | EP_Reward:   150.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 5s | All_Time:   0h 9m39s
(w_030) EP  113 | EP_Step:    83 | EP_Reward:    83.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 3s | All_Time:   0h 9m42s
(w_057) EP  104 | EP_Step:   157 | EP_Reward:   157.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 6s | All_Time:   0h 9m39s
(w_005) EP  115 | EP_Step:    15 | EP_Reward:    15.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 0s | All_Time:   0h 9m40s
(w_073) EP  120 | EP_Step:    97 | EP_Reward:    97.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 3s | All_Time:   0h 9m40s
(w_063) EP   96 | EP_Step:    43 | EP_Reward:    43.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 1s | All_Time:   0h 9m42s
(w_094) EP  106 | EP_Step:   102 | EP_Reward:   102.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 3s | All_Time:   0h 9m43s
(w_068) EP  115 | EP_Step:    68 | EP_Reward:    68.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 2s | All_Time:   0h 9m40s
(w_057) EP  105 | EP_Step:    48 | EP_Reward:    48.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 1s | All_Time:   0h 9m41s
(w_019) EP  111 | EP_Step:    49 | EP_Reward:    49.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 1s | All_Time:   0h 9m41s
(w_071) EP   93 | EP_Step:   200 | EP_Reward:   200.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 7s | All_Time:   0h 9m41s
(w_094) EP  107 | EP_Step:    44 | EP_Reward:    44.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 1s | All_Time:   0h 9m44s
(w_030) EP  114 | EP_Step:    72 | EP_Reward:    72.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 2s | All_Time:   0h 9m44s
(w_089) EP  118 | EP_Step:    96 | EP_Reward:    96.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 3s | All_Time:   0h 9m42s
(w_058) EP  110 | EP_Step:    94 | EP_Reward:    94.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 3s | All_Time:   0h 9m43s
(w_026) EP   94 | EP_Step:   110 | EP_Reward:   110.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 4s | All_Time:   0h 9m43s
(w_083) EP   93 | EP_Step:   153 | EP_Reward:   153.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 5s | All_Time:   0h 9m44s
(w_019) EP  112 | EP_Step:    91 | EP_Reward:    91.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 3s | All_Time:   0h 9m44s
(w_083) EP   94 | EP_Step:    18 | EP_Reward:    18.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 0s | All_Time:   0h 9m44s
(w_063) EP   97 | EP_Step:   128 | EP_Reward:   128.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 4s | All_Time:   0h 9m46s
(w_030) EP  115 | EP_Step:    69 | EP_Reward:    69.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 2s | All_Time:   0h 9m47s
(w_068) EP  116 | EP_Step:   124 | EP_Reward:   124.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 4s | All_Time:   0h 9m44s
(w_083) EP   95 | EP_Step:    18 | EP_Reward:    18.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 0s | All_Time:   0h 9m45s
(w_014) EP   99 | EP_Step:   200 | EP_Reward:   200.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 7s | All_Time:   0h 9m45s
(w_057) EP  106 | EP_Step:   140 | EP_Reward:   140.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 4s | All_Time:   0h 9m46s
(w_019) EP  113 | EP_Step:    55 | EP_Reward:    55.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 1s | All_Time:   0h 9m46s
(w_089) EP  119 | EP_Step:   114 | EP_Reward:   114.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 3s | All_Time:   0h 9m46s
(w_005) EP  116 | EP_Step:   187 | EP_Reward:   187.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 6s | All_Time:   0h 9m46s
(w_005) EP  117 | EP_Step:    13 | EP_Reward:    13.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 0s | All_Time:   0h 9m47s
(w_083) EP   96 | EP_Step:    76 | EP_Reward:    76.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 2s | All_Time:   0h 9m47s
(w_058) EP  111 | EP_Step:   136 | EP_Reward:   136.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 4s | All_Time:   0h 9m48s
(w_071) EP   94 | EP_Step:   200 | EP_Reward:   200.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 6s | All_Time:   0h 9m47s
(w_014) EP  100 | EP_Step:    82 | EP_Reward:    82.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 2s | All_Time:   0h 9m48s
(w_030) EP  116 | EP_Step:   106 | EP_Reward:   106.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 3s | All_Time:   0h 9m51s
(w_068) EP  117 | EP_Step:   106 | EP_Reward:   106.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 3s | All_Time:   0h 9m48s
(w_057) EP  107 | EP_Step:    71 | EP_Reward:    71.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 2s | All_Time:   0h 9m48s
(w_094) EP  108 | EP_Step:   187 | EP_Reward:   187.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 6s | All_Time:   0h 9m51s
(w_083) EP   97 | EP_Step:    34 | EP_Reward:    34.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 1s | All_Time:   0h 9m49s
(w_057) EP  108 | EP_Step:    10 | EP_Reward:    10.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 0s | All_Time:   0h 9m49s
(w_089) EP  120 | EP_Step:    70 | EP_Reward:    70.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 2s | All_Time:   0h 9m49s
(w_019) EP  114 | EP_Step:    81 | EP_Reward:    81.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 2s | All_Time:   0h 9m49s
(w_026) EP   95 | EP_Step:   200 | EP_Reward:   200.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 6s | All_Time:   0h 9m50s
(w_058) EP  112 | EP_Step:    77 | EP_Reward:    77.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 2s | All_Time:   0h 9m50s
(w_071) EP   95 | EP_Step:    83 | EP_Reward:    83.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 2s | All_Time:   0h 9m50s
(w_019) EP  115 | EP_Step:    56 | EP_Reward:    56.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 1s | All_Time:   0h 9m51s
(w_030) EP  117 | EP_Step:    78 | EP_Reward:    78.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 2s | All_Time:   0h 9m53s
(w_063) EP   98 | EP_Step:   184 | EP_Reward:   184.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 6s | All_Time:   0h 9m52s
(w_068) EP  118 | EP_Step:    83 | EP_Reward:    83.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 2s | All_Time:   0h 9m51s
(w_083) EP   98 | EP_Step:    76 | EP_Reward:    76.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 2s | All_Time:   0h 9m51s
(w_014) EP  101 | EP_Step:    99 | EP_Reward:    99.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 3s | All_Time:   0h 9m51s
(w_071) EP   96 | EP_Step:    30 | EP_Reward:    30.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 1s | All_Time:   0h 9m51s
(w_026) EP   96 | EP_Step:    84 | EP_Reward:    84.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 2s | All_Time:   0h 9m52s
(w_019) EP  116 | EP_Step:    55 | EP_Reward:    55.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 1s | All_Time:   0h 9m53s
(w_057) EP  109 | EP_Step:   127 | EP_Reward:   127.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 3s | All_Time:   0h 9m52s
(w_068) EP  119 | EP_Step:    59 | EP_Reward:    59.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 1s | All_Time:   0h 9m52s
(w_058) EP  113 | EP_Step:    93 | EP_Reward:    93.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 3s | All_Time:   0h 9m53s
(w_057) EP  110 | EP_Step:    22 | EP_Reward:    22.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 0s | All_Time:   0h 9m53s
(w_071) EP   97 | EP_Step:    51 | EP_Reward:    51.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 1s | All_Time:   0h 9m53s
(w_005) EP  118 | EP_Step:   200 | EP_Reward:   200.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 6s | All_Time:   0h 9m53s
(w_094) EP  109 | EP_Step:   149 | EP_Reward:   149.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 5s | All_Time:   0h 9m56s
(w_057) EP  111 | EP_Step:    17 | EP_Reward:    17.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 0s | All_Time:   0h 9m54s
(w_083) EP   99 | EP_Step:    82 | EP_Reward:    82.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 2s | All_Time:   0h 9m54s
(w_019) EP  117 | EP_Step:    48 | EP_Reward:    48.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 1s | All_Time:   0h 9m54s
(w_083) EP  100 | EP_Step:    11 | EP_Reward:    11.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 0s | All_Time:   0h 9m54s
(w_026) EP   97 | EP_Step:    65 | EP_Reward:    65.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 2s | All_Time:   0h 9m55s
(w_058) EP  114 | EP_Step:    46 | EP_Reward:    46.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 1s | All_Time:   0h 9m55s
(w_071) EP   98 | EP_Step:    47 | EP_Reward:    47.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 1s | All_Time:   0h 9m54s
(w_019) EP  118 | EP_Step:    55 | EP_Reward:    55.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 1s | All_Time:   0h 9m56s
(w_030) EP  118 | EP_Step:   181 | EP_Reward:   181.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 5s | All_Time:   0h 9m59s
(w_068) EP  120 | EP_Step:   125 | EP_Reward:   125.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 3s | All_Time:   0h 9m56s
(w_063) EP   99 | EP_Step:   200 | EP_Reward:   200.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 6s | All_Time:   0h 9m59s
(w_094) EP  110 | EP_Step:   119 | EP_Reward:   119.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 3s | All_Time:   0h10m 0s
(w_019) EP  119 | EP_Step:    55 | EP_Reward:    55.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 1s | All_Time:   0h 9m57s
(w_058) EP  115 | EP_Step:   108 | EP_Reward:   108.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 3s | All_Time:   0h 9m58s
(w_026) EP   98 | EP_Step:   113 | EP_Reward:   113.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 3s | All_Time:   0h 9m58s
(w_014) EP  102 | EP_Step:   200 | EP_Reward:   200.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 6s | All_Time:   0h 9m58s
(w_005) EP  119 | EP_Step:   150 | EP_Reward:   150.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 4s | All_Time:   0h 9m58s
(w_057) EP  112 | EP_Step:   163 | EP_Reward:   163.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 4s | All_Time:   0h 9m58s
(w_094) EP  111 | EP_Step:    53 | EP_Reward:    53.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 1s | All_Time:   0h10m 1s
(w_058) EP  116 | EP_Step:    56 | EP_Reward:    56.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 1s | All_Time:   0h 9m59s
(w_026) EP   99 | EP_Step:    58 | EP_Reward:    58.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 1s | All_Time:   0h 9m59s
(w_019) EP  120 | EP_Step:    77 | EP_Reward:    77.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 2s | All_Time:   0h10m 0s
(w_083) EP  101 | EP_Step:   200 | EP_Reward:   200.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 5s | All_Time:   0h10m 0s
(w_057) EP  113 | EP_Step:    56 | EP_Reward:    56.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 1s | All_Time:   0h10m 0s
(w_030) EP  119 | EP_Step:   144 | EP_Reward:   144.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 3s | All_Time:   0h10m 2s
(w_071) EP   99 | EP_Step:   200 | EP_Reward:   200.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 5s | All_Time:   0h10m 0s
(w_094) EP  112 | EP_Step:    76 | EP_Reward:    76.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 1s | All_Time:   0h10m 3s
(w_058) EP  117 | EP_Step:    57 | EP_Reward:    57.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 1s | All_Time:   0h10m 1s
(w_063) EP  100 | EP_Step:   135 | EP_Reward:   135.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 3s | All_Time:   0h10m 2s
(w_057) EP  114 | EP_Step:    54 | EP_Reward:    54.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 1s | All_Time:   0h10m 1s
(w_094) EP  113 | EP_Step:    47 | EP_Reward:    47.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 1s | All_Time:   0h10m 4s
(w_030) EP  120 | EP_Step:    82 | EP_Reward:    82.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 2s | All_Time:   0h10m 5s
(w_014) EP  103 | EP_Step:   200 | EP_Reward:   200.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 4s | All_Time:   0h10m 3s
(w_005) EP  120 | EP_Step:   200 | EP_Reward:   200.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 5s | All_Time:   0h10m 3s
(w_057) EP  115 | EP_Step:    75 | EP_Reward:    75.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 1s | All_Time:   0h10m 3s
(w_014) EP  104 | EP_Step:    31 | EP_Reward:    31.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 0s | All_Time:   0h10m 3s
(w_057) EP  116 | EP_Step:    20 | EP_Reward:    20.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 0s | All_Time:   0h10m 3s
(w_058) EP  118 | EP_Step:   139 | EP_Reward:   139.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 3s | All_Time:   0h10m 4s
(w_026) EP  100 | EP_Step:   200 | EP_Reward:   200.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 4s | All_Time:   0h10m 4s
(w_014) EP  105 | EP_Step:    41 | EP_Reward:    41.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 0s | All_Time:   0h10m 4s
(w_094) EP  114 | EP_Step:   115 | EP_Reward:   115.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 2s | All_Time:   0h10m 7s
(w_083) EP  102 | EP_Step:   200 | EP_Reward:   200.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 4s | All_Time:   0h10m 4s
(w_083) EP  103 | EP_Step:    12 | EP_Reward:    12.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 0s | All_Time:   0h10m 5s
(w_071) EP  100 | EP_Step:   200 | EP_Reward:   200.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 4s | All_Time:   0h10m 4s
(w_063) EP  101 | EP_Step:   180 | EP_Reward:   180.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 4s | All_Time:   0h10m 6s
(w_014) EP  106 | EP_Step:    37 | EP_Reward:    37.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 0s | All_Time:   0h10m 5s
(w_058) EP  119 | EP_Step:    65 | EP_Reward:    65.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 1s | All_Time:   0h10m 5s
(w_014) EP  107 | EP_Step:    25 | EP_Reward:    25.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 0s | All_Time:   0h10m 5s
(w_071) EP  101 | EP_Step:    30 | EP_Reward:    30.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 0s | All_Time:   0h10m 5s
(w_057) EP  117 | EP_Step:   125 | EP_Reward:   125.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 2s | All_Time:   0h10m 6s
(w_094) EP  115 | EP_Step:   118 | EP_Reward:   118.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 2s | All_Time:   0h10m 9s
(w_057) EP  118 | EP_Step:    41 | EP_Reward:    41.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 0s | All_Time:   0h10m 7s
(w_058) EP  120 | EP_Step:   111 | EP_Reward:   111.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 2s | All_Time:   0h10m 7s
(w_094) EP  116 | EP_Step:    31 | EP_Reward:    31.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 0s | All_Time:   0h10m10s
(w_083) EP  104 | EP_Step:   120 | EP_Reward:   120.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 2s | All_Time:   0h10m 7s
(w_057) EP  119 | EP_Step:    17 | EP_Reward:    17.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 0s | All_Time:   0h10m 7s
(w_063) EP  102 | EP_Step:   137 | EP_Reward:   137.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 2s | All_Time:   0h10m 9s
(w_057) EP  120 | EP_Step:    25 | EP_Reward:    25.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 0s | All_Time:   0h10m 8s
(w_026) EP  101 | EP_Step:   200 | EP_Reward:   200.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 4s | All_Time:   0h10m 8s
(w_063) EP  103 | EP_Step:    37 | EP_Reward:    37.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 0s | All_Time:   0h10m10s
(w_094) EP  117 | EP_Step:    62 | EP_Reward:    62.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 1s | All_Time:   0h10m11s
(w_063) EP  104 | EP_Step:    40 | EP_Reward:    40.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 0s | All_Time:   0h10m10s
(w_071) EP  102 | EP_Step:   200 | EP_Reward:   200.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 3s | All_Time:   0h10m 9s
(w_014) EP  108 | EP_Step:   200 | EP_Reward:   200.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 3s | All_Time:   0h10m 9s
(w_014) EP  109 | EP_Step:    19 | EP_Reward:    19.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 0s | All_Time:   0h10m 9s
(w_083) EP  105 | EP_Step:   200 | EP_Reward:   200.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 3s | All_Time:   0h10m10s
(w_063) EP  105 | EP_Step:   118 | EP_Reward:   118.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 1s | All_Time:   0h10m12s
(w_094) EP  118 | EP_Step:   169 | EP_Reward:   169.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 2s | All_Time:   0h10m13s
(w_026) EP  102 | EP_Step:   200 | EP_Reward:   200.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 3s | All_Time:   0h10m11s
(w_014) EP  110 | EP_Step:   149 | EP_Reward:   149.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 2s | All_Time:   0h10m12s
(w_083) EP  106 | EP_Step:    92 | EP_Reward:    92.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 1s | All_Time:   0h10m12s
(w_071) EP  103 | EP_Step:   196 | EP_Reward:   196.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 3s | All_Time:   0h10m12s
(w_071) EP  104 | EP_Step:    23 | EP_Reward:    23.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 0s | All_Time:   0h10m12s
(w_014) EP  111 | EP_Step:    61 | EP_Reward:    61.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 0s | All_Time:   0h10m13s
(w_094) EP  119 | EP_Step:   132 | EP_Reward:   132.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 1s | All_Time:   0h10m15s
(w_063) EP  106 | EP_Step:   162 | EP_Reward:   162.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 2s | All_Time:   0h10m15s
(w_071) EP  105 | EP_Step:    66 | EP_Reward:    66.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 0s | All_Time:   0h10m13s
(w_014) EP  112 | EP_Step:    83 | EP_Reward:    83.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 1s | All_Time:   0h10m14s
(w_014) EP  113 | EP_Step:     9 | EP_Reward:     9.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 0s | All_Time:   0h10m14s
(w_071) EP  106 | EP_Step:    36 | EP_Reward:    36.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 0s | All_Time:   0h10m13s
(w_094) EP  120 | EP_Step:    69 | EP_Reward:    69.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 0s | All_Time:   0h10m16s
(w_026) EP  103 | EP_Step:   200 | EP_Reward:   200.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 2s | All_Time:   0h10m14s
(w_014) EP  114 | EP_Step:    45 | EP_Reward:    45.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 0s | All_Time:   0h10m14s
(w_083) EP  107 | EP_Step:   200 | EP_Reward:   200.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 2s | All_Time:   0h10m15s
(w_014) EP  115 | EP_Step:    36 | EP_Reward:    36.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 0s | All_Time:   0h10m15s
(w_063) EP  107 | EP_Step:   120 | EP_Reward:   120.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 1s | All_Time:   0h10m16s
(w_014) EP  116 | EP_Step:    26 | EP_Reward:    26.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 0s | All_Time:   0h10m15s
(w_014) EP  117 | EP_Step:    11 | EP_Reward:    11.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 0s | All_Time:   0h10m15s
(w_071) EP  107 | EP_Step:   128 | EP_Reward:   128.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 1s | All_Time:   0h10m15s
(w_071) EP  108 | EP_Step:     8 | EP_Reward:     8.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 0s | All_Time:   0h10m15s
(w_071) EP  109 | EP_Step:    51 | EP_Reward:    51.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 0s | All_Time:   0h10m16s
(w_014) EP  118 | EP_Step:    81 | EP_Reward:    81.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 0s | All_Time:   0h10m16s
(w_071) EP  110 | EP_Step:    20 | EP_Reward:    20.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 0s | All_Time:   0h10m16s
(w_026) EP  104 | EP_Step:   200 | EP_Reward:   200.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 2s | All_Time:   0h10m17s
(w_063) EP  108 | EP_Step:   135 | EP_Reward:   135.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 1s | All_Time:   0h10m18s
(w_014) EP  119 | EP_Step:    22 | EP_Reward:    22.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 0s | All_Time:   0h10m16s
(w_071) EP  111 | EP_Step:    24 | EP_Reward:    24.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 0s | All_Time:   0h10m16s
(w_083) EP  108 | EP_Step:   200 | EP_Reward:   200.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 2s | All_Time:   0h10m17s
(w_014) EP  120 | EP_Step:    56 | EP_Reward:    56.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 0s | All_Time:   0h10m17s
(w_071) EP  112 | EP_Step:    87 | EP_Reward:    87.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 0s | All_Time:   0h10m17s
(w_026) EP  105 | EP_Step:   118 | EP_Reward:   118.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 1s | All_Time:   0h10m18s
(w_063) EP  109 | EP_Step:   200 | EP_Reward:   200.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 1s | All_Time:   0h10m20s
(w_071) EP  113 | EP_Step:   114 | EP_Reward:   114.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 1s | All_Time:   0h10m18s
(w_083) EP  109 | EP_Step:   200 | EP_Reward:   200.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 1s | All_Time:   0h10m19s
(w_071) EP  114 | EP_Step:    81 | EP_Reward:    81.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 0s | All_Time:   0h10m19s
(w_026) EP  106 | EP_Step:   200 | EP_Reward:   200.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 1s | All_Time:   0h10m20s
(w_063) EP  110 | EP_Step:   153 | EP_Reward:   153.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 1s | All_Time:   0h10m21s
(w_083) EP  110 | EP_Step:   115 | EP_Reward:   115.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 1s | All_Time:   0h10m20s
(w_071) EP  115 | EP_Step:   134 | EP_Reward:   134.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 1s | All_Time:   0h10m20s
(w_071) EP  116 | EP_Step:    37 | EP_Reward:    37.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 0s | All_Time:   0h10m20s
(w_063) EP  111 | EP_Step:   136 | EP_Reward:   136.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 1s | All_Time:   0h10m22s
(w_026) EP  107 | EP_Step:   198 | EP_Reward:   198.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 1s | All_Time:   0h10m22s
(w_083) EP  111 | EP_Step:   200 | EP_Reward:   200.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 1s | All_Time:   0h10m22s
(w_026) EP  108 | EP_Step:    55 | EP_Reward:    55.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 0s | All_Time:   0h10m22s
(w_063) EP  112 | EP_Step:   140 | EP_Reward:   140.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 1s | All_Time:   0h10m24s
(w_071) EP  117 | EP_Step:   200 | EP_Reward:   200.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 1s | All_Time:   0h10m22s
(w_026) EP  109 | EP_Step:   105 | EP_Reward:   105.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 0s | All_Time:   0h10m23s
(w_071) EP  118 | EP_Step:    33 | EP_Reward:    33.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 0s | All_Time:   0h10m23s
(w_071) EP  119 | EP_Step:    21 | EP_Reward:    21.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 0s | All_Time:   0h10m23s
(w_026) EP  110 | EP_Step:    47 | EP_Reward:    47.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 0s | All_Time:   0h10m23s
(w_071) EP  120 | EP_Step:    31 | EP_Reward:    31.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 0s | All_Time:   0h10m23s
(w_083) EP  112 | EP_Step:   200 | EP_Reward:   200.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 1s | All_Time:   0h10m23s
(w_063) EP  113 | EP_Step:   142 | EP_Reward:   142.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 1s | All_Time:   0h10m25s
(w_026) EP  111 | EP_Step:    71 | EP_Reward:    71.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 0s | All_Time:   0h10m24s
(w_026) EP  112 | EP_Step:    89 | EP_Reward:    89.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 0s | All_Time:   0h10m25s
(w_063) EP  114 | EP_Step:   152 | EP_Reward:   152.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 1s | All_Time:   0h10m26s
(w_026) EP  113 | EP_Step:    50 | EP_Reward:    50.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 0s | All_Time:   0h10m25s
(w_083) EP  113 | EP_Step:   200 | EP_Reward:   200.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 1s | All_Time:   0h10m25s
(w_026) EP  114 | EP_Step:    48 | EP_Reward:    48.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 0s | All_Time:   0h10m25s
(w_063) EP  115 | EP_Step:   160 | EP_Reward:   160.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 1s | All_Time:   0h10m27s
(w_026) EP  115 | EP_Step:    98 | EP_Reward:    98.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 0s | All_Time:   0h10m26s
(w_083) EP  114 | EP_Step:   200 | EP_Reward:   200.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 1s | All_Time:   0h10m26s
(w_026) EP  116 | EP_Step:   101 | EP_Reward:   101.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 0s | All_Time:   0h10m27s
(w_063) EP  116 | EP_Step:   147 | EP_Reward:   147.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 1s | All_Time:   0h10m28s
(w_026) EP  117 | EP_Step:    61 | EP_Reward:    61.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 0s | All_Time:   0h10m27s
(w_026) EP  118 | EP_Step:    49 | EP_Reward:    49.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 0s | All_Time:   0h10m27s
(w_083) EP  115 | EP_Step:   200 | EP_Reward:   200.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 1s | All_Time:   0h10m28s
(w_026) EP  119 | EP_Step:    57 | EP_Reward:    57.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 0s | All_Time:   0h10m28s
(w_063) EP  117 | EP_Step:   200 | EP_Reward:   200.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 1s | All_Time:   0h10m30s
(w_026) EP  120 | EP_Step:   114 | EP_Reward:   114.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 0s | All_Time:   0h10m29s
(w_083) EP  116 | EP_Step:   200 | EP_Reward:   200.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 1s | All_Time:   0h10m29s
(w_063) EP  118 | EP_Step:   161 | EP_Reward:   161.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 0s | All_Time:   0h10m30s
(w_083) EP  117 | EP_Step:   200 | EP_Reward:   200.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 0s | All_Time:   0h10m30s
(w_063) EP  119 | EP_Step:   190 | EP_Reward:   190.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 0s | All_Time:   0h10m31s
(w_083) EP  118 | EP_Step:   200 | EP_Reward:   200.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 0s | All_Time:   0h10m31s
(w_063) EP  120 | EP_Step:   180 | EP_Reward:   180.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 0s | All_Time:   0h10m32s
(w_083) EP  119 | EP_Step:    84 | EP_Reward:    84.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 0s | All_Time:   0h10m31s
(w_083) EP  120 | EP_Step:   200 | EP_Reward:   200.00 | MAX_R: 1.00  | epsilon: -0.0000 | EP_Time:   0m 0s | All_Time:   0h10m32s
t_83 get max reward =  68.81
'''
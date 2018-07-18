# Run gym with different seed
#   ex: 
#       python gym_try_seed.py DDPG_CartPole-v0.yaml
#       
# Author:  kbehouse  <https://github.com/kbehouse/>
#          

import sys, os
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)),'../../'))
from client_standalone import Client, EnvSpace, Server
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

class Server_try_seed(Server):
    def __init__(self, target_env_class, i_cfg, project_name=None, retrain_model = False):
        self.threadLock = Lock()
        
        all_t =[]
        for seed in range(1,100):  #39, 118
            s = seed
            t = ThreadWithReturnValue(target=self.gym_thread, args=(s,), name='t_'+ str(seed), )
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
    
    def gym_thread(self, seed):
    
        self.threadLock.acquire()
        # print('seed = ', seed)
        cfg_copy = copy.deepcopy(cfg) #cfg.deepcopy()
        self.threadLock.release()
        cfg_copy = gym_cfg(cfg_copy)
        # print('cfg_copy of seed=', seed)
        
        cfg_copy['misc']['random_seed'] = seed
        cfg_copy['misc']['render'] = False
        cfg_copy['misc']['gpu_memory_ratio'] = 0.76
        cfg_copy['misc']['max_ep'] = 150
        cfg_copy['misc']['worker_nickname'] = 'w_%03d' % (seed)
        prj_name ='gym-%s_seed_%04d' % ( get_yaml_name(), seed)
        
        cfg_copy['misc']['gym_monitor_path'] = None
        # cfg_copy['misc']['gym_monitor_path'] = set_gym_monitor_path(cfg_copy['misc']['gym_monitor_path_origin'], i_project_name=prj_name)
        # print("cfg['misc']['gym_monitor_path'] = ", cfg_copy['misc']['gym_monitor_path'])    
        worker = self.server_on_session(prj_name, cfg_copy, True)
        c = Client(GymBasic, project_name=prj_name, i_cfg = cfg_copy)
        c.set_worker(worker)
        c.set_callback_queue(worker.get_callback_queue())

    
        c.run()
        avg_r = c.env_space.worker.avg_ep_reward()
        return avg_r
    
if __name__ == '__main__':
    s = Server_try_seed(GymBasic, i_cfg = gym_cfg(cfg) , project_name='gym-' + get_yaml_name() ,retrain_model= True)
    

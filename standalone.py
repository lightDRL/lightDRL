import numpy as np
import asyncio
import websockets
import pickle 
import time

from client import LogRL
from server import ServerBase
    

class Standalone(LogRL, ServerBase):
    def __init__(self, i_cfg , project_name=None):
        self.log_init(i_cfg , project_name)
        self.build_worker(project_name, i_cfg)      # self.worker ready

    def run(self):
        self.env_init()
        while True:  # loop 1
            state = self.env_reset()
            a = self.worker.predict(state)

            while True: # loop 2, break when ep done
                step_action = np.argmax(a) if  self.cfg['RL']['action_discrete'] else a
                # self.log_time('before step')
                s, r, d, s_ = self.on_action_response(step_action) 
                # self.log_time('step')
                # self.log_time('before dic')
                dic ={'s': s, 'a': a, 'r': r, 'd': d, 's_': s_}
                # self.log_time('after dic')
                self.worker.train_process(dic)
                self.log_data_step(r)
                # self.log_time('train')
                if d:
                    ep, ep_use_steps, ep_reward, all_ep_sum_reward = self.log_data_done()    # loop 2 done
                    # print(f'[] ep = {ep}, ep_reward = {ep_reward},  all_ep_sum_reward = {all_ep_sum_reward}')                  
                    if hasattr(self, 'ep_done_cb'):
                        self.ep_done_cb(ep = ep, ep_reward = ep_reward, all_ep_sum_reward =  all_ep_sum_reward)
                    break
                else:
                    action = self.worker.predict(s_)
                    # print('before actoin  =', action)
                    a = self.worker.add_action_noise(action, r)
                    # print('a  =', a)

            if self.ep > self.cfg['misc']['max_ep']: # loop 1 done
                break 

    def set_ep_done_cb(self, cb):
        self.ep_done_cb = cb

    def __del__(self):
        if hasattr(self, 'env') and hasattr(self.env, 'close') and callable(getattr(self.env, 'close')):
            print('[I] env.close')
            self.env.close()
            if hasattr(self.env.env, 'close') and callable(getattr(self.env.env, 'close')):
                print('[I] env.env.close')
                self.env.env.close()
        
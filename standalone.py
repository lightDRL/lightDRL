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

            while True: # loop 2
                step_action = np.argmax(a) if  self.cfg['RL']['action_discrete'] else a
                s, r, d, s_ = self.on_action_response(step_action) 
                dic ={'s': s, 'a': a, 'r': r, 'd': d, 's_': s_}
                self.worker.train_process(dic)
                if d:
                    self.log_data_done()    # loop 2 done
                    break
                else:
                    action = self.worker.predict(s_)
                    # print('before actoin  =', action)
                    a = self.worker.add_action_noise(action, r)
                    # print('a  =', a)
                self.log_data_step(r)

            if self.ep > self.cfg['misc']['max_ep']: # loop 1 done
                break 


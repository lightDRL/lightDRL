import numpy as np
import asyncio
import websockets
import pickle 
import time

class LogRL:   
    def log_init(self, i_cfg , project_name=None):
        self.project_name = project_name
        self.cfg = i_cfg
        self.log_data_init()

        
    def log_data_init(self):
        self.start_time = time.time()
        self.ep_s_time = time.time()
        self.ep = 1#0
        self.ep_use_step = 0
        self.ep_reward = 0
        self.all_ep_reward = 0

    def log_data_step(self, r):
        self.ep_use_step += 1
        self.ep_reward += r
        self.all_ep_reward += r

    def log_data_done(self):
        ret_ep, ret_use_step, ret_reward, ret_all_ep_reward = self.ep, self.ep_use_step, self.ep_reward, self.all_ep_reward
        self.ep+=1
        self.ep_use_step = 0
        self.ep_reward = 0
        self.ep_s_time = time.time()  # update episode start time

        return ret_ep, ret_use_step, ret_reward, ret_all_ep_reward
      
    
    def log(self):
        print('EP:%5d, STEP:%4d, r: %7.2f, ep_t:%s, all_t:%s' % \
            ( self.ep,  self.ep_use_step, self.ep_reward, self.time_str(self.ep_s_time, True), self.time_str(self.start_time)))

    def time_str(self, start_time, min=False):
        use_secs = time.time() - start_time
        if min:
            return '%3dm%2ds' % (use_secs/60, use_secs % 60 )
        return  '%3dh%2dm%2ds' % (use_secs/3600, (use_secs%3600)/60, use_secs % 60 )

    def log_time(self, s):
        if hasattr(self, 'ts'):
            print(s + ' use time = ' + str( time.time() - self.ts  ) + ', time=' + str(time.time()))
        self.ts = time.time()

    

class Client(LogRL):
    def __init__(self, i_cfg , project_name=None):
        self.log_init(i_cfg , project_name)

    async def loop_step(self):
        async with websockets.connect('ws://localhost:8765') as websocket:
            print('[I] Client Init finish')
            await self.create_session(websocket)
            while True:
                state = self.env_init()
                a = await self.send_state_get_action(websocket, state)

                while True:
                    step_action = np.argmax(a) if  self.cfg['RL']['action_discrete'] else a
                    s, r, d, s_ = self.on_action_response(step_action)
                    a = await self.send_train_get_action(websocket, s, a, r, d, s_)
                    self.log_data_step(r)
                    if d:
                        self.log_data_done()
                        break

                if self.ep > self.cfg['misc']['max_ep']:
                    break 
    def run(self):
        print('in run')
        asyncio.get_event_loop().run_until_complete(self.loop_step())

    async def create_session(self, ws):
        dic ={'cmd':'new_session', 'project_name': self.project_name, 'config': self.cfg}
        dic_p = pickle.dumps(dic, -1) 
        await ws.send(dic_p)
        recv = await ws.recv()
        print(f"[I] Server's say {recv} !")

    async def send_state_get_action(self, ws, state):
        # state       = state.tolist()  if type(state) == np.ndarray else state # if type(state) != list else state
        dic ={'cmd':'predict', 's': state}
        dic_p = pickle.dumps(dic, -1) 
        # self.log_time('before send')
        await ws.send(dic_p)
        recv = await ws.recv()
        action = pickle.loads(recv)
        return action

    async def send_train_get_action(self, ws, state, action, reward, done,next_state):
        dic ={'cmd':'train_predict',
                's': state, 
                'a': action, #action, 
                'r': reward, 
                'd'  : done,
                's_': next_state}
        # self.log_time('before pickle')
        dic_p = pickle.dumps(dic, -1) 
        # self.log_time('pickle')
        # self.log_time('before send')
        await ws.send(dic_p)
        # self.log_time('before recv')
        recv = await ws.recv()
        action = pickle.loads(recv)
        # self.log_time('recv')
        return action

    
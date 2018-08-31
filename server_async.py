import os
# import shortuuid
import yaml
import numpy as np
import time
import sys
from server import ServerBase
from worker import WorkerBase

import asyncio
import websockets
import pickle

# def flappybird_cfg(cfg):
#     # of course, you colud set following in .yaml
#     cfg['RL']['state_discrete'] = True     
#     cfg['RL']['state_shape']  = (80,80,4)         
#     # action
#     cfg['RL']['action_discrete'] = True 
#     cfg['RL']['action_shape'] = (2,)
   
#     return cfg

# cfg = flappybird_cfg(cfg)


class Server(ServerBase):
      
    async def core(self, websocket, path):
        print(f'path={path}')
        sys.stdout.flush() 
        async for data in websocket:
            # data = await websocket.recv()
            p = pickle.loads(data)
            if p['cmd']=='new_session':
                self.build_worker(p['project_name'], p['config'])
                await websocket.send('Worker Ready')
            elif p['cmd']=='predict':
                predict = self.on_predict(p)
                predict_p = pickle.dumps(predict, -1) 
                await websocket.send(predict_p)
            elif p['cmd']=='train_predict':
                predict = self.on_train_and_predict(p)
                predict_p = pickle.dumps(predict, -1) 
                await websocket.send(predict_p)
            
    def run(self):
        start_server = websockets.serve(self.core, 'localhost', 8765)
        print('[I] Server is ready!')
        asyncio.get_event_loop().run_until_complete(start_server)
        asyncio.get_event_loop().run_forever()

    def build_worker(self, prj_name, cfg):
        # create tf graph & tf session
        tf_new_graph, tf_new_sess = self.create_new_tf_graph_sess(0.7, 3)
        # create logdir and save cfg
        model_log_dir = self.create_model_log_dir(prj_name, recreate_dir = True)
        with open(os.path.join(model_log_dir, 'config.yaml') , 'w') as outfile:
            yaml.dump(cfg, outfile, default_flow_style=False)

        self.worker = WorkerBase()
        self.worker.base_init(cfg, tf_new_graph, tf_new_sess, model_log_dir )


    def on_predict(self, data):
        action = self.worker.predict(data['s'])

        # print('type(action) = ', type(action))
        # action = action.tolist() if type(action) == np.ndarray else action
        return action

    def on_train_and_predict(self, data):
        # self.log_time('(S) [H]------get train data-------')
        self.worker.train_process(data)
        if not data['d']:
            action = self.worker.predict(data['s_'])
            action = self.worker.add_action_noise(action, data['r'])
            # action = action.tolist() if type(action) == np.ndarray else action
            # print('type(action) = ', type(action))
            # self.log_time('(S) train finish')
            return action
        else:
            # self.log_time('(S) train finish')
            return None


    def log_time(self, s):
        if hasattr(self, 'ts'):
            print(s + ' use time = ' + str( time.time() - self.ts  ) + ', time=' + str(time.time()))
        self.ts = time.time()
    
if __name__ == '__main__':
    s = Server()
    s.run()

    
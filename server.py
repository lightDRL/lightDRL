import os
# import shortuuid
import yaml
import numpy as np
import time
import sys
from worker import WorkerBase

import asyncio
import websockets
import pickle

import tensorflow as tf

from DRL.Base import RL,DRL
from DRL.DQN import DQN
from DRL.A3C import A3C
from DRL.DDPG import DDPG


DATA_POOL = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data_pool')
#------ check data_pool/ exist -------#
if not os.path.isdir(DATA_POOL):
    os.mkdir(DATA_POOL)

class ServerBase(object):
        # def check_output_graph(self):
    #     if 'misc' in cfg and cfg['misc']['output_tf']:create_new_tf_graph_sess
    #         log_dir = cfg['misc']['output_tf_dir']
    #         if os.path.exists(log_dir):
    #             shutil.rmtree(log_dir)
    #         tf.summary.FileWriter(log_dir, self.sess.graph)

    def create_new_tf_graph_sess(self, gpu_memory_ratio = None, random_seed=1234):
        tf_new_graph = tf.Graph()
        tf_new_graph.seed = random_seed

        if gpu_memory_ratio != None:
            config = tf.ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_ratio
            tf_new_sess = tf.Session(config=config, graph=tf_new_graph)
            print('[I] Create session with gpu memory raito: '+  str(gpu_memory_ratio) )
        else:
            tf_new_sess = tf.Session(graph=tf_new_graph)

        return tf_new_graph, tf_new_sess

    def create_model_log_dir(self, project_name, recreate_dir = False):
        # self.model_log_dir = '{}/{}/'.format(DATA_POOL, project_name)   if project_name != None else None
        model_log_dir = os.path.join(DATA_POOL, project_name) if project_name != None else None

        if model_log_dir !=None:
            if not os.path.isdir(model_log_dir):
                os.mkdir(model_log_dir)
                print('[I] create model_log_dir:' + model_log_dir)
            else:
                if recreate_dir:
                    from shutil import rmtree
                    rmtree(model_log_dir)
                    os.mkdir(model_log_dir)
                    print('[I] REcreate model_log_dir:' + model_log_dir)

        return model_log_dir

    def build_worker(self, prj_name, cfg): 
        # create tf graph & tf session
        tf_new_graph, tf_new_sess = self.create_new_tf_graph_sess(cfg['misc']['gpu_memory_ratio'], cfg['misc']['random_seed'])
        # create logdir and save cfg
        recreate_dir = cfg['misc']['model_retrain']
        model_log_dir = self.create_model_log_dir(prj_name, recreate_dir = recreate_dir)
        with open(os.path.join(model_log_dir, 'config.yaml') , 'w') as outfile:
            yaml.dump(cfg, outfile, default_flow_style=False)

        self.worker = WorkerBase()
        self.worker.base_init(cfg, tf_new_graph, tf_new_sess, model_log_dir )

        if cfg['misc']['redirect_stdout_2_file']:
            sys.stdout = open(model_log_dir +'/stdout.log', 'w')

        print('[I] worker ready!')

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

    
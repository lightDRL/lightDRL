from flask import Flask
from flask_restful import  Api
from flask_socketio import SocketIO, send, emit, Namespace

import os
import shortuuid
import yaml
import numpy as np
import time

from config import cfg
from server import ServerBase
from worker import WorkerBase

def flappybird_cfg(cfg):
    # of course, you colud set following in .yaml
    cfg['RL']['state_discrete'] = True     
    cfg['RL']['state_shape']  = (80,80,4)         
    # action
    cfg['RL']['action_discrete'] = True 
    cfg['RL']['action_shape'] = (2,)
   
    return cfg

cfg = flappybird_cfg(cfg)


class SocketServer(Namespace, ServerBase):
    def __init__(self, namespace = '/',  sock = None):
        super(SocketServer, self).__init__(namespace=namespace)
        self.socketio = sock
        # self.init_method()
        
        prj_name = 'severns'
        # create tf graph & tf session
        tf_new_graph, tf_new_sess = self.create_new_tf_graph_sess(0.7, 3)
        # create logdir and save cfg
        model_log_dir = self.create_model_log_dir(prj_name, recreate_dir = True)
        with open(os.path.join(model_log_dir, 'config.yaml') , 'w') as outfile:
            yaml.dump(cfg, outfile, default_flow_style=False)
        
        
        print('Build server RL session socket ')
        # (self, cfg, graph, sess, model_log_dir, net_scope = None)
        self.worker = WorkerBase()
        self.worker.base_init(cfg, tf_new_graph, tf_new_sess, model_log_dir )
        # if self.use_DRL and cfg['RL']['method']=='A3C':
        #     self.socketio.on_namespace(WorkerConn(ns, new_uuid, tf_new_sess, None) )
        # elif self.use_DRL :
        #     self.socketio.on_namespace(WorkerConn(ns, new_uuid, cfg, model_log_dir=model_log_dir, graph = tf_new_graph, sess = tf_new_sess) )
        # else:

        print('[I] Server is ready!')

    def on_connect(self):
        print('[I] In Server\'s on_connect()')

    def on_session(self, *data):
        print('[I] Server in on_session()')
        print('get data = ', data)

        emit('session_response', 'server session echo')

    def on_predict(self, data):
        action = self.worker.predict(data['state'])
        action = action.tolist() if type(action) == np.ndarray else action
        emit('predict_response', action)

    def on_train_and_predict(self, data):
        self.log_time('(S) [H]------get train data-------')
        self.worker.train_process(data)
        if not data['done']:
            action = self.worker.predict(data['next_state'])
            action = action.tolist() if type(action) == np.ndarray else action
            emit('predict_response', action)

        self.log_time('(S) train finish')

    def log_time(self, s):
        if hasattr(self, 'ts'):
            print(s + ' use time = ' + str( time.time() - self.ts  ) + ', time=' + str(time.time()))
        self.ts = time.time()
    
if __name__ == '__main__':
    #-------Flask Init--------#
    app = Flask(__name__, static_folder='static', static_url_path='')
    api = Api(app)
    
    #init socketio 
    socketio = SocketIO(app)
    socketio.on_namespace(SocketServer(namespace = '/',  sock =socketio))
    socketio.run(app, host='0.0.0.0')


    
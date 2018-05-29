from flask import Flask
from flask_restful import  Api
from flask_socketio import SocketIO, send, emit, Namespace

import os
import shortuuid
import yaml

# from config import cfg
from worker import Worker
from dashboard import Dashboard

from DRL.Base import RL,DRL
from DRL.A3C import A3C
from DRL.DDPG import DDPG
from DRL.TD import SARSA, QLearning
import tensorflow as tf
#-------set log level--------#
# import logging
# log = logging.getLogger('werkzeug')
# log.setLevel(logging.ERROR)

# default dir is same path as server.py
DATA_POOL = os.path.abspath(os.path.dirname(__file__) + '/data_pool/')
#------ check data_pool/ exist -------#
if not os.path.isdir(DATA_POOL):
    os.mkdir(DATA_POOL)

#------ Dynamic Namespce Predict -------#
class SocketServer(Namespace):
    def __init__(self, namespace = '/',  sock = None):
        super(SocketServer, self).__init__(namespace=namespace)
        self.socketio = sock
        # self.init_method()
        print('[I] Server is ready!')

    #------- for connect and get id------#
    def on_connect(self):
        print('[I] In Server\'s on_connect()')

    def on_session(self, *data):
        print('Server in on_session()')
        print('data = ', data)
        # print('data_2 = ', data_2)
        prj_name = data[0]
        cfg = data[1]

        # get method class
        method_class = globals()[cfg['RL']['method'] ]
        self.use_DRL = True if issubclass(method_class, DRL) else False
        # uuid & namespace
        new_uuid = shortuuid.uuid()
        ns = '/' + new_uuid + '/rl_session' 

        # create tf graph & tf session
        tf_new_graph, tf_new_sess = self.create_new_tf_graph_sess(cfg['misc']['gpu_memory_ratio'])
        # create logdir and save cfg
        model_log_dir = self.create_model_log_dir(prj_name)
        with open(os.path.join(model_log_dir, 'config.yaml') , 'w') as outfile:
            yaml.dump(cfg, outfile, default_flow_style=False)
        
        print('Build server RL session socket withs ns: {}'.format(ns))
        if self.use_DRL and cfg['RL']['method']=='A3C':
            self.socketio.on_namespace(Worker(ns, new_uuid, tf_new_sess, None) )
        elif self.use_DRL :
            self.socketio.on_namespace(Worker(ns, new_uuid, cfg, model_log_dir=model_log_dir, graph = tf_new_graph, sess = tf_new_sess) )
        # else:
        #     self.socketio.on_namespace(Worker(ns, new_uuid) )
        
        emit('session_response', new_uuid)
    
    '''    
    def init_method(self):
        # print('[I] Use {} Method'.format(cfg['RL']['method']))
        
        method_class = globals()[cfg['RL']['method'] ]
        self.use_DRL = True if issubclass(method_class, DRL) else False

        # A3C Init
        
        if issubclass(method_class, DRL): 
            self.sess = tf.Session()
            method = cfg['RL']['method']
            if method == 'A3C':
                self.dnn_main_net = method_class(self.sess, cfg[method]['dnn_scope'])  
            # else:
            #     self.dnn_main_net = method_class(self.sess)
        
        elif issubclass(method_class, RL):
            pass
        else:
            print('E: Worker::__init__() say error method name={}'.format(cfg['RL']['method'] ))
        
    
        # print('self.use_DRL = {}'.format(self.use_DRL))
        # DL Init 2
        # if self.use_DRL and method == 'A3C':
        if method == 'A3C':
            # COORD = tf.train.Coordinator()
            self.sess.run(tf.global_variables_initializer())
            # self.check_output_graph()
    '''    
        
    # def check_output_graph(self):
    #     if 'misc' in cfg and cfg['misc']['output_tf']:
    #         import os, shutil
    #         log_dir = cfg['misc']['output_tf_dir']
    #         if os.path.exists(log_dir):
    #             shutil.rmtree(log_dir)
    #         tf.summary.FileWriter(log_dir, self.sess.graph)

    def create_new_tf_graph_sess(self, gpu_memory_ratio = 0.2):
        tf_new_graph = tf.Graph()

        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_ratio
        tf_new_sess = tf.Session(config=config, graph=tf_new_graph)
        print('[I] Create session with gpu memory raito: '+  str(gpu_memory_ratio) )

        return tf_new_graph, tf_new_sess

    def create_model_log_dir(self, project_name):
        # self.model_log_dir = '{}/{}/'.format(DATA_POOL, project_name)   if project_name != None else None
        model_log_dir = os.path.join(DATA_POOL, project_name) if project_name != None else None
        
        if model_log_dir !=None:
            if not os.path.isdir(model_log_dir):
                os.mkdir(model_log_dir)

        print('[I] model_log_dir:' + model_log_dir)
        return model_log_dir



    

if __name__ == '__main__':
    #-------Flask Init--------#
    app = Flask(__name__, static_folder='static', static_url_path='')
    api = Api(app)
    # you can try on http://localhost:5000/dashboard
    api.add_resource(Dashboard,'/dashboard')

    #init socketio 
    socketio = SocketIO(app)
    socketio.on_namespace(SocketServer(namespace = '/',  sock =socketio))
    socketio.run(app, host='0.0.0.0')


    
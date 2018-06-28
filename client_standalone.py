import sys, os, time
import numpy as np
import json
import gym
from config import cfg

import signal                   # for ctrl+C
import sys                      # for ctrl+C

# ---------------for standalone! combine sever, worker in one file-----------------
import tensorflow as tf
import yaml
import threading
from worker import WorkerStandalone

from client import EnvBase

# default dir is same path as server.py
DATA_POOL = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data_pool')
#------ check data_pool/ exist -------#
if not os.path.isdir(DATA_POOL):
    os.mkdir(DATA_POOL)


class EnvSpace(EnvBase):

    # def env_init(self):
    #     pass
    def __init__(self):
        self.envbase_init()


    #--------------------------EnvSpace for standalone----------------------#
    def emit(self, event_name, data):
        if event_name=='train_and_predict':
            threading.Thread(target=(lambda: self.worker.on_train_and_predict(data))).start()
        elif event_name=='predict':
            threading.Thread(target=(lambda: self.worker.on_predict(data))).start()
        else:
            print("emit() say Error event_name = ", event_name)
            
    def from_main_thread_blocking(self):
        callback_action = self.callback_queue.get() #blocks until an item is available
        # callback()
        # print('callback_action = ', callback_action)
        self.on_predict_response(callback_action)

    # def from_main_thread_nonblocking(self):
    #     while True:
    #         try:
    #             callback = self.callback_queue.get(False) #doesn't block
    #         except Queue.Empty: #raised when queue is empty
    #             break
    #         callback()

    #---------------------------Server, worker re-implementation ------------------#

    def standalone_init(self, project_name, send_cfg, retrain_model):
        self.server_on_session(project_name, send_cfg, retrain_model )
        

    def server_on_session(self, *data):
        print('[I] Standalone Server in on_session()')

        prj_name = data[0]
        cfg = data[1]
        retrain_model = data[2]
        
        # create tf graph & tf session
        tf_new_graph, tf_new_sess = self.server_create_new_tf_graph_sess(cfg['misc']['gpu_memory_ratio'], cfg['misc']['random_seed'])
        # create logdir and save cfg
        model_log_dir = self.server_create_model_log_dir(prj_name, recreate_dir = retrain_model)
        with open(os.path.join(model_log_dir, 'config.yaml') , 'w') as outfile:
            yaml.dump(cfg, outfile, default_flow_style=False)
        
        import Queue

        self.callback_queue = Queue.Queue()
        self.worker = WorkerStandalone(cfg, main_queue = self.callback_queue,
                    model_log_dir=model_log_dir, graph = tf_new_graph, sess = tf_new_sess)


    def server_create_new_tf_graph_sess(self, gpu_memory_ratio = 0.2, random_seed=1234):
        tf_new_graph = tf.Graph()
        tf_new_graph.seed = random_seed

        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_ratio
        tf_new_sess = tf.Session(config=config, graph=tf_new_graph)
        print('[I] Create session with gpu memory raito: '+  str(gpu_memory_ratio) )

        return tf_new_graph, tf_new_sess

    def server_create_model_log_dir(self, project_name, recreate_dir = False):
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

    

class Client:
    def __init__(self, target_env_class, project_name=None,i_cfg = None, retrain_model = False):
        # Thread.__init__(self)
        self.target_env_class = target_env_class
        self.env_name = project_name

        send_cfg = cfg if i_cfg == None else cfg
        #self.socketIO.emit('session', project_name, cfg)  
        # self.socketIO.emit('session', project_name, send_cfg, retrain_model)  
        self.env_space = self.target_env_class()
        self.env_space.standalone_init(project_name, send_cfg, retrain_model)
        self.env_space.env_init()

        print('--------> after env_init()')
        while True:
            self.env_space.from_main_thread_blocking()
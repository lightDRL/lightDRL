import sys, os, time
import numpy as np
import json
import gym
# from config import cfg
# ---------------for standalone! combine sever and worker in one file-----------------
import tensorflow as tf
import yaml
import threading
from worker import WorkerStandalone
from server import ServerBase
from client import EnvBase

# default dir is same path as server.py
DATA_POOL = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data_pool')
#------ check data_pool/ exist -------#
if not os.path.isdir(DATA_POOL):
    os.mkdir(DATA_POOL)


class EnvSpace(EnvBase, ServerBase):

    # def env_init(self):
    #     pass
    def __init__(self):
        self.envbase_init()


    #--------------------------EnvSpace for standalone----------------------#
    # GymBasic.send_state_get_action->
    #   self.emit('predict',dic)
    # EnvSpace.emit()
    #   threading.Thread(target=(lambda: self.worker.on_predict(data))).start()
    # worker.on_predict()
    #   self.main_queue.put(action)

    def emit(self, event_name, data):
        if event_name=='train_and_predict':
            threading.Thread(target=(lambda: self.worker.on_train_and_predict(data))).start()
        elif event_name=='predict':
            threading.Thread(target=(lambda: self.worker.on_predict(data))).start()
        else:
            print("emit() say Error event_name = ", event_name)
            
    def from_main_thread_blocking(self):
        callback_action = self.callback_queue.get() #blocks until an item is available
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
        tf_new_graph, tf_new_sess = self.create_new_tf_graph_sess(cfg['misc']['gpu_memory_ratio'], cfg['misc']['random_seed'])
        # create logdir and save cfg
        model_log_dir = self.create_model_log_dir(prj_name, recreate_dir = retrain_model)
        with open(os.path.join(model_log_dir, 'config.yaml') , 'w') as outfile:
            yaml.dump(cfg, outfile, default_flow_style=False)
        
        import Queue

        self.callback_queue = Queue.Queue()
        self.worker = WorkerStandalone(cfg, main_queue = self.callback_queue,
                    model_log_dir=model_log_dir, graph = tf_new_graph, sess = tf_new_sess)


class Client:
    def __init__(self, target_env_class, i_cfg, project_name=None, retrain_model = False):
        self.target_env_class = target_env_class
        self.env_name = project_name

        # print('i_cfg=',i_cfg)
        self.target_env_class = target_env_class  
        self.env_space = self.target_env_class()
        self.env_space.set_cfg(i_cfg)
        self.env_space.standalone_init(project_name, i_cfg, retrain_model)
        self.env_space.env_init()

    def run(self):
        while True:
            if self.env_space.worker.is_done:
                break
            self.env_space.from_main_thread_blocking()

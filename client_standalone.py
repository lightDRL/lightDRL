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
import copy

from DRL.component.utils import print_tf_var

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
        

    def set_worker(self, i_worker):
        self.worker = i_worker

    def set_callback_queue(self, i_cb_queue):
        self.callback_queue = i_cb_queue
    #--------------------------EnvSpace for standalone----------------------#
    # GymBasic.send_state_get_action->
    #   self.emit('predict',dic)
    # EnvSpace.emit()
    #   threading.Thread(target=(lambda: self.worker.on_predict(data))).start()
    # worker.on_predict()
    #   self.main_queue.put(action)
    # def get_callback_queue(self):
    #     return self.callback_queue

    def emit(self, event_name, data):
        if event_name=='train_and_predict':
            threading.Thread(target=(lambda: self.worker.on_train_and_predict(data))).start()
        elif event_name=='predict':
            threading.Thread(target=(lambda: self.worker.on_predict(data))).start()
        else:
            print("emit() say Error event_name = ", event_name)
            
    def from_main_thread_blocking(self):
        callback_action = self.callback_queue.get() #blocks until an item is available

        # if self.ep <= (self.cfg['misc']['max_ep']):
        if callback_action!='WORKER_GET_DONE':
            self.on_predict_response(callback_action)
        # else:
        #     print('--------in WORKER_DONE----------')

    # def from_main_thread_nonblocking(self):
    #     while True:
    #         try:
    #             callback = self.callback_queue.get(False) #doesn't block
    #         except Queue.Empty: #raised when queue is empty
    #             break
    #         callback()

    #---------------------------Server, worker re-implementation ------------------#

    # def standalone_init(self, project_name, send_cfg, retrain_model):
        # self.server_on_session(project_name, send_cfg, retrain_model )
    


class Client:
    def __init__(self, target_env_class, i_cfg, project_name):
        # np.random.seed(i_cfg['misc']['random_seed'])
        self.target_env_class = target_env_class
        self.env_name = project_name

        # print('i_cfg=',i_cfg)
        self.target_env_class = target_env_class  
        self.env_space = self.target_env_class()
        self.env_space.set_cfg(i_cfg)
        # self.env_space.standalone_init(project_name, i_cfg, retrain_model)
        # self.env_space.env_init()

    def run(self):
        self.env_space.env_init()
        while True:
            # print('self.env_space.worker.is_max_ep= ', self.env_space.worker.is_max_ep)
            # print('self.env_space.worker.ep= ', self.env_space.worker.ep)       
            self.env_space.from_main_thread_blocking()
            # print('self.env_space.worker.is_max_ep= ', self.env_space.worker.is_max_ep)
            # print('self.env_space.worker.ep= ', self.env_space.worker.ep)
            if self.env_space.worker.is_max_ep:
                break
            # if self.env_space.ep > (self.env_space.cfg['misc']['max_ep']):
            #     break 

    def set_worker(self, i_worker):
        self.env_space.set_worker(i_worker)

    def set_callback_queue(self, i_cb_queue):
        self.env_space.set_callback_queue( i_cb_queue)


class Server(ServerBase):
    def __init__(self, target_env_class, i_cfg, project_name=None, retrain_model = True):
        self.best_avg_reward = -9999
        if i_cfg['RL']['method']=='A3C':  # for multiple worker
            tf_graph, tf_sess = self.create_new_tf_graph_sess(i_cfg['misc']['gpu_memory_ratio'], i_cfg['misc']['random_seed'])
            # build main_net
            model_log_dir = self.server_create_log_dir(i_cfg, project_name, retrain_model)
            main_worker = self.server_on_session( i_cfg, model_log_dir, tf_graph, tf_sess, i_cfg['A3C']['main_net_scope'])
            
            # self.cond_main_wait_other_ready = threading.Condition()
            self.worker_ready = 0   
            cond = threading.Condition()  

            
            self.threadLock = threading.Lock()
            all_thread =[]
            # print("i_cfg['A3C']['worker_num'] = ", i_cfg['A3C']['worker_num'])
            for i in range(i_cfg['A3C']['worker_num']):  #39, 118
                
                net_scope = 'net_%03d' % i
                sync_model_log_dir = model_log_dir + '_%03d' % i
                self.create_model_log_dir(sync_model_log_dir, recreate_dir = retrain_model)
                t = threading.Thread(target=self.asyc_thread, 
                                    args=(target_env_class,i_cfg, sync_model_log_dir, i, tf_graph, tf_sess, net_scope, cond, ),
                                     name='t_'+ str(i))
                t.start()
                all_thread.append(t)
            # wait all sub thread build ready
            while True:
                time.sleep(0.5)
                print('main -> self.worker_ready = ', self.worker_ready)
                if self.worker_ready>= i_cfg['A3C']['worker_num']:
                    break
            # print_tf_var(graph = tf_graph)  
            with tf_graph.as_default():          
                main_worker.RL.init_or_restore_model(tf_sess)
            # run all asyc_thread
            cond.acquire()
            cond.notify_all()
            cond.release()
            
            
            for t in all_thread:
                t.join()

            # print('Best ep avg reward = ', self.asyc_best_reward)

        else:
            model_log_dir = self.server_create_log_dir(i_cfg, project_name, retrain_model)
            worker = self.server_on_session(i_cfg, model_log_dir)

            c = Client(target_env_class, i_cfg = i_cfg, project_name=project_name)
            c.set_worker(worker)
            c.set_callback_queue(worker.get_callback_queue())
            c.run()
            # print('finish ')
            self.best_avg_reward = c.env_space.worker.avg_ep_reward_show()
            
    def asyc_thread(self, env_class, cfg, model_log_dir,  thread_id, tf_graph, tf_sess, net_scope, cond):
        self.threadLock.acquire()
        cfg_copy = copy.deepcopy(cfg) 
        cfg_copy['misc']['worker_nickname'] += '-asyc-%03d' % (thread_id)
        if cfg_copy['misc']['random_seed']!=None: # let use diff  seed
            cfg_copy['misc']['random_seed'] += thread_id * 5

        print("cfg_copy['misc']['random_seed'] = ", cfg_copy['misc']['random_seed'])       
        if thread_id == 0:
            # only output the thread id = 0 to monitor
            if cfg_copy['misc']['gym_monitor_path']!=None:
                from config import set_gym_monitor_path
                cfg_copy['misc']['gym_monitor_path'] = set_gym_monitor_path(cfg['misc']['gym_monitor_path'])
                cfg_copy['misc']['gym_monitor_path'] += '%03d' % thread_id
                print('monitor path = ', cfg_copy['misc']['gym_monitor_path'])
        else:
            cfg_copy['misc']['gym_monitor_path'] = None
            cfg_copy['misc']['render'] = False
        worker = self.server_on_session(cfg_copy, model_log_dir, tf_graph, tf_sess, net_scope)
        self.worker_ready += 1
        
        self.threadLock.release()
        
        prj_name = 'asyc-%03d' % (thread_id)
        c = Client(env_class, project_name=prj_name, i_cfg = cfg_copy)
        c.set_worker(worker)
        c.set_callback_queue(worker.get_callback_queue())
        # start wait
        cond.acquire()
        cond.wait()         # block here, wait all graph build finish
        cond.release()    
        # go run
        c.run()
        avg_r = c.env_space.worker.avg_ep_reward_show()
        self.threadLock.acquire()
        # self.asyc_best_reward = avg_r if avg_r > self.asyc_best_reward else self.asyc_best_reward
        self.best_avg_reward = avg_r if avg_r > self.best_avg_reward else self.best_avg_reward
        self.threadLock.release()

    def server_create_log_dir(self, cfg, dir_name, recreate_dir = True):
        # create logdir and save cfg
        model_log_dir = self.create_model_log_dir(dir_name, recreate_dir = recreate_dir)
        with open(os.path.join(model_log_dir, 'config.yaml') , 'w') as outfile:
            yaml.dump(cfg, outfile, default_flow_style=False)

        return model_log_dir
        

    def server_on_session(self, cfg, model_log_dir, tf_graph = None, tf_sess = None, net_scope = None):
        print('[I] Standalone Server in on_session()')

        # create tf graph & tf session
        if tf_graph == None or tf_sess == None:
            # create new graph & new session
            tf_graph, tf_sess = self.create_new_tf_graph_sess(cfg['misc']['gpu_memory_ratio'], cfg['misc']['random_seed'])
             
        worker = WorkerStandalone(cfg,
                    model_log_dir=model_log_dir, graph = tf_graph, sess = tf_sess, net_scope = net_scope)
        
        return worker
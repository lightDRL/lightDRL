import numpy as np
import asyncio
import websockets
import pickle 
import time

from client import LogRL
from server import ServerBase
import threading
import copy
from worker import WorkerBase

def standalone_switch( standalone_env_class, cfg , project_name):
    if cfg['RL']['method']=='A3C':
        s = StandaloneAsync(standalone_env_class, cfg, project_name=project_name)
    else:
        s = standalone_env_class(cfg, project_name=project_name)
        s.set_success(threshold_r = cfg['misc']['threshold_r'], threshold_successvie_count = cfg['misc']['threshold_successvie_count'])
        # s.run()

    return s

class Standalone(LogRL, ServerBase):
    def __init__(self, i_cfg , project_name=None, worker = None):
        self.log_init(i_cfg , project_name)
        if worker==None:
            self.build_worker(project_name, i_cfg)      # self.worker ready
        else:
            self.worker = worker

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
                # print('43, self.ep_use_step = ', self.ep_use_step)
                # self.log_time('before dic')
                if not self.cfg['misc']['evaluation']:
                    dic ={'s': s, 'a': a, 'r': r, 'd': d, 's_': s_}
                    # self.log_time('after dic')
                    self.worker.train_process(dic)
                
                self.log_data_step(r)
                    
                # self.log_time('train')
                if d:
                    ep, ep_use_steps, ep_reward, all_ep_sum_reward, is_success = self.log_data_done()    # loop 2 done
                    # print(f'[] ep = {ep}, ep_reward = {ep_reward},  all_ep_sum_reward = {all_ep_sum_reward}')                  
                    if hasattr(self, 'ep_done_cb'):
                        self.ep_done_cb(ep = ep, ep_reward = ep_reward, all_ep_sum_reward =  all_ep_sum_reward)
                    
                    if self.cfg['misc']['evaluation']:
                        if not hasattr(self, 'evaluation_ep'):
                            self.evaluation_ep = 0
                        
                        self.evaluation_ep +=1
                        self.evaluation_done_cb(ep = self.evaluation_ep, terminal_reward = r, ep_use_steps = ep_use_steps)
                    break
                else:
                    action = self.worker.predict(s_)
                    # print('before actoin  =', action)
                    a = self.worker.add_action_noise(action, r)
                    # print('a  =', a)

            if self.ep > self.cfg['misc']['max_ep']: # loop 1 done
                
                    # print(f'[] ep = {ep}, ep_reward = {ep_reward},  all_ep_sum_reward = {all_ep_sum_reward}')                  
                if hasattr(self, 'ep_done_cb'):
                    ep, ep_use_steps, ep_reward, all_ep_sum_reward, is_success = self.log_data_done()    # loop 2 done
                    self.ep_done_cb(ep = ep, ep_reward = 0, all_ep_sum_reward =  all_ep_sum_reward)
                break
 
            elif is_success:
                if self.cfg['misc']['threshold_successvie_break']: 
                    print('{} First success EP = {}, use_time = {:.2f}'.format(self.project_name, self.ep, time.time() - self.start_time) )
                    break
            elif self.cfg['misc']['evaluation']:
                if self.evaluation_ep >=self.cfg['misc']['evaluation_ep']:
                    break

                # break 

        print('Use tiem =',  time.time() - self.start_time)

    def set_ep_done_cb(self, cb):
        self.ep_done_cb = cb

    def get_success(self):
        return self.check_success()

    def __del__(self):
        if hasattr(self, 'env') and hasattr(self.env, 'close') and callable(getattr(self.env, 'close')):
            print('[I] env.close')
            self.env.close()
            if hasattr(self.env.env, 'close') and callable(getattr(self.env.env, 'close')):
                print('[I] env.env.close')
                self.env.env.close()
        
class StandaloneAsync(ServerBase):
    def __init__(self, target_env_class, i_cfg ,  project_name=None):
        self.prj_name = project_name
        self.tf_graph, self.tf_sess = self.create_new_tf_graph_sess(i_cfg['misc']['gpu_memory_ratio'], i_cfg['misc']['random_seed'])
        # build main_net
        self.model_log_dir = self.create_model_log_dir(project_name, i_cfg['misc']['model_retrain'])
        self.main_worker = self.build_worker( i_cfg, self.model_log_dir, self.tf_graph, self.tf_sess, i_cfg['A3C']['main_net_scope'])
        
        if i_cfg['misc']['redirect_stdout_2_file']:
            import sys
            sys.stdout = open(self.model_log_dir +'/_stdout.log', 'w')

        # self.cond_main_wait_other_ready = threading.Condition()
        self.worker_ready = 0   
        self.cond = threading.Condition()  
        
        self.threadLock = threading.Lock()
        self.all_thread =[]
        # print("i_cfg['A3C']['worker_num'] = ", i_cfg['A3C']['worker_num'])
        for i in range(i_cfg['A3C']['worker_num']):  #39, 118
            
            net_scope = 'net_%03d' % i
            sync_model_log_dir = self.model_log_dir + '_%03d' % i
            self.create_model_log_dir(sync_model_log_dir, recreate_dir = i_cfg['misc']['model_retrain'])
            t = threading.Thread(target=self.asyc_thread, 
                                args=(target_env_class,i_cfg, sync_model_log_dir, i, self.tf_graph, self.tf_sess, net_scope, self.cond, ),
                                    name='t_'+ str(i))
            # t.start()
            t.daemon = True
            self.all_thread.append(t)

        self.cfg = i_cfg

        self.success_best_less_ep = 99999999999
        self.success_best_less_ep_thread_id = -1


    def run(self):
        for t in self.all_thread:
            t.start()

        while True:
            time.sleep(0.5)
            print('main -> self.worker_ready = ', self.worker_ready)
            if self.worker_ready>= self.cfg['A3C']['worker_num']:
                break
        # print_tf_var(graph = tf_graph)  
        with self.tf_graph.as_default():          
            self.main_worker.init_or_restore_model(self.tf_sess)
        # run all asyc_thread
        self.cond.acquire()
        self.cond.notify_all()
        self.cond.release()


        while True:
            time.sleep(1)
            is_alive_list = [t.is_alive() for t in self.all_thread]
            # print(is_alive_list)
            if not all(is_alive_list):
                break

        self.threadLock.acquire()
        self.main_worker.save_model(self.model_log_dir, self.success_best_less_ep)
        self.threadLock.release()
        print('success_thread_id = %d, success_best_less_ep=%d' % (self.success_best_less_ep_thread_id, self.success_best_less_ep))

    def asyc_thread(self, env_class, cfg, model_log_dir,  thread_id, tf_graph, tf_sess, net_scope, cond):
        self.threadLock.acquire()
        
        print('--------- in asyc_thread--------')

        cfg_copy = self.modify_cfg(cfg, thread_id)
        

        print('in asyc_thread net_scope= ', net_scope)
        #-----build async worker run with env----#
        # print("cfg['RL']['method']=", cfg['RL']['method'])
        worker = WorkerBase()
        worker.base_init(cfg_copy, model_log_dir=model_log_dir, graph = tf_graph, sess = tf_sess, net_scope = net_scope)
        worker.RL.set_thread_lock(self.threadLock)
        self.worker_ready += 1
        self.threadLock.release()
        # worker = self.build_worker( cfg, model_log_dir, tf_graph, tf_sess, net_scope)
        
        # if cfg['misc']['redirect_stdout_2_file']:
        #     sys.stdout = open(model_log_dir +'/_stdout.log', 'w')

        s = env_class(cfg_copy , project_name=self.prj_name + '-asyc-%03d' % (thread_id), worker=worker)
        s.set_success(threshold_r = cfg_copy['misc']['threshold_r'], threshold_successvie_count = cfg_copy['misc']['threshold_successvie_count'])

        # start wait
        cond.acquire()
        cond.wait()         # block here, wait all graph build finish
        cond.release()    
        # go run
        s.run()

        successvie_count_max, first_over_threshold_ep = s.check_success()

        # get less ep
        if first_over_threshold_ep < self.success_best_less_ep:
            self.threadLock.acquire()
            self.success_best_less_ep = s.ep
            self.success_best_less_ep_thread_id = thread_id
            self.start_time = s.start_time 
            self.use_time = time.time()-s.start_time 
            self.all_ep_reward = s.all_ep_reward 
            self.is_success = s.is_success 

            self.successvie_count_max = successvie_count_max
            self.first_over_threshold_ep = first_over_threshold_ep
            self.reward_list = s.reward_list
            self.threshold_success_time = s.threshold_success_time
            
            self.threadLock.release()

       
        # avg_r = c.env_space.worker.avg_ep_reward_show()
        # self.threadLock.acquire()
        # # self.asyc_best_reward = avg_r if avg_r > self.asyc_best_reward else self.asyc_best_reward
        # self.best_avg_reward = avg_r if avg_r > self.best_avg_reward else self.best_avg_reward
        # self.threadLock.release()

    def check_success(self):
        return self.successvie_count_max, self.first_over_threshold_ep

    def modify_cfg(self, cfg, thread_id):
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
            
        return cfg_copy
        
    def build_worker(self, cfg, model_log_dir, tf_graph = None, tf_sess = None, net_scope = None):
        print('[I] Standalone Server in on_session()')

        # create tf graph & tf session
        if tf_graph == None or tf_sess == None:
            # create new graph & new session
            tf_graph, tf_sess = self.create_new_tf_graph_sess(cfg['misc']['gpu_memory_ratio'], cfg['misc']['random_seed'])
             
        worker = WorkerBase()
        worker.base_init(cfg, model_log_dir=model_log_dir, graph = tf_graph, sess = tf_sess, net_scope = net_scope)
        return worker


    @property
    def ep(self):
        return self.success_best_less_ep
    
    
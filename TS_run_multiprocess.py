# Tompson Sampling (TS) with multiprocessing
from multiprocessing import Process, Value, Lock, Manager, Pipe
import sys, os
import time
from ctypes import c_bool
from os.path import basename, splitext
from config import load_config
import  numpy as np
import random

MAX_EP = 2000

class Thompson(object):
    def __init__(self):
        self.a = 1
        self.b = 1
        self.n = 0
        
    def sample(self):
        # if self.a <=0 or self.b<=0:
        #     return random.random()
        # else:
        return np.random.beta(self.a, self.b)
        # beta_a = 1 if self.a <=1 else self.a
        # beta_b = 1 if self.b <=1 else self.b
        # return np.random.beta(beta_a, beta_b)
    
    @property
    def mean(self):
        return self.a/(self.a + self.b)
    
    def update(self, x):
        x = np.clip(x, -1, 1)

        if x >= 0:
            self.a += x
            self.b += 1 - x
        else:
            # self.a += x
            self.b += 1 -x

        self.n += 1     
        # self.a = 1 if self.a <1 else self.a
        # self.b = 1 if self.b <1 else self.b


        # self.a += x
        # self.b += 1 - x
        # self.n += 1 

        # self.a = 1 if self.a <1 else self.a
        # self.b = 1 if self.b <1 else self.b


class AgentPlayEnv:
    def __init__(self, prj_name):
        self.prj_name = prj_name

        self.beta_a = 1.
        self.beta_b = 1.
        self.beta_n = 0

        self.reward_list =[]
        # set success threshold, update by set_success
        self.threshold_r = 1
        self.threshold_successvie_count = 20
        self.threshold_success_time = None

    def mark_start_time(self):
        self.start_time = time.time()

    def ep_done_cb(self, ep_reward, ep = None,  all_ep_sum_reward = None):
        # print(f'[{self.prj_name}_{os.getpid()}] ep = {ep}, ep_reward = {ep_reward:.2f},  all_ep_sum_reward = {all_ep_sum_reward:.2f}')
        # self.beta_update(ep_reward)
        successvie_count_max, first_over_threshold_ep = self.check_success(ep_reward)


        # if ep >=MAX_EP:
        #     self.show_result()
        if self.threshold_success_time ==None and first_over_threshold_ep>0:
            self.threshold_success_time = time.time() - self.start_time

        if not self.p_ready.value:
            with self.p_lock:
                self.p_dic['name'] = self.prj_name
                self.p_dic['ep'] = ep
                self.p_dic['r'] = ep_reward
                self.p_dic['sum_r'] = all_ep_sum_reward
                self.p_dic['successvie_count_max'] = successvie_count_max
                self.p_dic['first_over_threshold_ep'] = first_over_threshold_ep
                self.p_dic['from_start_time'] = time.time() - self.start_time
                
                self.p_ready.value = True


        # important, block it
        self.p_conn.recv()          # block


    def set_process_switch(self, conn, ready, dic, lock):
        self.p_conn = conn
        self.p_ready = ready
        self.p_dic = dic
        self.p_lock = lock
    
    def beta_update(self, x):
        self.beta_a += x
        self.beta_b += 1. - x
        self.beta_n += 1

    def set_success(self, i_threshold_r, i_threshold_successvie_count):
        self.threshold_r = i_threshold_r
        self.threshold_successvie_count = i_threshold_successvie_count

    def check_success(self, ep_reward = None):
        successvie_count = 0
        successvie_count_max = 0
        first_over_threshold_ep = -1
        if ep_reward!=None:
            self.reward_list.append(ep_reward)

        # print(f'ep = {ep}, len(self.reward_list)={len(self.reward_list)}')

        for i, r in enumerate(self.reward_list):
            if r >= self.threshold_r:
                successvie_count+=1
            else:
                successvie_count = 0

            if successvie_count > successvie_count_max:
                successvie_count_max = successvie_count
            if successvie_count >= self.threshold_successvie_count and first_over_threshold_ep==-1:
                first_over_threshold_ep = i

            
        if successvie_count_max >= self.threshold_successvie_count:
            return successvie_count_max, first_over_threshold_ep
        else:
            return successvie_count_max, 0

    def show_result(self):
        r_list = self.reward_list
        r_list_half_len = int(len(r_list)*0.5)
        half_avg_r = np.mean( r_list[: r_list_half_len])
        avg_r =  np.mean( r_list)

        successvie_count_max, success_ep = self.check_success()
        print(f"[{self.prj_name}] -> success_ep={success_ep}, successvie_count_max={successvie_count_max}")
        print(f"TS_data_{self.prj_name}=[{success_ep},{self.threshold_success_time} ,{half_avg_r} , {avg_r}]")
        print(f"TS_r_{self.prj_name}={r_list}")


def cmd_get_all_yaml():
    '''get all *.yaml, *.yml files in command line'''
    import argparse

    parser = argparse.ArgumentParser()

    def file_choices(choices,fname):
        ext = splitext(fname)[1][1:]
        if ext not in choices:
            parser.error("file doesn't end with one of {}".format(choices))
        return fname

    parser.add_argument('files', nargs='*',type=lambda s:file_choices(("yaml","yml"),s))
    cmd_parse = parser.parse_args()

    # print('cmd_parse.files =', cmd_parse.files)

    return cmd_parse.files

def duplicate_cmd_yaml_2_p_num(p_num):
    '''
        python xx.py DQN.yaml DDPG.yaml, 
           if p_num=5, return ['DQN.yaml', 'DQN.yaml', 'DQN.yaml', 'DDPG.yaml', 'DDPG.yaml']
           if p_num=4, return ['DQN.yaml', 'DQN.yaml', 'DDPG.yaml', 'DDPG.yaml']
    '''

    cmd_fname_list = cmd_get_all_yaml()
    duplicate = p_num // len(cmd_fname_list)  # // get int, ex: 5//2=2
    # print('duplicate = ', duplicate)

    yaml_fname_list = []
    if p_num % duplicate == 1:
        yaml_fname_list.append(cmd_fname_list[0]) 
    for fname in cmd_fname_list:
        for _ in range(int(duplicate)):
            yaml_fname_list.append(fname)

    return yaml_fname_list

def read_yaml(yaml_name, rand_seed):
    # yaml_name = get_yaml_name_from_arg()
    import config
    print('read yaml ->  ', yaml_name)
    cfg = load_config(yaml_name)
    cfg['misc']['random_seed'] = rand_seed
    cfg['misc']['redirect_stdout_2_file'] = True
    cfg['misc']['gym_monitor_path'] = None
    print('random_seed = ', cfg['misc']['random_seed'])
    
    return cfg

def sec_2_hms(t):
    hms = '%02dh%02dm%02ds' % (t/3600, t/60 % 60  , t%60)
    return hms

def log_one_ready(l, ts):#p_dic_list, any_one_ready):
    # l = p_dic_list[any_one_ready]
    name = l['name']
    ep = l['ep']
    ep_r = l['r']
    successvie_count_max = l['successvie_count_max']
    use_time = sec_2_hms(l['from_start_time'])
    

    print(f"[{name:>20}] a={ts.a:9.1f}, b={ts.b:9.1f}, n={ts.n:8d}, ep={ep:8d}, ep_r={ep_r:4.2f}, suc_max={successvie_count_max:5d}, time={use_time}")

def thompson_sample(process_func, p_num = 100, pool_max = 4, parse_yaml = True):
    np.random.seed(3)
    threads = []
    p_ready_list = []
    p_parent_conn_list = []
    p_child_conn_list = []
    # p_reward_list = []
    p_dict_list = []
    p_lock_list = []
    
    now_play_list = []
    thompsons = [Thompson() for _ in range(p_num) ]

    sample_t = 0

    for i in range(p_num):
        check = Value(c_bool, False)
        parent_conn, child_conn = Pipe()
        manager = Manager()

        # now_reward = Value('f', 0.)
        p_parent_conn_list.append(parent_conn)
        p_child_conn_list.append(child_conn)
        
        p_ready_list.append(check)
        # p_reward_list.append(now_reward)
        p_dict_list.append(manager.dict())
        p_lock_list.append(manager.Lock())

    if parse_yaml:
        yaml_fname_list = duplicate_cmd_yaml_2_p_num(p_num)
    for i in range(p_num):
        if parse_yaml:
            t = Process(target=process_func, args=(i,yaml_fname_list[i], p_child_conn_list[i], p_ready_list[i], p_dict_list[i], p_lock_list[i] ))
        else:
            t = Process(target=process_func, args=(i,p_child_conn_list[i], p_ready_list[i], p_dict_list[i], p_lock_list[i] ))
        t.daemon = True
        threads.append(t)

    for i in range(len(threads)):
        threads[i].start()

    while True:
        # choose 1 to run
        while len(now_play_list) < pool_max: 
            # sample Thompson
            sample_result = [  t.sample()  for t in thompsons]
            for already_run in now_play_list:
                sample_result[already_run] = -999
            play_id = np.argmax(sample_result)

            sample_t+=1
            sample_str = [f'{i:.2f}' for i in sample_result]
            # print(f'({sample_t:>2}) sample_result = {sample_str}, choose={play_id}' )
            
            with p_lock_list[play_id]:
                p_ready_list[play_id].value = False
            p_parent_conn_list[play_id].send('GO')
            
            now_play_list.append(play_id)

   
        any_one_ready = -1
        while True:
            # time.sleep(1)
            for i in now_play_list:
                if threads[i] !=None:
                    if p_ready_list[i].value:
                        # print(f'[main] say -> {i} ready')
                        any_one_ready = i
                        break
                    # else:
                        # print(f'[main] say -> {i} not ready')
            if any_one_ready != -1:
                break
            time.sleep(0.1)

        
        # any_one_reaward = p_reward_list[any_one_ready].value
        any_one_reaward = p_dict_list[any_one_ready]['r']
        thompsons[any_one_ready].update(any_one_reaward)
        now_play_list.remove(any_one_ready)


        log_one_ready( p_dict_list[any_one_ready], thompsons[any_one_ready])

        # if p_dict_list[any_one_ready]['first_over_threshold_ep'] > 0 \
        #         and p_dict_list[any_one_ready]['ep'] >= MAX_EP:  # the config need bigger than this
        #     print('='*10 + 'First Success' +'='*10)
        #     print('[{}] first succes, first_over_ep={}'.format(  \
        #                   any_one_ready  ,p_dict_list[any_one_ready]['first_over_threshold_ep']))
        #     print('='*10 + '=============' +'='*10)

        #     break       # -------break the while loop

    # wait all finish
    while len(now_play_list) > 0:
        any_one_ready = -1
        for i in now_play_list:
            if threads[i] !=None:
                if p_ready_list[i].value:
                    any_one_ready = i
                    break
                # else:
                    # print(f'[main] say -> {i} not ready')
        if any_one_ready != -1:
            now_play_list.remove(any_one_ready)
            any_one_ready = -1
        print('Wait now_play_list = ', now_play_list)
        time.sleep(1.0)

    # all_mean = [  t.mean  for t in thompsons]
    # print(all_mean)
    for i, t in enumerate(thompsons):
        print('[{:03d}] mean={:5.2f}, n={:5d}, a={:8.2f}, b={:8.2f}'.format(i, t.mean, t.n, t.a, t.b))

    # print(p_dict_list)
    for i, p_dic in enumerate( p_dict_list):
        # print(p_dic)
        if 'from_start_time' in p_dic:
            print('[{:02d}-{:>20}] ep={:5d}, sum_r={:8.2f}, successvie_count_max={:3d}, first_success_ep={:3d}, use_time={:.2f}'.format(
                i, p_dic['name'], 
                p_dic['ep'], p_dic['sum_r'] , 
                p_dic['successvie_count_max'], p_dic['first_over_threshold_ep'], p_dic['from_start_time'] ))


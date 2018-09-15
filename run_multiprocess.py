from multiprocessing import Process, Value, Lock, Manager, Pipe
import sys, os
import time
from ctypes import c_bool
from os.path import basename, splitext
from config import load_config
import  numpy as np
import random
'''
# def process_func(p_index, yaml_path, check, step, lock):
def process_func(p_index, yaml_path):
    # start = time.time()
    # print('process id:', os.getpid(), 'start')
    # sum=0
    # for i in range(int(1e17)):
    #     if check.value:
    #         with lock:
    #             step.value = i
    #             check.value = False
    #     sum+=i
    # print('process id:', os.getpid(), 'finish, sum = ', sum,', use time= ', time.time() - start)
    from flappybird import FlappyBird
    
    yaml_name = splitext(yaml_path)[0]
    cfg = load_config(yaml_path)

    print('random_seed = ', cfg['misc']['random_seed'])
    print('yaml_name = ', yaml_name)

    # m_cfg = maze_cfg(cfg)
    c = FlappyBird(cfg, project_name='flappy-' + yaml_name + '-' + str(p_index), redirect_stdout=True )
    #c = MazeStandalone(cfg, project_name='maze-' + yaml_name + '-' + str(p_index), redirect_stdout=True )
    c.run()
'''

def check_cpu_info():
    import platform, os
 
    print('platform.system()  = ', platform.system() )
    if platform.system() == 'Windows':
        return platform.processor()
    elif platform.system() == 'Darwin':
        command = '/usr/sbin/sysctl -n machdep.cpu.brand_string'
        return os.popen(command).read().strip()
    elif platform.system() == 'Linux':
        command = 'cat /proc/cpuinfo'
        return os.popen(command).read().strip()
    return 'platform not identified'

# print(check_cpu_info())

def cmd_get_all_yaml():
    '''get all *.yaml, *.yml files'''
    import argparse

    parser = argparse.ArgumentParser()

    def file_choices(choices,fname):
        ext = splitext(fname)[1][1:]
        if ext not in choices:
            parser.error("file doesn't end with one of {}".format(choices))
        return fnametime.sleep(1)

    parser.add_argument('files', nargs='*',type=lambda s:file_choices(("yaml","yml"),s))
    cmd_parse = parser.parse_args()

    # print('cmd_parse.files =', cmd_parse.files)

    return cmd_parse.files


def basic_multiprocess_from_arg(process_func):
    yaml_list = cmd_get_all_yaml()
    p_num = len(yaml_list)
    p_list = []
    for i in range(p_num):
        check = Value(c_bool, False)
        now_step = Value('i', 0)  #i is integer

    for i in range(p_num):
        t = Process(target=process_func, args=(i, yaml_list[i] ))
        t.daemon = True
        p_list.append(t)

    #start 
    for i in range(len(p_list)):
        p_list[i].start()

    # wait all finish
    for j in range(len(p_list)):
        p_list[j].join()

    

def choose_best_from_arg(process_func):
    yaml_list = cmd_get_all_yaml()
    p_num = len(yaml_list)
    p_list = []
    p_check_list = []
    p_step_list = []
    p_lock_list = []
    for i in range(p_num):
        check = Value(c_bool, False)
        now_step = Value('i', 0)
        p_check_list.append(check)
        p_step_list.append(now_step)
        p_lock_list.append(Lock())

    for i in range(p_num):
        t = Process(target=process_func, args=(i, yaml_list[i], p_check_list[i],  p_step_list[i], p_lock_list[i] ))
        t.daemon = True
        p_list.append(t)

    #start 
    for i in range(len(p_list)):
        p_list[i].start()


    while True:
        time.sleep(5)

        print('---------set all check to True-----------')
        for i in range(len(p_check_list)):
            if p_list[i] !=None:
                with p_lock_list[i]:
                    p_check_list[i].value = True
        
        start = time.time()
   
        # check all is ready to get data
        while True:
            time.sleep(0.1)
            one_not_ready = False
            for i in range(len(p_check_list)):
                if p_list[i] !=None and p_check_list[i].value:
                    print(f'{i} not ready')
                    one_not_ready = True
            if not one_not_ready:
                break 
        
        print('check use time= ', time.time() - start)
        # get min reward
        last_i = -1
        now_min = 9999999999999999999
        for i in range(len(p_step_list)):
            if  p_list[i] !=None: 
                if p_step_list[i].value < now_min:
                    last_i = i
                    now_min = p_step_list[i].value
                
                print(f'{i} step: {p_step_list[i].value}')

        # terminate min reward agent
        print('p_list.count(None) = ', p_list.count(None))
        if (p_num - p_list.count(None) )> 1:
            print('KILL last_i=', last_i)
            p_list[last_i].terminate()
            p_list[last_i] = None


    # check again
    for j in range(len(p_list)):
        p_list[j].join()

    print("------------------------")
    print('main process pid:', os.getpid())
    

def check_p_reward_from_arg(process_func):
    yaml_list = cmd_get_all_yaml()
    p_num = len(yaml_list)
    p_list = []
    p_check_list = []
    p_dict_list = []
    p_lock_list = []
    for i in range(p_num):
        check = Value(c_bool, False)
        manager = Manager()
        p_check_list.append(check)
        p_dict_list.append(manager.dict())
        p_lock_list.append(Lock())

    for i in range(p_num):
        t = Process(target=process_func, args=(i, yaml_list[i], p_check_list[i],  p_dict_list[i], p_lock_list[i] ))
        t.daemon = True
        p_list.append(t)

    #start 
    for i in range(len(p_list)):
        p_list[i].start()


    while True:
        time.sleep(5)

        # print('---------set all check to True-----------')
        for i in range(len(p_check_list)):
            if p_list[i] !=None:
                with p_lock_list[i]:
                    p_check_list[i].value = True
        
        start = time.time()
   
        # check all is ready to get data
        while True:
            time.sleep(0.1)
            one_not_ready = False
            for i in range(len(p_check_list)):
                if p_list[i] !=None and p_check_list[i].value:
                    # print(f'{i} not ready')
                    one_not_ready = True
            if not one_not_ready:
                break 
        
        # print('check use time= ', time.time() - start)
        # get min reward
        result_str = ''
        for i in range(p_num):
            result_str += f'[{i}]={p_step_list[i].value}\t' 
            
        print(f'reward -> {result_str}')

    # check again
    for j in range(len(p_list)):
        p_list[j].join()

    print("------------------------")
    print('main process pid:', os.getpid())
    


def pass_dict_from_arg(process_func):
    yaml_list = cmd_get_all_yaml()
    p_num = len(yaml_list)
    p_list = []
    p_check_list = []
    p_dict_list = []
    p_lock_list = []
    for i in range(p_num):
        check = Value(c_bool, False)
        manager = Manager()

        p_check_list.append(check)
        p_dict_list.append(manager.dict())
        p_lock_list.append(manager.Lock())

    for i in range(p_num):
        t = Process(target=process_func, args=(i, yaml_list[i], p_check_list[i],  p_dict_list[i], p_lock_list[i] ))
        t.daemon = True
        p_list.append(t)

    #start 
    for i in range(len(p_list)):
        p_list[i].start()


    while True:
        time.sleep(5)

        # print('---------set all check to True-----------')
        for i in range(len(p_check_list)):
            if p_list[i] !=None:
                with p_lock_list[i]:
                    p_check_list[i].value = True
        
        start = time.time()
   
        # check all is ready to get data
        while True:
            time.sleep(0.1)
            one_not_ready = False
            for i in range(len(p_check_list)):
                if p_list[i] !=None and p_check_list[i].value:
                    # print(f'{i} not ready')
                    one_not_ready = True
            if not one_not_ready:
                break 
        
        # print('check use time= ', time.time() - start)
        # get min reward
        
        for i in range(p_num):
            dic_str ='[{}] '.format(p_dict_list[i]['name'])
            for key, val in p_dict_list[i].items():
                if key=='name': 
                    continue
                if type(val)==float:
                    dic_str+= f'{key}={val:.2f} '
                else:
                    dic_str+= f'{key}={val} '
            
            print(dic_str)
        
        # for i in range(p_num):
        #     dic = p_dict_list[i]
        #     print('[{}] ep = {}, r = {}, sum_r = {}'.format(dic['name'], dic['ep'], dic['r'], dic['sum_r']))

            
            # result_str += f'[{i}]={p_step_list[i].value}\t' 
            
        # print(f'reward -> {result_str}')

    # check again
    for j in range(len(p_list)):
        p_list[j].join()

    print("------------------------")
    print('main process pid:', os.getpid())



class Thompson(object):
    def __init__(self):
        self.a = 1
        self.b = 1
        self.n = 0
        
    def sample(self):
        if self.a <=0 or self.b<=0:
            return random.random()
        else:
            return np.random.beta(self.a, self.b)
    
    @property
    def mean(self):
        return self.a/(self.a + self.b)
    
    def update(self, x):
        self.a += x
        self.b += 1 - x
        self.n += 1 


def thompson_sample(process_func, p_num = 100, pool_max = 4):
    threads = []
    p_ready_list = []
    p_parent_conn_list = []
    p_child_conn_list = []
    # p_reward_list = []
    p_dict_list = []
    p_lock_list = []
    
    now_play_list = []
    thompsons = [Thompson() for _ in range(p_num) ]

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

    for i in range(p_num):
        t = Process(target=process_func, args=(i, p_child_conn_list[i], p_ready_list[i], p_dict_list[i], p_lock_list[i] ))
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

        if p_dict_list[any_one_ready]['first_over_threshold_ep'] > 0:
            print('='*10 + 'First Success' +'='*10)
            print('[{}] first succes, first_over_ep={}'.format(  \
                          any_one_ready  ,p_dict_list[any_one_ready]['first_over_threshold_ep']))
            print('='*10 + '=============' +'='*10)

            break       # -------break the while loop

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
        print('[{}] mean={:2f}, n={}, a={}, b={}'.format(i, t.mean, t.n, t.a, t.b))

    # print(p_dict_list)
    for i, p_dic in enumerate( p_dict_list):
        # print(p_dic)
        if 'from_start_time' in p_dic:
            print('[{:02d}-{}] ep={}, sum_r={}, successvie_count_max={}, first_success_ep={}, use_time={:.2f}'.format(
                i, p_dic['name'], 
                p_dic['ep'], p_dic['sum_r'] , 
                p_dic['successvie_count_max'], p_dic['first_over_threshold_ep'], p_dic['from_start_time'] ))

    # for p in p_parent_conn_list:
    #     p.send('print_wait_time')

    # time.sleep(10)
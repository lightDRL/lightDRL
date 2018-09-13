from multiprocessing import Process, Value, Lock
import sys, os
import time
from ctypes import c_bool
from os.path import basename, splitext
from config import load_config

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

def cmd_get_all_yaml():
    '''get all *.yaml, *.yml files'''
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
    
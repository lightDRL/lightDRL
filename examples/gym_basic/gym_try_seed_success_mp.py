# Run gym with different seed
#   ex: 
#       python gym_try_seed.py DDPG_CartPole-v0.yaml
#       python gym_try_seed_success_multiprocess_pool.py DQN_CartPole-v0.yamlce: CUDA_ERROR_OUT_OF_MEMORY
# Author:  kbehouse  <https://github.com/kbehouse/>
#          

import sys, os
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)),'../../'))
from gym_basic_standalone import GymStandalone, gym_cfg
from standalone import standalone_switch
import gym
import time
from config import set_gym_monitor_path, load_config
import numpy as np
import multiprocessing as mp
from itertools import product

def gym_thread(seed, yaml_path):
    print('thread get path = ', yaml_path)
    cfg_copy = gym_cfg( load_config(yaml_path)  )
    print('cfg_copy', cfg_copy)
    
    cfg_copy['misc']['random_seed'] = seed
    cfg_copy['misc']['render'] = False
    # cfg_copy['misc']['gpu_memory_ratio'] = 0.03     # all share same ratio
    cfg_copy['misc']['max_ep'] = 1000
    cfg_copy['misc']['worker_nickname'] = 'w_%03d' % (seed)
    cfg_copy['misc']['redirect_stdout_2_file'] = False
    cfg_copy['misc']['gym_monitor_path'] = None
    
    prj_name ='gym-%s_seed_%04d' % ( cfg_copy['RL']['method'], seed)
    
    s = standalone_switch(GymStandalone, cfg_copy, prj_name)
    s.run()

    # ep = s.ep
    # return ep
    use_time = time.time()-s.start_time 
    avg_reward = s.all_ep_reward / s.ep
    ret_dic = {'ep': s.ep-1, 'use_time': use_time, 'all_ep_reward': s.all_ep_reward,'avg_reward': avg_reward, 'is_success': s.is_success}

    return ret_dic

if __name__ == '__main__':
    yaml_path = sys.argv[1]
    print('Main thread yaml_paht = ', yaml_path)
    all_t =[]

    s_time = time.time()

    start_seed = 0
    end_seed = 0

    find_success = False
    while not find_success:
        start_seed = end_seed
        end_seed+= 10

        rand_seed_list = range(start_seed,end_seed)
        seed_num  = len(rand_seed_list)
        yaml_path_list = [yaml_path  for _ in range(seed_num)]
        print('yaml_path_list = ' , (yaml_path_list))
        with mp.Pool(processes=seed_num) as pool:
            results = pool.starmap(gym_thread, product(rand_seed_list, [yaml_path]))
        # res = mp.Pool().map(gym_thread, rand_seed_list, yaml_path)

        # print(f'({start_seed} -> {end_seed}) : results = ' , results) 

        # for res in results:
        #     if res < 1000:
        #         find_success = True
        #         break
        for dic in results:
            if dic['is_success']:
                find_success = True
                break
    


    max_seed_ep = -999999
    min_seed_ep  =  999999
    max_rand = None
    min_rand = None
    # for i, ep  in enumerate(results):
    #     if ep  > max_seed_ep :
    #         max_rand = rand_seed_list[i]
    #         max_seed_ep  = ep 

    #     if ep  < min_seed_ep :
    #         min_rand = rand_seed_list[i]
    #         min_seed_ep  = ep 

    for i, dic  in enumerate(results):
        if dic['ep']  > max_seed_ep :
            max_rand = rand_seed_list[i]
            max_seed_ep  = dic['ep'] 

        if dic['ep']  < min_seed_ep :
            min_rand = rand_seed_list[i]
            min_seed_ep  = dic['ep'] 



    for i, d  in enumerate(results):
        print(f"[seed={rand_seed_list[i]}], is_success={d['is_success']}, ep={d['ep']:5d}, all_ep_reward={d['all_ep_reward']:8.2f}, avg_reward={d['avg_reward']:6.2f}, use_time={d['use_time']:6.2f}")

    print('Use time = ', time.time() - s_time)
    print('%s get min ep  = %6.2f' % (min_rand, min_seed_ep ))   
    print('%s get max ep  = %6.2f' % (max_rand, max_seed_ep ))
    
# Run gym with different seed
#   ex: 
#       python gym_try_seed.py DDPG_CartPole-v0.yaml
#       python gym_try_seed_success_multiprocess_pool.py DQN_CartPole-v0.yamlce: CUDA_ERROR_OUT_OF_MEMORY
# Author:  kbehouse  <https://github.com/kbehouse/>
#          

import sys, os
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)),'../../'))
from gym_basic_standalone import GymStandalone, gym_cfg
import gym
import time
from config import set_gym_monitor_path, load_config
import numpy as np
import multiprocessing as mp
from itertools import product

def gym_thread(seed, yaml_path):
    # print('thread get path = ', yaml_path)
    cfg_copy = gym_cfg( load_config(yaml_path)  )
    # print('cfg_copy of seed=', seed)
    
    cfg_copy['misc']['random_seed'] = seed
    cfg_copy['misc']['render'] = False
    # cfg_copy['misc']['gpu_memory_ratio'] = 0.03     # all share same ratio
    cfg_copy['misc']['max_ep'] = 5
    cfg_copy['misc']['worker_nickname'] = 'w_%03d' % (seed)
    prj_name ='gym-%s_seed_%04d' % ( cfg_copy['RL']['method'], seed)
    
    cfg_copy['misc']['gym_monitor_path'] = None

    c = GymStandalone(cfg_copy , project_name=prj_name)
    c.run()

    all_ep_reward = c.all_ep_reward
    print('all_ep_reward = ', all_ep_reward)
    return all_ep_reward

if __name__ == '__main__':
    yaml_path = sys.argv[1]
    print('Main thread yaml_paht = ', yaml_path)
    all_t =[]

    
    rand_seed_list = range(0,30)
    seed_num  = len(rand_seed_list)
    yaml_path_list = [yaml_path  for _ in range(seed_num)]
    print('yaml_path_list = ' , (yaml_path_list))
    with mp.Pool(processes=16) as pool:
        results = pool.starmap(gym_thread, product(rand_seed_list, [yaml_path]))
    # res = mp.Pool().map(gym_thread, rand_seed_list, yaml_path)

    print('results = ' , results)  


    max_seed_r = -999999
    min_seed_r =  999999
    max_rand = None
    min_rand = None
    for i, r in enumerate(results):
        if r > max_seed_r:
            max_rand = rand_seed_list[i]
            max_seed_r = r

        if r < min_seed_r:
            min_rand = rand_seed_list[i]
            min_seed_r = r

    print('%s get min reward = %6.2f' % (min_rand, min_seed_r))   
    print('%s get max reward = %6.2f' % (max_rand, max_seed_r))
    
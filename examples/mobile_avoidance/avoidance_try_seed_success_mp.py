# Run gym with different seed
#   ex: 
#       python avoidance_try_seed_success_mp.py DDPG.yaml
# Author:  kbehouse  <https://github.com/kbehouse/>
#          

import sys, os
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)),'../../'))
from avoidance_standalone import AvoidanceStandalone
from standalone import standalone_switch
import gym
import time
from config import set_gym_monitor_path, load_config
import numpy as np
import multiprocessing as mp
from itertools import product

def try_mp(seed, yaml_path):
    print('thread get path = ', yaml_path)
    cfg_copy =  load_config(yaml_path) 
    # print('cfg_copy', cfg_copy)
    
    cfg_copy['misc']['random_seed'] = seed
    cfg_copy['misc']['render'] = False
    # cfg_copy['misc']['gpu_memory_ratio'] = 0.03     # all share same ratio
    cfg_copy['misc']['worker_nickname'] = 'w_%03d' % (seed)
    cfg_copy['misc']['redirect_stdout_2_file'] = False
    cfg_copy['misc']['gym_monitor_path'] = None
    
    prj_name ='avoidance-%s_seed_%04d' % ( cfg_copy['RL']['method'], seed)
    
    s = standalone_switch(AvoidanceStandalone, cfg_copy, prj_name)
    s.run()

    r_list = s.reward_list
    r_list_half_len = int(len(r_list)*0.5)
    half_avg_r = np.mean( r_list[: r_list_half_len])
    avg_r =  np.mean( r_list)

    successvie_count_max, first_over_threshold_ep = s.check_success()
    # ep = s.ep
    # return ep
    # use_time = time.time()-s.start_time 
    # avg_reward = s.all_ep_reward / s.ep
    # ret_dic = {'ep': s.ep-1, 'use_time': use_time, 'all_ep_reward': s.all_ep_reward,'avg_reward': avg_reward, 'is_success': s.is_success}
    ret_dic = {'success_ep': first_over_threshold_ep, 'success_time': s.threshold_success_time,
                'successvie_count_max': successvie_count_max, 
                'half_avg_r': half_avg_r,'avg_r': avg_r, 'r_list':r_list}
    

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
        end_seed+= 4

        rand_seed_list = range(start_seed,end_seed)
        seed_num  = len(rand_seed_list)
        yaml_path_list = [yaml_path  for _ in range(seed_num)]
        print('yaml_path_list = ' , (yaml_path_list))
        with mp.Pool(processes=seed_num) as pool:
            results = pool.starmap(try_mp, product(rand_seed_list, [yaml_path]))
        # res = mp.Pool().map(gym_thread, rand_seed_list, yaml_path)

        # print(f'({start_seed} -> {end_seed}) : results = ' , results) 

        # for res in results:
        #     if res < 1000:
        #         find_success = True
        #         break
        for dic in results:
            if dic['success_ep'] > 0:
                find_success = True
                break
    


    max_seed_ep = -999999
    min_seed_ep  =  999999
    max_rand = None
    min_rand = None

    for i, dic  in enumerate(results):
        if dic['success_ep']  > max_seed_ep :
            max_rand = rand_seed_list[i]
            max_seed_ep  = dic['success_ep'] 

        if dic['success_ep']  < min_seed_ep :
            min_rand = rand_seed_list[i]
            min_seed_ep  = dic['success_ep'] 

    for i, d  in enumerate(results):
        print(f"{rand_seed_list[i]}_seed_want=[{d['success_ep']},{d['success_time']} ,{d['half_avg_r']} , {d['avg_r']}]")
        print(f"{rand_seed_list[i]}_seed_list={d['r_list']}")

    for i, d  in enumerate(results):
        show_s = f'[seed={rand_seed_list[i]}] '
        for key,val in  d.items():
            if type(val)==float:
                show_s += f'{key} = {val:.2f},'
            elif type(val)==int:
                show_s += f'{key} = {val:3d},'
            else:
                show_s += f'{key} = {val},'

        print(show_s)
        
    print('Use time = ', time.time() - s_time)
    print('%s get min ep  = %6.2f' % (min_rand, min_seed_ep ))   
    print('%s get max ep  = %6.2f' % (max_rand, max_seed_ep ))
    
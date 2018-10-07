# python gym_TS_with_random_seed.py DQN.yaml
# python gym_TS_with_random_seed.py DQN_CartPole-v0.yaml DDPG_CartPole-v0.yaml | tee  log/test-p8-r3.log
# python gym_TS_with_random_seed.py DQN.yaml DDPG.yaml Q-learning.yaml | tee  test-p2-r3.log
# python gym_TS_with_random_seed.py DQN.yaml DDPG.yaml Q-learning.yaml A3C.yaml | tee  test-p16-r3.log
# python gym_TS_with_random_seed.py DQN_CartPole-v0.yaml DDPG_CartPole-v0.yaml | teelog/p16-slot4.lo

# Choose best agent Use Thompson Sampling (TS) from radom seed (0~14)
from os.path import splitext
import sys, os
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)),'../../'))
from gym_basic_standalone import GymStandalone, gym_cfg
from TS_run_multiprocess import thompson_sample, AgentPlayEnv, read_yaml
from config import set_gym_monitor_path, load_config

import time

# def process_func(p_index, yaml_path):
def process_func(p_index, yaml_name, conn, ready, dic, lock):
    conn.recv()

    cfg = gym_cfg( read_yaml(yaml_name, p_index)  )

    # init AgentPlayEnv
    prj_name = 'gym-{}-{:03d}'.format( yaml_name ,p_index) 
    agent = AgentPlayEnv(prj_name)
    agent.set_process_switch(conn, ready, dic, lock)
    agent.set_success(cfg['misc']['threshold_r'], cfg['misc']['threshold_successvie_count'])
    
    c = GymStandalone(cfg, project_name=prj_name)
    c.set_ep_done_cb(agent.ep_done_cb)

    print(f'[{prj_name}] start run with {yaml_name}')
    agent.mark_start_time()
    c.run()


if __name__ == '__main__':
    thompson_sample(process_func, p_num =12, pool_max = 4)

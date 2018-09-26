# python maze_TS_with_random_seed.py DQN.yaml
# python maze_TS_with_random_seed.py DQN.yaml DDPG.yaml | tee  test-p2-r3.log
# python maze_TS_with_random_seed.py DQN.yaml DDPG.yaml Q-learning.yaml | tee  test-p2-r3.log
# python maze_TS_with_random_seed.py DQN.yaml DDPG.yaml Q-learning.yaml A3C.yaml | tee  test-p16-r3.log

# Choose best agent Use Thompson Sampling (TS) from radom seed (0~14)
from os.path import splitext
import sys, os
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)),'../../'))
from TS_run_multiprocess import thompson_sample, AgentPlayEnv, read_yaml
from config import load_config_from_arg, get_yaml_name_from_arg
import time

# def process_func(p_index, yaml_path):
def process_func(p_index, yaml_name, conn, ready, dic, lock):
    conn.recv()

    cfg = read_yaml(yaml_name, p_index)
    # init AgentPlayEnv
    prj_name = 'avoidance-TS-{}-{:03d}'.format( yaml_name ,p_index) 
    agent = AgentPlayEnv(prj_name)
    agent.set_process_switch(conn, ready, dic, lock)
    agent.set_success(50, 20)
    
    from avoidance_standalone import AvoidanceStandalone
    c = AvoidanceStandalone(cfg, project_name=prj_name)
    c.set_ep_done_cb(agent.ep_done_cb)

    print(f'[{prj_name}] start run with {yaml_name}')
    agent.mark_start_time()
    c.run()


if __name__ == '__main__':
    thompson_sample(process_func, p_num =12, pool_max = 4)

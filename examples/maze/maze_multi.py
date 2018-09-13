# python maze_multi.py DQN.yaml DQN_redirect_stdout_2_file.yaml
from os.path import splitext
import sys, os
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)),'../../'))
from run_multiprocess import basic_multiprocess_from_arg, choose_best_from_arg
from config import load_config

class AgentPlayEnv:
    def __init__(self, prj_name):
        self.prj_name = prj_name

    def ep_done_cb(self, ep, ep_reward, all_ep_sum_reward):
        # print(f'[{self.prj_name}_{os.getpid()}] ep = {ep}, ep_reward = {ep_reward},  all_ep_sum_reward = {all_ep_sum_reward}')

        if self.p_check.value:
            with self.p_lock:
                self.p_step.value = all_ep_sum_reward
                self.p_check.value = False


    def set_process_switch(self, check, step, lock):
        self.p_check = check
        self.p_step = step
        self.p_lock = lock



# def process_func(p_index, yaml_path):
def process_func(p_index, yaml_path, check, step, lock):
    yaml_name = splitext(yaml_path)[0]
    cfg = load_config(yaml_path)

    print('random_seed = ', cfg['misc']['random_seed'])
    print('yaml_name = ', yaml_name)

    # init AgentPlayEnv
    prj_name = 'maze-' + yaml_name + '-' + str(p_index)
    agent = AgentPlayEnv(prj_name)
    agent.set_process_switch(check, step, lock)
    
    
    from maze_standalone import MazeStandalone
    c = MazeStandalone(cfg, project_name=prj_name)
    c.set_ep_done_cb(agent.ep_done_cb)
    c.run()


if __name__ == '__main__':
    choose_best_from_arg(process_func)


'''
def process_func(p_index, yaml_path):
    
    yaml_name = splitext(yaml_path)[0]
    cfg = load_config(yaml_path)

    print('random_seed = ', cfg['misc']['random_seed'])
    print('yaml_name = ', yaml_name)

    from maze_standalone import MazeStandalone
    c = MazeStandalone(cfg, project_name='flappy-' + yaml_name + '-' + str(p_index))
    c.run()


if __name__ == '__main__':
    basic_multiprocess_from_arg(process_func)
'''
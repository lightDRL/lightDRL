# python maze_multi_dict.py DQN.yaml DQN_redirect_stdout_2_file.yaml
from os.path import splitext
import sys, os
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)),'../../'))
from run_multiprocess import pass_dict_from_arg
from config import load_config

class AgentPlayEnv:
    def __init__(self, prj_name):
        self.prj_name = prj_name

        self.beta_a = 1.
        self.beta_b = 1.
        self.beta_n = 0

    def ep_done_cb(self, ep, ep_reward, all_ep_sum_reward):
        # print(f'[{self.prj_name}_{os.getpid()}] ep = {ep}, ep_reward = {ep_reward},  all_ep_sum_reward = {all_ep_sum_reward}')
        self.beta_update(ep_reward)

        if self.p_check.value:
            with self.p_lock:
                self.p_dic['name'] = self.prj_name
                self.p_dic['ep'] = ep
                self.p_dic['r'] = ep_reward
                self.p_dic['sum_r'] = all_ep_sum_reward
                
                self.p_dic['beta_a'] = self.beta_a
                self.p_dic['beta_b'] = self.beta_b
                self.p_dic['beta_avg'] = self.beta_a / ( self.beta_a + self.beta_b)
                

                self.p_check.value = False


    def set_process_switch(self, check, dic, lock):
        self.p_check = check
        self.p_dic = dic
        self.p_lock = lock
    
    def beta_update(self, x):
        self.beta_a += x
        self.beta_b += 1. - x
        self.beta_n += 1


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
    #choose_best_from_arg(process_func)
    pass_dict_from_arg(process_func)

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
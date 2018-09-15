# python maze_TS_with_random_seed.py DQN.yaml
# Choose best agent Use Thompson Sampling (TS) from radom seed (0~14)
from os.path import splitext
import sys, os
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)),'../../'))
from run_multiprocess import pass_dict_from_arg, thompson_sample
from config import load_config_from_arg, get_yaml_name_from_arg
import time
class AgentPlayEnv:
    def __init__(self, prj_name):
        self.prj_name = prj_name

        self.beta_a = 1.
        self.beta_b = 1.
        self.beta_n = 0

        self.reward_list =[]

    def mark_start_time(self):
        self.start_time = time.time()

    def ep_done_cb(self, ep, ep_reward, all_ep_sum_reward):
        print(f'[{self.prj_name}_{os.getpid()}] ep = {ep}, ep_reward = {ep_reward},  all_ep_sum_reward = {all_ep_sum_reward}')
        self.beta_update(ep_reward)
        successvie_count_max, first_over_threshold_ep = self.check_success(ep, ep_reward)

        if not self.p_ready.value:
            with self.p_lock:
                self.p_dic['name'] = self.prj_name
                self.p_dic['ep'] = ep
                self.p_dic['r'] = ep_reward
                self.p_dic['sum_r'] = all_ep_sum_reward
                
                self.p_dic['successvie_count_max'] = successvie_count_max
                self.p_dic['first_over_threshold_ep'] = first_over_threshold_ep
                self.p_dic['from_start_time'] = time.time() - self.start_time
                
                # self.p_dic['beta_a'] = self.beta_a
                # self.p_dic['beta_b'] = self.beta_b
                # self.p_dic['beta_avg'] = self.beta_a / ( self.beta_a + self.beta_b)
                

                self.p_ready.value = True

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

    def check_success(self, ep, ep_reward):
        threshold_r = 1
        threshold_successvie_count = 20

        successvie_count = 0
        successvie_count_max = 0
        first_over_threshold_ep = -1
        self.reward_list.append(ep_reward)

        print(f'ep = {ep}, len(self.reward_list)={len(self.reward_list)}')

        for i, r in enumerate(self.reward_list):
            if r >= threshold_r:
                successvie_count+=1
            else:
                successvie_count = 0

            if successvie_count > successvie_count_max:
                successvie_count_max = successvie_count
            if successvie_count >= threshold_successvie_count and first_over_threshold_ep==-1:
                first_over_threshold_ep = i

            
        if successvie_count_max >= threshold_successvie_count:
            return successvie_count_max, first_over_threshold_ep
        else:
            return successvie_count_max, 0



# def process_func(p_index, yaml_path):
def process_func(p_index, conn, ready, dic, lock):
    conn.recv()

    yaml_name = get_yaml_name_from_arg()
    cfg = load_config_from_arg()
    cfg['misc']['random_seed'] = p_index

    print('random_seed = ', cfg['misc']['random_seed'])
    print('yaml_name = ', yaml_name)

    # init AgentPlayEnv
    prj_name = 'maze-' + yaml_name + '-' + str(p_index)
    agent = AgentPlayEnv(prj_name)
    agent.set_process_switch(conn, ready, dic, lock)
    
    
    from maze_standalone import MazeStandalone
    c = MazeStandalone(cfg, project_name=prj_name)
    c.set_ep_done_cb(agent.ep_done_cb)

    agent.mark_start_time()
    c.run()


if __name__ == '__main__':
    thompson_sample(process_func, p_num =15, pool_max = 4)

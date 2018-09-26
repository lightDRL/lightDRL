import yaml
import sys

# DATA_POOL = 'data_pool/'

# cfg = []

""" For loader tuple"""
class YAMLPatch(yaml.SafeLoader):
    def construct_python_tuple(self, node):
        return tuple(self.construct_sequence(node))

YAMLPatch.add_constructor(u'tag:yaml.org,2002:python/tuple', YAMLPatch.construct_python_tuple)


def set_none_if_not_exist(cfg):
    cfg['RL']['train_multi_steps'] = cfg['RL']['train_multi_steps'] if 'train_multi_steps' in cfg['RL'] else 1
    cfg['RL']['add_data_steps'] = cfg['RL']['add_data_steps'] if 'add_data_steps' in cfg['RL'] else 1
    cfg['RL']['reward_reverse'] = cfg['RL']['reward_reverse'] if 'reward_reverse' in cfg['RL'] else False
    cfg['RL']['exploration'] = cfg['RL']['exploration'] if 'exploration' in cfg['RL'] else 0
    
    cfg['RL']['action_epsilon'] = cfg['RL']['action_epsilon'] if 'action_epsilon' in cfg['RL'] else None
    cfg['RL']['action_epsilon_add'] = cfg['RL']['action_epsilon_add'] if 'action_epsilon_add' in cfg['RL'] else None
    cfg['RL']['method'] = 'Qlearning'   if cfg['RL']['method']=='Q-learning'  else cfg['RL']['method']    # for python variable rule
    cfg['RL']['reward_factor'] =  1.0 if not 'reward_factor' in cfg['RL'] else cfg['RL']['reward_factor']
    cfg['RL']['action_noise'] = None if  not 'action_noise' in cfg['RL'] else cfg['RL']['action_noise']

    cfg['misc']['random_seed'] = None  if 'random_seed' not in cfg['misc'] else cfg['misc']['random_seed']
    cfg['misc']['render'] = False if  not 'render' in cfg['misc'] else cfg['misc']['render']
    cfg['misc']['render_after_ep'] = 0 if  not 'render_after_ep' in cfg['misc'] else cfg['misc']['render_after_ep']
    cfg['misc']['max_ep'] = 1000  if  not 'max_ep' in cfg['misc'] else cfg['misc']['max_ep']
    # py2: sys.maxint=9223372036854775807, py3: sys.maxsize 9223372036854775807
    cfg['misc']['ep_max_step'] = 922337203685477580  if  not 'ep_max_step' in cfg['misc'] else cfg['misc']['ep_max_step']
    cfg['misc']['worker_nickname'] ='worker'  if  not 'worker_nickname' in cfg['misc'] else cfg['misc']['worker_nickname']
    cfg['misc']['gym_monitor_path'] = None  if 'gym_monitor_path' not in cfg['misc'] else cfg['misc']['gym_monitor_path']
    cfg['misc']['gym_monitor_episode'] = 1  if 'gym_monitor_episode' not in cfg['misc'] else cfg['misc']['gym_monitor_episode']
    cfg['misc']['gym_monitor_path_origin'] =  cfg['misc']['gym_monitor_path']
    cfg['misc']['gym_monitor_path'] = set_gym_monitor_path(cfg['misc']['gym_monitor_path'])

    cfg['misc']['model_retrain'] = False  if 'model_retrain' not in cfg['misc'] else cfg['misc']['model_retrain']
    cfg['misc']['gpu_memory_ratio'] = None  if 'gpu_memory_ratio' not in cfg['misc'] else cfg['misc']['gpu_memory_ratio']
    cfg['misc']['redirect_stdout_2_file'] = False  if 'redirect_stdout_2_file' not in cfg['misc'] else cfg['misc']['redirect_stdout_2_file']
    cfg['misc']['threshold_successvie_break'] = True if 'threshold_successvie_break' not in cfg['misc'] else cfg['misc']['threshold_successvie_break'] 

def set_gym_monitor_path(gym_monitor_path, i_project_name = None):
    if gym_monitor_path==None:
        return None
    else:
        import os
        # from server.py
        DATA_POOL = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data_pool')
        # from gym_basic.py
        project_name='gym-' + get_yaml_name_from_arg() if i_project_name==None else i_project_name
        monitor_path = os.path.join(DATA_POOL, project_name, gym_monitor_path)
        # print('monitor_path= ', monitor_path)
        # print('DATA_POOL=%s, project_name=%s')
        return monitor_path

def get_yaml_name_from_arg():
    from os.path import basename, splitext
    f_name = basename(sys.argv[1])
    f_name = splitext(f_name)[0]
    # print('f_name = ' , f_name)
    return f_name


def load_config(f_name = "config/default.yaml"):
    cfg=[]
    print("[I] Load {}".format(f_name))
    """ Load File"""
    with open(f_name, 'r') as stream:
        try:
            cfg = yaml.load(stream, Loader=YAMLPatch)
            # print(yaml.dump(cfg))
        except yaml.YAMLError as exc:
            print('in load_config: yaml.YAMLError -> '+exc)
    set_none_if_not_exist(cfg)
    return cfg

def load_config_from_arg():
    if len(sys.argv) >= 2 and (sys.argv[1].endswith('.yaml') or sys.argv[1].endswith('.yml')) :
        return load_config(sys.argv[1])
    else:
        print("Please specific arg[2] with [f_name].yaml")
        return None
    

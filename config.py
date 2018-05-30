import yaml
import sys

DATA_POOL = 'data_pool/'

cfg = []

""" For loader tuple"""
class YAMLPatch(yaml.SafeLoader):
    def construct_python_tuple(self, node):
        return tuple(self.construct_sequence(node))

YAMLPatch.add_constructor(u'tag:yaml.org,2002:python/tuple', YAMLPatch.construct_python_tuple)

def load_config(f_name = "config/default.yaml"):
    global cfg
    print("[I] Load {}".format(f_name))
    """ Load File"""
    with open(f_name, 'r') as stream:
        try:
            cfg = yaml.load(stream, Loader=YAMLPatch)
            # print(yaml.dump(cfg))
        except yaml.YAMLError as exc:
            print(exc)


if len(sys.argv) >= 2 and sys.argv[1].endswith('.yaml'):
    load_config(sys.argv[1])
else:
    load_config()


def set_none_if_not_exist():
    cfg['RL']['train_run_steps'] = cfg['RL']['train_run_steps'] if 'train_run_steps' in cfg['RL'] else None
    cfg['RL']['reward_reverse_norm'] = cfg['RL']['reward_reverse_norm'] if 'reward_reverse_norm' in cfg['RL'] else None


def get_yaml_name():
    from os.path import basename, splitext
    f_name = basename(sys.argv[1])
    f_name = splitext(f_name)[0]
    print('f_name = ' , f_name)
    return f_name



set_none_if_not_exist()
# train_run_steps = cfg['RL']['train_run_steps'] if 'train_run_steps' in cfg['RL'] else None
# print("cfg['RL']['train_run_steps'] = ", cfg['RL']['train_run_steps'])


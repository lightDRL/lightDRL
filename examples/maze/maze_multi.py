# python maze_multi.py DQN.yaml DQN_redirect_stdout_2_file.yaml
from os.path import splitext
import sys, os
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)),'../../'))
from run_multiprocess import basic_multiprocess_from_arg
from config_selfload import load_config

def process_func(p_index, yaml_path):
    # start = time.time()
    # print('process id:', os.getpid(), 'start')
    # sum=0
    # for i in range(int(1e17)):
    #     if check.value:
    #         with lock:
    #             step.value = i
    #             check.value = False
    #     sum+=i
    # print('process id:', os.getpid(), 'finish, sum = ', sum,', use time= ', time.time() - start)
    
    
    yaml_name = splitext(yaml_path)[0]
    cfg = load_config(yaml_path)

    print('random_seed = ', cfg['misc']['random_seed'])
    print('yaml_name = ', yaml_name)

    from maze_standalone import MazeStandalone
    c = MazeStandalone(cfg, project_name='flappy-' + yaml_name + '-' + str(p_index))
    c.run()


if __name__ == '__main__':
    basic_multiprocess_from_arg(process_func)
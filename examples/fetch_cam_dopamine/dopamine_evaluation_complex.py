# py dopamine_evaluation_complex.py -p rainbow_orinetwork_rgb_84_3obj_complex_obj_r_measure

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dopamine.agents.dqn import dqn_agent
from dopamine.agents.implicit_quantile import implicit_quantile_agent
from dopamine.agents.rainbow import rainbow_rgb_agent

# import TS_run_experiment
import eval_run_experiment
import tensorflow as tf
import sys, os, time
import argparse



def create_agent(sess, summary_writer=None):
  return rainbow_rgb_agent.RainbowRGBAgent(
        sess, num_actions=5,
        summary_writer=summary_writer)


def create_fetch_cam_environment(is_render, real_bot=True):
  if not real_bot:
    sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__) )+ '/../fetch_camera/'))
    from fetch_cam import FetchDiscreteCamEnv
    env = FetchDiscreteCamEnv(dis_tolerance = 0.001, use_tray = False, step_ds=0.005, gray_img = False, only_show_obj0=False, is_render=True)
    return env
  else:
    from wrs_env import WRSEnv
    env = WRSEnv()

    return env
  # env = FetchDiscreteCamEnv(dis_tolerance = 0.001, use_tray = False, step_ds=0.005, gray_img = False, only_show_obj0=False, is_render=True)
  # return env

class EvalClass:
  def __init__(self, ):
    self.start_time = time.time()
    self.EP_Success = 0
    self.EP_Overstep = 0
    self.EP_Fail = 0
    self.sum_steps = 0

  def evaluation_done_cb(self, ep = None, terminal_reward = None, ep_use_steps = None):
    
    # print('in Fetch_Cam_Standalone() ep_done_cb()!, terminal_reward = ', terminal_reward)
    if terminal_reward >0.2:
        self.EP_Success+=1
    elif terminal_reward ==0:
        self.EP_Overstep+=1
    elif terminal_reward ==-1:
        self.EP_Fail+=1
    else:
        print('Strange Reward!! terminal_reward = ', terminal_reward)

    if ep_use_steps != None:
      self.sum_steps+=ep_use_steps

    t = time.time() - self.start_time
    hms = '%02dh%02dm%02ds' % (t/3600, t/60 % 60  , t%60)
    avg_step_per_second = self.sum_steps/int(t)
    sum_count = self.EP_Success+self.EP_Overstep+self.EP_Fail
    success_ratio = self.EP_Success / sum_count
    print(f'EP={ep:5d}, EP_Success={self.EP_Success:5d}, EP_Fail={self.EP_Fail:5d}, EP_Overstep={self.EP_Overstep:5d},sum_count={sum_count:5d}, t = {hms}, success_ratio={success_ratio:5.2f}')




parser = argparse.ArgumentParser()
parser.add_argument("-p", "--project",  help="input project name ")
args = parser.parse_args()

print('parser.project = ', args.project)

# set runner
# base_dir = '../../experiment/pick_3obj_rgb/%s' % args.project
base_dir = '../../experiment/pick_complex/%s' % args.project
gin_list = ['rgb.gin']

tf.logging.set_verbosity(tf.logging.INFO)
eval_run_experiment.load_gin_configs(gin_list, [])
e = EvalClass()
runner =  eval_run_experiment.Runner(base_dir, create_agent, gpu_ratio=0.1, \
                            evaluation_done_cb = e.evaluation_done_cb, evaluation_max_ep=10000,
                            is_render=True, create_environment_fn = create_fetch_cam_environment)
runner.run_experiment()
    
    
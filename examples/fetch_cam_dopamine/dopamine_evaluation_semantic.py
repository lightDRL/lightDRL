# binary img
# py dopamine_evaluation_semantic.py -p rainbow_bin_r_measure
# py dopamine_evaluation_semantic.py -p segnet_CHO_r_0_3cm

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dopamine.agents.rainbow import rainbow_agent

# import TS_run_experiment
import eval_run_experiment
import tensorflow as tf
import sys, os, time
import argparse



def create_agent(sess, summary_writer=None):
  return rainbow_agent.RainbowAgent(
        sess, num_actions=5,
        summary_writer=summary_writer)


def create_fetch_cam_environment(is_render, env_type ='sim'):
  if env_type=='sim':
    sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__) )+ '/../fetch_camera/'))
    from fetch_cam.fetch_discrete_cam import FetchDiscreteCamEnv, IMG_TYPE, EnvType
    
    env = FetchDiscreteCamEnv(env_type=EnvType.ThreeObj,dis_tolerance = 0.001, step_ds=0.005, img_type = IMG_TYPE.SEMANTIC, only_show_obj0=True, is_render=True)
    return env
  elif env_type=='feed':
    sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__) )+ '/../fetch_camera/'))
    from feed_img_env.feed_img_env import FeedImgEnv
    from fetch_cam import IMG_TYPE

    env = FeedImgEnv(img_type = IMG_TYPE.SEMANTIC)
    return env
  elif env_type=='real':
    from wrs_env import WRSEnv, IMG_TYPE
    env = WRSEnv(img_type = IMG_TYPE.SEMANTIC)

    return env
  else:
    assert False, 'Fase Env Name'
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
    elif terminal_reward >=-0.1 and terminal_reward <0.2:
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
base_dir = '../../experiment/semantic/%s' % args.project
gin_list = ['rainbow.gin']

tf.logging.set_verbosity(tf.logging.INFO)
eval_run_experiment.load_gin_configs(gin_list, [])
e = EvalClass()
runner =  eval_run_experiment.Runner(base_dir, create_agent, gpu_ratio=0.1, \
                            evaluation_done_cb = e.evaluation_done_cb, evaluation_max_ep=10000,
                            is_render=True, create_environment_fn = create_fetch_cam_environment)
runner.run_experiment()
    
    
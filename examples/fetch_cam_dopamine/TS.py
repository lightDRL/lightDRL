# Copyright 2018 The Dopamine Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dopamine.agents.dqn import dqn_agent
from dopamine.agents.implicit_quantile import implicit_quantile_agent
from dopamine.agents.rainbow import rainbow_agent

import TS_run_experiment
import tensorflow as tf
import sys, os
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)),'../../'))
from TS_run_multiprocess import thompson_sample, AgentPlayEnv, read_yaml


def create_agent(sess, environment, summary_writer=None):
  """Creates a DQN agent.

  Args:
    sess: A `tf.Session` object for running associated ops.
    environment: An Atari 2600 Gym environment.
    summary_writer: A Tensorflow summary writer to pass to the agent
      for in-agent training statistics in Tensorboard.

  Returns:
    agent: An RL agent.

  Raises:
    ValueError: If `agent_name` is not in supported list.
  """
  
  return rainbow_agent.RainbowAgent(
        sess, num_actions=5,
        summary_writer=summary_writer)


def process_func(p_index, conn, ready, dic, lock):
    conn.recv()

    # init AgentPlayEnv
    prj_name = 'fetch_cam_rainbow-{:03d}'.format(p_index) 
    agent = AgentPlayEnv(prj_name)
    agent.set_process_switch(conn, ready, dic, lock)
    agent.set_success(1.0, 50)
    agent.mark_start_time()

    # set runner
    base_dir = '../../data_pool/TS_fetch_cam_rainbow_%03d' % p_index
    gin_list = ['TS_rainbow.gin']

    tf.logging.set_verbosity(tf.logging.INFO)
    TS_run_experiment.load_gin_configs(gin_list, [])
    runner =  TS_run_experiment.Runner(base_dir, create_agent, gpu_ratio=0.1)
    runner.set_ep_done_cb(agent.ep_done_cb)
    runner.run_experiment()
     
    
if __name__ == '__main__':
  thompson_sample(process_func, p_num =3, pool_max = 1, parse_yaml = False)

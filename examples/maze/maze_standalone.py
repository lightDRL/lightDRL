#   python maze_standalone.py DQN.yaml
#   python maze.py Q-learning.yaml
#   
# Author:  kbehouse  <https://github.com/kbehouse/>
#          

import sys, os
import time
import numpy as np
from maze_env import Maze
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)),'../../'))
from config import cfg, get_yaml_name
from standalone import Standalone


class MazeStandalone(Standalone):
    def env_init(self):
        self.env = Maze()    
    
    def env_reset(self):
        self.state = self.env.reset()
        return self.state

    def on_action_response(self, action):
        next_state, reward, done, _ = self.env.step(action)
        now_state = self.state
        self.state = next_state
        self.env.render()   

        return now_state, reward, done, next_state


if __name__ == '__main__':
    c = MazeStandalone(cfg, project_name='maze-' + get_yaml_name()).run()
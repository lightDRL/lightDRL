# Run with A3C
#   python server.py               config/two_dof_arm_A3C.yaml
#   python examples/two_dof_arm.py config/two_dof_arm_A3C.yaml
#   
# Author:  Kartik,Chen  <kbehouse(at)gmail(dot)com>
#          

import sys, os
sys.path.append(os.path.abspath(os.path.dirname(__file__)+'/../'))
from client import Client, EnvSpace
from envs.arm_env import ArmEnv
import time

TRAIN_MODE = True
class TwoDofArm(EnvSpace):
    def env_init(self):
        self.render = False
        self.EP_MAXSTEP = 300
        self.env = ArmEnv(mode='hard')
        self.state = self.env.reset()
        self.send_state_get_action(self.state)

    def on_predict_response(self, action):
        next_state, reward, done, _ = self.env.step(action)
        done = True if self.ep_use_step >= self.EP_MAXSTEP else done
        self.send_train_get_action(self.state, action, reward, done, next_state)
        self.state = next_state
    
        # print('self.env_name=',self.env_name)
        if self.env_name =='Env-0':
            self.env.render()   
        if done:
            self.state =  self.env.reset()
            self.send_state_get_action(self.state)


if __name__ == '__main__':
    for i in range(4):
        c = Client(TwoDofArm, env_name='Env-%d' % i)
        c.start()
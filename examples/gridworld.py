# Run With SARSA
#   python server.py config/gridworld_SARSA.yaml
#   python examples/gridworld.py config/gridworld_SARSA.yaml
# Run with QLearning
#   python server.py config/gridworld_QLearning.yaml
#   python examples/gridworld.py config/gridworld_QLearning.yaml
# Run With A3C
#   python server.py config/gridworld_A3C.yaml
#   python examples/gridworld.py config/gridworld_A3C.yaml

import sys, os
import Queue # matplotlib cannot plot in main thread,
import time
import gym  #OpenAI gym
# append this repo's root path
sys.path.append(os.path.abspath(os.path.dirname(__file__)+'/../'))
from client import Client, EnvSpace
import envs
class Gridworld_EX(EnvSpace):

    def env_init(self):
        self.EP_MAXSTEP = 1000
        self.env = gym.make('gridworld-v0')
        self.state = self.env.reset()

        self.send_state_get_action(self.state)

    def on_predict_response(self, action):
        next_state, reward, done, _ = self.env.step(action)
        if self.ep_use_step >= self.EP_MAXSTEP:
            done = True

        if self.ep % 50 == 25:
            self.env._render(title = 'Episode: %3d, Step: %3d' % (self.ep,self.ep_use_step+1))

        self.send_train_get_action(self.state, action, reward, done, next_state)
        self.state = next_state 
        if done:
            self.state =  self.env.reset()
            self.send_state_get_action(self.state)

if __name__ == '__main__':
    c = Client(Gridworld_EX,'Main-Env')
    c.start()
        
# Run with A3C
#   python server.py               config/cartpole_A3C.yaml
#   python examples/gym-carpole.py config/cartpole_A3C.yaml
#   
#  Author:  Kartik, Chen  <kbehouse(at)gmail(dot)com>,
#          

import sys, os, time
import gym
sys.path.append(os.path.abspath(os.path.dirname(__file__)+'/../'))
from client import Client

GymGame = 'CartPole-v0'
GLOBAL_EP = 0
START_TIME = time.time()

class Cartpole_EX:
    """ Init Client """
    def __init__(self, client_id):
        self.done = True

        self.env = gym.make(GymGame) 
        self.env.reset()
        self.client = Client(client_id)
        self.client.set_state(self.get_state)
        self.client.set_train(self.train)
        self.client.start()

    def get_state(self):
        if self.done:
            self.done = False
            self.ep_t = 0
            self.ep_r = 0
            self.next_state =  self.env.reset()

        self.state = self.next_state
        
        return self.state

    def train(self,action):
        self.next_state, reward, self.done, _ = self.env.step(action)
        self.ep_t += 1
        self.ep_r += reward

        if self.done:
            self.log_and_show()

        return (reward, self.done, self.state)

    def log_and_show(self):
        global GLOBAL_EP, START_TIME

        if self.client.client_id == 'Client-0':
            self.env.render()

        # if self.done:
        use_secs = time.time() - START_TIME
        time_str = '%3dh%3dm%3ds' % (use_secs/3600, (use_secs%3600)/60, use_secs % 60 )
        print('%s -> EP:%4d, STEP:%3d, EP_R:%3d, T:%s' % (self.client.client_id,  GLOBAL_EP, self.ep_t, self.ep_r, time_str))
            
        GLOBAL_EP += 1


if __name__ == '__main__':
    for i in range(4):
        Cartpole_EX('Client-%d' % i ) 
   
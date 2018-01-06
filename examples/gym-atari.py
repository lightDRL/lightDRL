#
#   Send raw picture to server.py
#   Get gary image(84x84) from server (use worker)  
#   Save the gray image(84x84)
#   Modify from ZMQ example (http://zguide.zeromq.org/py:lpclient)
#   
#   Author:  Kartik, Chen  <kbehouse(at)gmail(dot)com>,
#          

import cv2
import sys, os
import gym
import scipy.misc
import numpy as np
sys.path.append(os.path.abspath(os.path.dirname(__file__)+'/../'))
from client import Client
from config import STATE_SHAPE

Atari_Game_Name = 'Breakout-v0'

class Atari:
    """ Init Client """
    def __init__(self, client_id):
        self.done = True

        self.env = gym.make(Atari_Game_Name) 
        self.env.reset()
        self.client = Client(client_id)
        self.client.set_state(self.get_state)
        self.client.set_train(self.train)
        self.client.start()

    def get_state(self):
        if self.done:
            self.done = False
            s = self.env.reset()
        else:
            s = self.state

        return self.state_preprocess(s)

    def train(self,action):
        self.state, reward, self.done, _ = self.env.step(action)
        return (reward, self.done)

    
    def state_preprocess(self,state_im):
        y = 0.2126 * state_im[:, :, 0] + 0.7152 * state_im[:, :, 1] + 0.0722 * state_im[:, :, 2]
        y = y.astype(np.uint8)
        resized = scipy.misc.imresize(y, STATE_SHAPE)

        return resized


if __name__ == '__main__':
    for i in range(5):
        Atari('Client-%d' % i ) 
   
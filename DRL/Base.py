#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: DRL.py
# Description: Abstract Class For DRL Methods
# Author: Kartic <kbehouse@gmail.com>

import six
from abc import ABCMeta,abstractmethod
import numpy as np
from config import cfg


@six.add_metaclass(ABCMeta)
class RL(object):

    def read_config(self):
        self.a_dim = np.squeeze(cfg['RL']['action_num']) 
        self.s_dim = np.squeeze(cfg['RL']['state_shape'])
        if not cfg['RL']['action_discrete']:    
            self.a_bound = cfg['RL']['action_bound']

        self.r_dicount = self.reward_discount = cfg['RL']['reward_discount']


    @abstractmethod
    def choose_action(self, state):
        pass   

    @abstractmethod
    def train(self, states, actions, rewards, next_state, done):
        pass

@six.add_metaclass(ABCMeta)
class DRL(RL):

    @abstractmethod
    def _build_net(self, msg):
        pass

    
    
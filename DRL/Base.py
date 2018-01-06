#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: DRL.py
# Description: Abstract Class For DRL Methods
# Author: Kartic <kbehouse@gmail.com>

import six
from abc import ABCMeta,abstractmethod

@six.add_metaclass(ABCMeta)
class RL(object):
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

    
    
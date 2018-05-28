#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: DRL.py
# Description: Abstract Class For DRL Methods
# Author: Kartic <kbehouse@gmail.com>

import six
from abc import ABCMeta,abstractmethod
import numpy as np
import tensorflow as tf
import os 

@six.add_metaclass(ABCMeta)
class RL(object):

    def rl_init(self, cfg, model_log_dir):
        self.ep = 0         # episode
        self.model_log_dir = model_log_dir
        # self.cfg = cfg
        self.set_rl_basic_from_config(cfg)

        print('rl_init() model_log_dir = ' + self.model_log_dir)
        
    def set_rl_basic_from_config(self, cfg):
        self.a_dim = np.squeeze(cfg['RL']['action_num']) 
        self.s_dim = np.squeeze(cfg['RL']['state_shape'])
        if not cfg['RL']['action_discrete']:    
            self.a_bound = cfg['RL']['action_bound']

        self.r_dicount = self.reward_discount = cfg['RL']['reward_discount']

        
        self.model_save_cycle = cfg['misc']['model_save_cycle'] if ('misc' in cfg) and ('model_save_cycle' in cfg['misc']) else None

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

    @abstractmethod
    def train(self,  s, a, r, s_, done):
        self.train_times+=1
        
        if done and self.model_save_cycle!=None:
            self.ep +=1
            if self.ep % self.model_save_cycle ==0:
                self.save_model(self.model_log_dir, self.ep)


    def drl_init(self, sess):
        self.train_times = 0
        self.sess = sess

    def save_model(self, save_model_dir, g_step):
        saver = tf.train.Saver()
        save_path = os.path.join(self.model_log_dir , 'model.ckpt') 
        save_ret = saver.save(self.sess, save_path, global_step = g_step)

        print('Save Model to ' + save_ret  + ' !')


    def init_or_restore_model(self, sess,  model_dir = None ):
        if model_dir == None:
            model_dir = self.model_log_dir
        assert model_dir != None, 'init_or_restore_model model_dir = None'
        model_file = tf.train.latest_checkpoint(model_dir)
        start_ep = 0
        if model_file is not None:
            #-------restore------#e 
            ind1 = model_file.index('model.ckpt')
            start_ep = int(model_file[ind1 +len('model.ckpt-'):]) + 1
            saver = tf.train.Saver()
            saver.restore(sess, model_file)

            print('[I] Use model_file = ' + str(model_file) + ' ! Train from epoch = %d' % start_ep )
            
        else:
            print('[I] Initialize all variables')
            sess.run(tf.global_variables_initializer())
            print('[I] Initialize all variables Finish')

    
    
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: DRL.py
# Description: Abstract Class For DRL Methods
# Author: Kartic <kbehouse@gmail.com>

import six
from abc import ABCMeta,abstractmethod
import numpy as np
from config import cfg, DATA_POOL
import tensorflow as tf
import os 

@six.add_metaclass(ABCMeta)
class RL(object):

    def rl_init(self):
        self.ep = 0         # episode

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

    @abstractmethod
    def train(self,  s, a, r, s_, done):
        self.train_times+=1
        
        if done:
            self.ep +=1
            if self.ep % cfg['log']['model_save_cycle'] ==0:
                self.save_model(self.model_log_dir, self.ep)


    def drl_init(self, sess, project_name = None):
        super(DRL, self).rl_init()
        self.train_times = 0
        self.sess = sess
        self.create_model_log_dir(project_name)

    
    def create_model_log_dir(self, project_name):
        # self.model_log_dir = '{}/{}/'.format(DATA_POOL, project_name)   if project_name != None else None
        self.model_log_dir = os.path.join(DATA_POOL, project_name) if project_name != None else None
        print('self.model_log_dir = ', self.model_log_dir)
        if self.model_log_dir !=None:
            if not os.path.isdir(self.model_log_dir):
                os.mkdir(self.model_log_dir)


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

    
    
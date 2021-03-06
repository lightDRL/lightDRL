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
        self.set_rl_basic_from_config(cfg)

        # print('rl_init() model_log_dir = ' + self.model_log_dir)
        # self.check_s_a_dim()
        
    def set_rl_basic_from_config(self, cfg):
        # self.a_dim = np.squeeze(cfg['RL']['action_shape'])   # continuous-> OK, discrete: onehost
        self.a_shape = cfg['RL']['action_shape']   
        
        self.a_discrete = cfg['RL']['action_discrete']  # true -> discrete, false-> continuous

        if  cfg['RL']['action_discrete']:    
            self.a_discrete_n = cfg['RL']['action_discrete_n']
            self.a_bound = 1.
            self.a_dim = self.a_discrete_n
        else:
            self.a_bound = cfg['RL']['action_bound']
            self.a_dim = np.squeeze(np.array(self.a_shape))

        # state
        # print("cfg['RL']['state_shape']=", cfg['RL']['state_shape'])
        # self.s_dim = np.squeeze(cfg['RL']['state_shape'])
        self.s_shape = cfg['RL']['state_shape']
        self.s_dim =  np.squeeze(np.array(self.s_shape))
        self.s_discrete = cfg['RL']['state_discrete']
        if self.s_discrete:
            self.s_discrete_n = cfg['RL']['state_discrete_n']
        
        # self.r_discount = self.reward_discount = cfg['DRL']['reward_discount']
        # self.r_reverse_norm =  self.reward_reverse_norm = cfg['RL']['reward_reverse_norm']
        
        self.model_save_cycle = cfg['misc']['model_save_cycle'] if ('misc' in cfg) and ('model_save_cycle' in cfg['misc']) else None


    def check_s_a_dim(self):
        print('self.a_shape = ', self.a_shape)
        print('self.a_dim = ', self.a_dim)
        print('self.a_bound = ', self.a_bound)
        print('self.action_discrete = ', self.a_discrete)
        if self.a_discrete:
            print('self.a_discrete_n = ', self.a_discrete_n )
        

        print('self.s_shape = ', self.s_shape)
        print('self.s_dim = ', self.s_dim)
        print('self.s_discrete = ', self.s_discrete)
        if self.s_discrete:
            print('self.s_discrete_n = ', self.s_discrete_n)

    @abstractmethod
    def choose_action(self, state):
        pass   

    @abstractmethod
    #def train(self, states, actions, rewards, done, next_state):
    def train(self):
        pass

    @abstractmethod
    def add_data(self, states, actions, rewards, done, next_state):
        pass
    
  
@six.add_metaclass(ABCMeta)
class DRL(RL):


    @abstractmethod
    def _build_net(self, msg):
        pass

    @abstractmethod
    # def train(self,  s, a, r, done, s_):
    def train(self):
        self.train_times+=1
        
        # if done and self.model_save_cycle!=None:
        #     self.ep +=1
        #     if self.ep % self.model_save_cycle ==0:
        #         self.save_model(self.model_log_dir, self.ep)
            # print('----------------EP=%d--------' % self.ep)

    @abstractmethod
    def add_data(self, states, actions, rewards, done, next_state):
        pass


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

    

    def onehot(self, argmax_ary):
        assert self.a_dim != 0, 'self.a_dim == 0 or None'
        # onehot_ary = np.zeros(self.a_dim) 
        # onehot_ary[argmax] = 1
        # return onehot_ary 
        return  np.eye(self.a_dim)[argmax_ary]

    def reverse_and_norm_rewards(self, ep_rs, r_dicount = 0.9):
        #print('reverse_and_norm_rewards ep_rs -> len = {}, {}'.format(len(ep_rs), ep_rs))

        # discount episode rewards
        discounted_ep_rs = np.zeros_like(ep_rs)
        running_add = 0
        for t in reversed(range(0, len(ep_rs))):
            running_add = running_add * r_dicount + ep_rs[t]
            discounted_ep_rs[t] = running_add

        mean = np.mean(discounted_ep_rs)
        std = np.std(discounted_ep_rs)
        discounted_ep_rs = (discounted_ep_rs- mean) /std
        # discounted_ep_rs -= np.mean(discounted_ep_rs)
        # discounted_ep_rs /= np.std(discounted_ep_rs)
        #print('reverse_and_norm_rewards -> discounted_ep_rs = ' + str(discounted_ep_rs))
        return discounted_ep_rs

    def reverse_add_rewards(self, ep_rs, r_dicount = 0.9):
        # print('reverse_and_norm_rewards ep_rs -> len = {}, {}'.format(len(ep_rs), ep_rs))
        # discount episode rewards
        discounted_ep_rs = np.zeros_like(ep_rs)
        running_add = 0
        for t in reversed(range(0, len(ep_rs))):
            running_add = running_add * r_dicount + ep_rs[t]
            discounted_ep_rs[t] = running_add
            
        # print('reverse_add_rewards -> discounted_ep_rs = ' + str(discounted_ep_rs))
        return discounted_ep_rs
        

    
    
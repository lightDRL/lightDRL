
#
#   Modify from ?: ?? https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow
#   

import tensorflow as tf
import numpy as np
import time
import sys, os
sys.path.append(os.path.abspath(os.path.dirname(__file__)+'/../'))
from Base import DRL
from collections import deque 
import random

class DDPG(DRL):
    def __init__(self, cfg, model_log_dir, sess):
        super(DDPG, self).rl_init(cfg, model_log_dir)
        super(DDPG, self).drl_init(sess)
        
        print('DDPG() model_log_dir = ' + self.model_log_dir)

        if not self.action_discrete:
            self.a_bound = self.a_bound[1]
        self.memory_capacity = cfg['DDPG']['memory_capacity']
        self.batch_size = cfg['DDPG']['batch_size']
        self.lr_actor = cfg['DDPG']['lr_actor']
        self.lr_critic = cfg['DDPG']['lr_critic']
        self.exp_decay = cfg['DDPG']['exp_decay']
        

        self.memory = deque()
        self.pointer = 0
        # self.sess = sess

        self.S = tf.placeholder(tf.float32, [None, self.s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, self.s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')
        self.a = self._build_a(self.S)

        q = self._build_c(self.S, self.a, )
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope= 'actor')   #scope +'/actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope= 'critic') # scope +'/critic')
        ema = tf.train.ExponentialMovingAverage(decay=1 - self.exp_decay)          # soft replacement

        def ema_getter(getter, name, *args, **kwargs):
            return ema.average(getter(name, *args, **kwargs))

        target_update = [ema.apply(a_params), ema.apply(c_params)]      # soft update operation
        a_ = self._build_a(self.S_, reuse=True, custom_getter=ema_getter)   # replaced target parameters
        q_ = self._build_c(self.S_, a_, reuse=True, custom_getter=ema_getter)

        a_loss = - tf.reduce_mean(q)  # maximize the q
        self.atrain = tf.train.AdamOptimizer(self.lr_actor).minimize(a_loss, var_list=a_params)

        with tf.control_dependencies(target_update):    # soft replacement happened at here
            q_target = self.R + self.r_discount * q_
            td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
            self.ctrain = tf.train.AdamOptimizer(self.lr_critic).minimize(td_error, var_list=c_params)

        # self.sess.run(tf.global_variables_initializer())
        # self.sess.run(tf.local_variables_initializer())

    def choose_action(self, s):
        if self.action_discrete:
            probs =  self.sess.run(self.a, {self.S: s[np.newaxis, :]})
            # print('probs', probs)
            action = np.random.choice(range(probs.shape[1]), p=probs.ravel())
            return action
        else:
            return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

    def train(self,  s, a, r, s_, done):
        if self.action_discrete:
            a = self.onehot(a)

        if self.reward_reverse_norm and len(r) > 1:
            # r = self.reverse_and_norm_rewards(r, self.r_discount)
            r = self.reverse_add_rewards(r, self.r_discount)


        self.store_transition(s, a, r , s_)
        if len(self.memory) < self.memory_capacity:
            return
         
        bt = random.sample(self.memory,self.batch_size)
        bt = np.array(bt)
        # print('bt.shape = ' + str(bt.shape))
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

        super(DDPG, self).train(s, a, r, s_, done)
        # print('train model finish, self.train_times =' + str(self.train_times))

    def store_transition(self, s, a, r, s_):
        # print('start-------in store_transition--------')
        # print('s.shape', s.shape)
        # print('r.shape', r.shape)
        # r = r[:, np.newaxis]
        if type(r) == list or type(r) == np.ndarray:      #mulitple step restore
            r = np.array(r)
            r = r[:, np.newaxis]  if len(r.shape) <= 1 else r
            transition = np.hstack((s, a, r, s_))
            for d in transition:
                self.memory.append(d)
        else:
            transition = np.hstack((s, a, r, s_))
            self.memory.append(transition)

        
        while len(self.memory) > self.memory_capacity:
			self.memory.popleft()


    def _build_a(self, s, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('actor', reuse=reuse, custom_getter=custom_getter):
            net = tf.layers.dense(s, 300, activation=tf.nn.relu, name='l1', trainable=trainable)
            net = tf.layers.dense(net, 300, activation=tf.nn.relu, name='l2', trainable=trainable)
            
            if self.action_discrete: 
                a = tf.layers.dense(net, self.a_dim, activation=tf.nn.relu6, name='a', trainable=trainable)
                return tf.nn.softmax(a, name='a_prob')  # use softmax to convert to probability
            else:
                a = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
                return tf.multiply(a, self.a_bound, name='scaled_a')

    def _build_c(self, s, a, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('critic', reuse=reuse, custom_getter=custom_getter):
            n_l1 = 300
            s_n1 = tf.layers.dense(s , n_l1, activation=tf.nn.relu, name='l1_s', trainable=trainable)
            w1_s = tf.get_variable('w1_s', [n_l1, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)

            net = tf.nn.relu(tf.matmul(s_n1, w1_s) + tf.matmul(a, w1_a) + b1)
            return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)
   
    # override
    def _build_net(self):
        pass

if __name__ == '__main__':
    sess = tf.Session()

    DDPG(sess)
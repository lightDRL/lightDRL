
#
#   Send raw picture to server.py
#   Get gary image(84x84) from server (use worker)  
#   Save the gray image(84x84)
#   
#   Author:  Kartik, Chen  <kbehouse(at)gmail(dot)com>,
#      

import tensorflow as tf
import numpy as np
from config import cfg
from Base import DRL


class A3C(DRL):
    def __init__(self, sess, scope, globalAC=None):
        self.sess = sess
        self.OPT_A = tf.train.RMSPropOptimizer(cfg['A3C']['LR_A'], name='RMSPropA')
        self.OPT_C = tf.train.RMSPropOptimizer(cfg['A3C']['LR_C'], name='RMSPropC')

        if scope == cfg['A3C']['main_net_scope']:   # get global network
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, cfg['RL']['state_shape'][0] ], 'S')
                self._build_net()
                self.a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
                self.c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
        else:   # local net, calculate losses
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, cfg['RL']['state_shape'][0] ], 'S')
                self.a_his = tf.placeholder(tf.float32, [None, cfg['RL']['action_num']], 'A')
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')

                mu, sigma, self.v = self._build_net()

                td = tf.subtract(self.v_target, self.v, name='TD_error')
                with tf.name_scope('c_loss'):
                    self.c_loss = tf.reduce_mean(tf.square(td))

                with tf.name_scope('wrap_a_out'):
                    self.test = sigma[0]
                    mu, sigma = mu * cfg['RL']['action_bound'][1], sigma + 1e-5

                normal_dist = tf.contrib.distributions.Normal(mu, sigma)

                with tf.name_scope('a_loss'):
                    log_prob = normal_dist.log_prob(self.a_his)
                    
                    exp_v = log_prob * td
                    
                    entropy = normal_dist.entropy()  # encourage exploration
                    self.exp_v = cfg['A3C']['ENTROPY_BETA'] * entropy + exp_v
                    self.a_loss = tf.reduce_mean(-self.exp_v)

                    # print('normal_dist',normal_dist)
                    # print('log_prob',log_prob)
                    # print('exp_v',exp_v)

                with tf.name_scope('choose_a'):  # use local params to choose action
                    self.A = tf.clip_by_value(tf.squeeze(normal_dist.sample(1), axis=0), *cfg['RL']['action_bound'])
                with tf.name_scope('local_grad'):
                    self.a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
                    self.c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
                    self.a_grads = tf.gradients(self.a_loss, self.a_params)
                    self.c_grads = tf.gradients(self.c_loss, self.c_params)

            with tf.name_scope('sync'):
                with tf.name_scope('pull'):
                    self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, globalAC.a_params)]
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, globalAC.c_params)]
                with tf.name_scope('push'):
                    self.update_a_op = self.OPT_A.apply_gradients(zip(self.a_grads, globalAC.a_params))
                    self.update_c_op = self.OPT_C.apply_gradients(zip(self.c_grads, globalAC.c_params))

            print('----------tf.report_uninitialized_variables()----------')
            print(self.sess.run(tf.report_uninitialized_variables()))
            # tf.variables_initializer(
            #     [v for v in tf.global_variables() if v.name.split(':')[0] in set(sess.run(tf.report_uninitialized_variables()))
            # ])
            self.sess.run(tf.global_variables_initializer())

            print('--again--------tf.report_uninitialized_variables()----------')

    def _build_net(self):
        w_init = tf.contrib.layers.xavier_initializer()
        with tf.variable_scope('actor'):
            l_a = tf.layers.dense(self.s, 400, tf.nn.relu6, kernel_initializer=w_init, name='la')
            l_a = tf.layers.dense(l_a, 300, tf.nn.relu6, kernel_initializer=w_init, name='la2')
            mu = tf.layers.dense(l_a, cfg['RL']['action_num'], tf.nn.tanh, kernel_initializer=w_init, name='mu')
            sigma = tf.layers.dense(l_a, cfg['RL']['action_num'], tf.nn.softplus, kernel_initializer=w_init, name='sigma')
        with tf.variable_scope('critic'):
            l_c = tf.layers.dense(self.s, 400, tf.nn.relu6, kernel_initializer=w_init, name='lc')
            l_c = tf.layers.dense(l_c, 200, tf.nn.relu6, kernel_initializer=w_init, name='lc2')
            v = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name='v')  # state value
        return mu, sigma, v

    def update_global(self, feed_dict):  # run by a local
        _, _, t = self.sess.run([self.update_a_op, self.update_c_op, self.test], feed_dict)  # local grads applies to global net
        return t

    def pull_global(self):  # run by a local
        self.sess.run([self.pull_a_params_op, self.pull_c_params_op])

    def choose_action(self, s):  # run by a local
        s = s[np.newaxis, :]
        return self.sess.run(self.A, {self.s: s})[0]

    def train(self,  states, actions, rewards, next_state, done):
        if done:
            v_s_ = 0   # terminal
        else:
            v_s_ = self.sess.run(self.v, {self.s: next_state[np.newaxis, :]})[0, 0]
        buffer_v_target = []

        # if type(rewards) == list: 
        #     rewards = np.array(rewards)
        # elif type(rewards) != np.ndarray: 
        #     ''' maybe flost, int'''
        #     rewards = np.array([rewards])

        for r in rewards[::-1]:    # reverse buffer r
            v_s_ = r + cfg['A3C']['gamma'] * v_s_
            buffer_v_target.append(v_s_)
        buffer_v_target.reverse()
        

        buffer_s, buffer_a, buffer_v_target = np.vstack(states), np.vstack(actions), np.vstack(buffer_v_target)

        feed_dict = {
            self.s: buffer_s,
            self.a_his: buffer_a,
            self.v_target: buffer_v_target,
        }

        self.update_global(feed_dict)
        self.pull_global()
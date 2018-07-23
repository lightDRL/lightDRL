"""
Deep Q-network (DQN)
Modify from: https://github.com/floodsung/DRL-FlappyBird
"""
import tensorflow as tf
import numpy as np
from Base import DRL
from component.replay_memory import ReplayMemory
from component.DNN_v3 import FC
from component.NNcomponent import NNcomponent
# Network Parameters - Hidden layers

class DQN(DRL):
    def __init__(self, cfg, model_log_dir="", sess = None):
        super(DQN, self).rl_init(cfg, model_log_dir)
        super(DQN, self).drl_init(sess)
        self.memory_capacity = cfg['DQN']['memory_capacity']
        self.memory_train_min = cfg['DQN']['memory_train_min']
        self.batch_size = cfg['DQN']['batch_size']
        self.update_Q_target_times = cfg['DQN']['update_Q_target_times']    
        self.lr = cfg['DQN']['lr']                              # learning rate
        self.gamma = cfg['RL']['reward_gamma']
        self.cfg_network = cfg['NN']
        self.sess = sess
        # init replay memory
        assert self.memory_capacity > self.memory_train_min, "Replay memory: memory_capacity < memory_train_min"
        self.mem = ReplayMemory(self.memory_capacity)
        self._build_net()

        self.notify_ep_done()  # only  for reset variable

    def choose_action(self, s):
        q_value = self.sess.run( self.QValue, feed_dict={self.stateInput:[s]})
        action = q_value[0]

        return action


    
    def add_data(self, s, a, r, d, s_):
        ''' self, states, actions, rewards, done, next_state''' 
        # print('---Before add_data----')
        # print('I: train get s.shape={}, type(s)={}'.format(np.shape(s), type(s)))
        # print('I: train get a.shape={}, type(a)={}'.format(np.shape(a), type(a)))
        # print('I: train get r.shape={}, type(r)={}'.format(np.shape(r), type(r)))
        # print('I: train get d.shape={}, type(d)={}'.format(np.shape(d), type(d)))
        # print('I: train get s_.shape={}, type(s_)={}'.format(np.shape(s_), type(s_)))
        

        self.mem.add(s, a, r, d, s_)
        # print('self.mem.size()= ' , self.mem.size())

    def train(self):
        # print(' self.mem.size() = ',  self.mem.size())
        # if self.mem.size() > self.exploration_step : #MINIBATCH_SIZE:
        if self.mem.size() > self.memory_train_min:
            # print('DQN in train')
            s_batch, a_batch, r_batch, d_batch, s2_batch = self.mem.sample_batch(self.batch_size)

            # Calculate targets
            q_target_value = self.sess.run( self.QValueT, feed_dict={self.stateInputT:s2_batch})
            
            y_i = []
            for k in range(self.batch_size):
                if d_batch[k]:
                    y_i.append(r_batch[k])
                else:
                    y_i.append(r_batch[k] + self.gamma * np.max(q_target_value[k] ))

            y_i = np.array(y_i)
            # print('s_batch = ', s_batch)
            # print('I: s_batch.shape={}, type(s_batch)={}'.format(s_batch.shape, type(s_batch)))
            # print('a_batch = ', a_batch)
            # print('I: a_batch.shape={}, type(a_batch)={}'.format(a_batch.shape, type(a_batch)))
            
            # print('y_i = ', y_i)
            # print('I: y_i.shape={}, type(y_i)={}'.format(y_i.shape, type(y_i)))
            
            q_loss, _ = self.sess.run([self.q_loss,  self.train_op], feed_dict={
                self.yInput : y_i,
                self.actionInput : a_batch,
                self.stateInput : s_batch
            })

            self.sum_q_loss += q_loss
            self.train_count += 1

            if self.train_count % self.update_Q_target_times == 0:
                self.sess.run(self.copy_Q_2_Qtarget)

     # override
    def _build_net(self):
        with tf.variable_scope('q_net'):
            self.stateInput,self.QValue = self.build_q_net()

        with tf.variable_scope('target_q_net'):
            self.stateInputT,self.QValueT = self.build_q_net()

        # print('self.stateInput = ', self.stateInput )

        # set copy op
        net_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_net')
        target_net_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_q_net')
        self.copy_Q_2_Qtarget = [tf.assign(t, e) for t, e in zip(target_net_params, net_params)]

        self.actionInput = tf.placeholder("float",[None,self.a_dim])
        self.yInput = tf.placeholder("float", [None]) 
        Q_Action = tf.reduce_sum(tf.multiply(self.QValue, self.actionInput), reduction_indices = 1)
        self.q_loss = tf.reduce_mean(tf.square(self.yInput - Q_Action))
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.q_loss)

    def build_q_net(self):
        state = tf.placeholder("float",[None, self.s_dim])
        # maze
        # fc1 = FC(state, 20, name_prefix = 'q_fc_1', op='relu', initializer = 'truncated_normal', bias_const=0.1)
        # cartpole
        # fc1 = FC(state, 200, name_prefix = 'q_fc_1', op='relu', initializer = 'truncated_normal', bias_const=0.03)
        # fc2 = FC(fc1   , 100, name_prefix = 'q_fc_2', op='relu', initializer = 'truncated_normal', bias_const=0.03)
        nn = NNcomponent(self.cfg_network, state)
        q_value = FC(nn , self.a_dim, name_prefix =  'q_value', op='tanh', initializer = 'truncated_normal', bias_const=0.03)

        return state, q_value

    def notify_ep_done(self):
        self.train_count = 0
        self.sum_q_loss = 0
        self.train_sum_critic_loss = 0

    def get_log_dic(self):
        avg_q_loss = self.sum_q_loss /self.train_count if self.train_count!=0 else 0
        log_dic = {'ep_avg_Q_loss':avg_q_loss }
        return log_dic

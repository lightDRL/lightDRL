"""
Deep Q-network (DQN)
Modify from: https://github.com/floodsung/DRL-FlappyBird
"""
import tensorflow as tf
import numpy as np
from .Base import DRL
from .component.replay_memory import ReplayMemory
from .component.DNN_v3 import FC
from .component.NNcomponent import NNcomponent
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

        self.update_target_count = 0 

    def choose_action(self, s):
        # q_value = self.sess.run( self.QValue, feed_dict={self.stateInput:[s]})
        # action = q_value[0]

        # return action
        q_value = self.sess.run( self.QValue, feed_dict={self.stateInput:[s]})
        # action = q_value[0]

        action = np.zeros(len(q_value[0]))
        action_index = np.argmax(q_value[0])
        action[action_index] = 1

        # print('action = ', action)
        return action



    
    def add_data(self, s, a, r, d, s_, add_multiple = False):
        ''' self, states, actions, rewards, done, next_state''' 
        # print('---Before add_data----')
        # print('I: train get s.shape={}, type(s)={}'.format(np.shape(s), type(s)))
        # print('I: train get a.shape={}, type(a)={}'.format(np.shape(a), type(a)))
        # print('I: train get r.shape={}, type(r)={}'.format(np.shape(r), type(r)))
        # print('I: train get d.shape={}, type(d)={}'.format(np.shape(d), type(d)))
        # print('I: train get s_.shape={}, type(s_)={}'.format(np.shape(s_), type(s_)))
        

        self.mem.add(s, a, r, d, s_, add_multiple)
        # print('self.mem.size()= ' , self.mem.size())
    def log_time(self, s):
        import time
        if hasattr(self, 'ts'):
            print(s + ' use time = ' + str( time.time() - self.ts  ))
        self.ts = time.time()

    def train(self):
        # print(' self.mem.size() = ',  self.mem.size())
        # if self.mem.size() > self.exploration_step : #MINIBATCH_SIZE:
        # self.sess.as_default()
        # self.log_time('before train')
        if self.mem.size() > self.memory_train_min:
            # print('DQN in train')
            s_batch, a_batch, r_batch, d_batch, s2_batch = self.mem.sample_batch(self.batch_size)
            
            # self.log_time('sample_batch')
            # Calculate targets
            q_target_value = self.sess.run( self.QValueT, feed_dict={self.stateInputT:s2_batch})
            # self.log_time('sess.run q_target_value ')

            y_i = []
            for k in range(self.batch_size):
                if d_batch[k]:
                    y_i.append(r_batch[k])
                else:
                    y_i.append(r_batch[k] + self.gamma * np.max(q_target_value[k] ))

            y_i = np.array(y_i)

            # self.log_time('y_i')
            # print('s_batch = ', s_batch)
            # print('I: s_batch.shape={}, type(s_batch)={}'.format(s_batch.shape, type(s_batch)))
            # print('a_batch = ', a_batch)
            # print('I: a_batch.shape={}, type(a_batch)={}'.format(a_batch.shape, type(a_batch)))
            
            # print('y_i = ', y_i)
            # print('I: y_i.shape={}, type(y_i)={}'.format(y_i.shape, type(y_i)))
            
            q_action, q_loss, _ = self.sess.run([self.q_action, self.q_loss,  self.train_op], feed_dict={
                self.yInput : y_i,
                self.actionInput : a_batch,
                self.stateInput : s_batch
            })

            self.ep_sum_q_action += np.mean(q_action)
            self.ep_sum_q_loss += q_loss
            self.ep_train_count += 1
            self.update_target_count+=1

            # print(f'train_count ={self.ep_train_count}')
            if self.update_target_count >= self.update_Q_target_times:
                # print('Update to Q target')
                self.sess.run(self.copy_Q_2_Qtarget)
                self.update_target_count = 0


        # self.log_time('train')

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
        self.q_action = tf.reduce_sum(tf.multiply(self.QValue, self.actionInput), reduction_indices = 1)
        self.q_loss = tf.reduce_mean(tf.square(self.yInput - self.q_action))
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.q_loss)

    def build_q_net(self):
        # state = tf.placeholder("float",[None, self.s_dim])
        # if np.isscalar(self.s_dim):
        #     print('isscalar')
        # else:
        #     print(f'self.s_dim.shape={self.s_dim.shape}')
        #     print(f'type(self.s_dim)={type(self.s_dim)}')
        #     print('not scalar')
        # print(f'self.s_dim={self.s_dim}')
        # s_dim = [self.s_dim] if np.isscalar(self.s_dim) else self.s_dim
        # print(f's_dim={s_dim}')
        # np_con =  np.concatenate( ([None], self.s_dim) )
        # print(f'np_con = {np_con}')
        state = tf.placeholder("float",  np.concatenate( ([None], self.s_dim) ))

        # print('self.s_dim = {}, state={}'.format(self.s_dim, state))
        nn = NNcomponent(self.cfg_network, state)
        q_value = FC(nn, self.a_dim, name_prefix = 'q_value', op='none', initializer = 'truncated_normal', bias_const=0.01)
        #q_value = FC(nn , self.a_dim, name_prefix =  'q_value', op='none', bias_const=0.01)
        # q_value = FC(nn , self.a_dim, name_prefix =  'q_value', op='tanh', initializer = 'truncated_normal', bias_const=0.03)
        # print('nn = ', nn)
        # print('q_value = ', q_value)
        return state, q_value #nn

    def notify_ep_done(self):
        self.ep_train_count = 0
        self.ep_sum_q_loss = 0
        self.ep_sum_q_action = 0

    def get_log_dic(self):
        avg_q_loss = self.ep_sum_q_loss /self.ep_train_count if self.ep_train_count!=0 else 0
        avg_q_action = self.ep_sum_q_action /self.ep_train_count if self.ep_train_count!=0 else 0
        log_dic = {'ep_avg_Q_loss':avg_q_loss,'ep_avg_Q_action':avg_q_action}
        return log_dic

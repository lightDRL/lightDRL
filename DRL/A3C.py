"""
Asynchronous Advantage Actor-Critic (A3C) 
Modify from: https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/experiments/Robot_arm/A3C.py
"""
import tensorflow as tf
import numpy as np
from DRL.Base import DRL
from component.simple_buffer import SimpleBuffer

class A3C(DRL):
    def __init__(self, cfg, model_log_dir, sess, scope):
        super(A3C, self).rl_init(cfg, model_log_dir)
        super(A3C, self).drl_init(sess)
        self.OPT_A = tf.train.RMSPropOptimizer(cfg['A3C']['LR_A'], name='RMSPropA')
        self.OPT_C = tf.train.RMSPropOptimizer(cfg['A3C']['LR_C'], name='RMSPropC')
        self.gamma = cfg['A3C']['gamma']

        self.simple_buf = SimpleBuffer(10)

        self.batch_size = cfg['A3C']['batch_size']

        # print('self.a_bound = ', self.a_bound)
        self.net_scope = scope
        if scope == cfg['A3C']['main_net_scope']:   # get global network
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, self.s_dim], 'state')
                self._build_net()
                # self.a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
                # self.c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
        else:   # local net, calculate losses
            
            main_net_scope = cfg['A3C']['main_net_scope']
            main_net_a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope= main_net_scope + '/actor')
            main_net_c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope= main_net_scope + '/critic')

            # print('[%s]'% scope, ' main_net_a_params = ', main_net_a_params)
            # print('[%s]'% scope, ' main_net_c_params = ', main_net_c_params)

            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, self.s_dim ], 'state')
                self.a_his = tf.placeholder(tf.float32, [None, self.a_dim], 'action')
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'v_target')

                if self.a_discrete:
                    self.a_prob, self.v = self._build_net()
                else:
                    mu, sigma, self.v = self._build_net()
                    mu, sigma = mu * self.a_bound, sigma + 1e-5
                    normal_dist = tf.contrib.distributions.Normal(mu, sigma)

                td = tf.subtract(self.v_target, self.v, name='TD_error')
                with tf.name_scope('c_loss'):
                    self.c_loss = tf.reduce_mean(tf.square(td))

                # with tf.name_scope('wrap_a_out'):
                #     self.test = sigma[0]
                #     mu, sigma = mu * self.a_bound, sigma + 1e-5

                with tf.name_scope('a_loss'):
                    if self.a_discrete:
                        argmax = tf.argmax(self.a_his, 1)
                        log_prob = tf.reduce_sum(tf.log(self.a_prob) * tf.one_hot(argmax, self.a_dim, dtype=tf.float32), axis=1, keep_dims=True)
                        # log_prob = tf.log(tf.reduce_max(self.a_his, 1, keep_dims=True)   )
                        entropy = -tf.reduce_sum(self.a_prob * tf.log(self.a_prob + 1e-5),
                                             axis=1, keep_dims=True)  # encourage exploration
                    else:    
                        log_prob = normal_dist.log_prob(self.a_his)
                        entropy = normal_dist.entropy()  # encourage exploration

                    # print('log_prob = ', log_prob)
                    
                    self.exp_v = log_prob * td
                    
                    self.exp_v = cfg['A3C']['ENTROPY_BETA'] * entropy + self.exp_v
                    self.a_loss = tf.reduce_mean(-self.exp_v)

                    # print('normal_dist',normal_dist)
                    # print('log_prob',log_prob)
                    # print('exp_v',exp_v)
                if not self.a_discrete:
                    with tf.name_scope('choose_a'):  # use local params to choose action
                        # self.A = tf.clip_by_value(tf.squeeze(normal_dist.sample(1), axis=0), *self.a_bound)
                        self.A = tf.clip_by_value(tf.squeeze(normal_dist.sample(1), axis=0), -self.a_bound, self.a_bound)
                    
                # with tf.name_scope('local_grad'):
                self.a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
                self.c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
                self.a_grads = tf.gradients(self.a_loss, self.a_params)
                self.c_grads = tf.gradients(self.c_loss, self.c_params)

                # print('self.a_loss = ', self.a_loss)
                # print('[%s]'% scope, 'self.a_params = ', self.a_params)
                # print('[%s]'% scope, 'self.a_grads = ', self.a_grads)
                # print('[%s]'% scope, 'self.c_params = ', self.c_params)
                # print('[%s]'% scope, 'self.c_grads = ', self.c_grads)


            with tf.name_scope('sync'):
                with tf.name_scope('pull'):
                    self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, main_net_a_params)]
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, main_net_c_params)]
                with tf.name_scope('push'):
                    self.update_a_op = self.OPT_A.apply_gradients(zip(self.a_grads, main_net_a_params))
                    self.update_c_op = self.OPT_C.apply_gradients(zip(self.c_grads, main_net_c_params))

            # print('----------tf.report_uninitialized_variables()----------')
            # print(self.sess.run(tf.report_uninitialized_variables()))
            # tf.variables_initializer(
            #     [v for v in tf.global_variables() if v.name.split(':')[0] in set(sess.run(tf.report_uninitialized_variables()))
            # ])
            # self.sess.run(tf.global_variables_initializer())

            # print('--again--------tf.report_uninitialized_variables()----------')

    def _build_net(self):
        w_init = tf.contrib.layers.xavier_initializer()
        # w_init = tf.random_normal_initializer(0., .1)
        with tf.variable_scope('actor'):
            l_a = tf.layers.dense(self.s, 200, tf.nn.relu6, kernel_initializer=w_init, name='la')
            #l_a = tf.layers.dense(l_a, 300, tf.nn.relu6, kernel_initializer=w_init, name='la2')
            if self.a_discrete:
                a_prob = tf.layers.dense(l_a, self.a_dim, tf.nn.softmax, kernel_initializer=w_init, name='ap')
            else:
                mu = tf.layers.dense(l_a, self.a_dim, tf.nn.tanh, kernel_initializer=w_init, name='mu')
                sigma = tf.layers.dense(l_a, self.a_dim, tf.nn.softplus, kernel_initializer=w_init, name='sigma')
        with tf.variable_scope('critic'):
            l_c = tf.layers.dense(self.s, 100, tf.nn.relu6, kernel_initializer=w_init, name='lc')
            #l_c = tf.layers.dense(l_c, 200, tf.nn.relu6, kernel_initializer=w_init, name='lc2')
            v = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name='v')  # state value
        
        if self.a_discrete:
            return a_prob, v
        else:
            return mu, sigma, v

    def update_global(self, feed_dict):  # run by a local
        # print('update_global')
        _, _ = self.sess.run([self.update_a_op, self.update_c_op], feed_dict)  # local grads applies to global net
        #return 

    def pull_global(self):  # run by a local
        # print('pull_global')
        self.sess.run([self.pull_a_params_op, self.pull_c_params_op])

    def choose_action(self, s):  # run by a local
        s = s[np.newaxis, :]
        if self.a_discrete:
            prob = self.sess.run(self.a_prob, feed_dict={self.s: s})[0]
            #a_prob = np.random.choice(len(prob),p=prob)
            return prob
        else:
            return self.sess.run(self.A, {self.s: s})[0]

    def add_data(self, s, a, r, d, s_):
        ''' self, states, actions, rewards, done, next_state''' 
        # print('---Before add_data----')
        # print('I: train get s.shape={}, type(s)={}'.format(np.shape(s), type(s)))
        # print('I: train get a.shape={}, type(a)={}, a = {}'.format(np.shape(a), type(a), a))
        # print('I: train get r.shape={}, type(r)={}'.format(np.shape(r), type(r)))
        # print('I: train get d.shape={}, type(d)={}'.format(np.shape(d), type(d)))
        # print('I: train get s_.shape={}, type(s_)={}'.format(np.shape(s_), type(s_)))
        # self.last_done = d
        self.simple_buf.add(s, a, r, d, s_)

        # if self.last_done:
        #     s_batch, a_batch, r_batch, d_batch, s2_batch = self.simple_buf.sample_batch(self.batch_size)
        #     done = d_batch[self.simple_buf.size() - 1]
        #     print('[{}] add_data() self.last_done={}, done={}'.format(self.net_scope, self.last_done, done ) )

    def train(self): #,  states, actions, rewards, next_state, done):
        done = self.simple_buf.get_last_done()
        if self.simple_buf.size() >= self.batch_size or done:
            s_batch, a_batch, r_batch, d_batch, s2_batch = self.simple_buf.sample_batch(self.batch_size)
            
            # print('[{}] self.simple_buf.size() ={}'.format( self.net_scope, self.simple_buf.size() ) )
            # print('[{}] self.simple_buf.size() ={}, len(d_batch)= {}, done = {}'.format(self.net_scope, self.simple_buf.size(), len(d_batch) , done) )
            last_s2 = s2_batch[len(s2_batch)-1]
            last_s2 = [last_s2] # same as last_s2[np.newaxis, :]
            if done:
                v_s_ = -5   # terminal
            else:
                #v_s_ = self.sess.run(self.v, {self.s: s2_batch[np.newaxis, :]})[0, 0]
                # run_one = self.sess.run(self.v, {self.s: last_s2})
                # print('run_one = ', run_one)
                v_s_ = self.sess.run(self.v, {self.s: last_s2})[0, 0]
            buffer_v_target = []

            # print('[%s]'% self.net_scope, 'v_s_ = ', v_s_)

            for r in r_batch[::-1]:    # reverse buffer r
                v_s_ = r + self.gamma * v_s_
                buffer_v_target.append(v_s_)
            buffer_v_target.reverse()
            
            buffer_s, buffer_a, buffer_v_target = np.vstack(s_batch), np.vstack(a_batch), np.vstack(buffer_v_target)
            # buffer_v_target = np.vstack(s_batch), np.vstack(a_batch), np.vstack(buffer_v_target)
            # print('s_batch = ', s_batch, ', s_batch.shape = ', s_batch.shape)
            # print('buffer_s = ', buffer_s, ', np.shape(buffer_s) = ', np.shape(buffer_s) )

            feed_dict = {
                self.s: s_batch, # buffer_s,
                self.a_his: a_batch,  #buffer_a,
                self.v_target: buffer_v_target,
            }

            self.update_global(feed_dict)
            self.pull_global()

            self.simple_buf.clear()

    def get_log_dic(self):
        return None

    def notify_ep_done(self):
        pass
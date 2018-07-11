"""
Deep Deterministic Policy Gradient (DDPG)
Modify from: https://github.com/liampetti/DDPG/blob/master/ddpg.py
"""
import tensorflow as tf
import numpy as np
from Base import DRL
from component.replay_memory import ReplayMemory
from component.DNN_v3 import *
from component.NNcomponent import NNcomponent

class DDPG(DRL):
    def __init__(self, cfg, model_log_dir="", sess = None):
        super(DDPG, self).rl_init(cfg, model_log_dir)
        super(DDPG, self).drl_init(sess)
        self.memory_capacity = cfg['DDPG']['memory_capacity']
        self.memory_train_min = cfg['DDPG']['memory_train_min']
        self.batch_size = cfg['DDPG']['batch_size']
        self.lr_actor = cfg['DDPG']['lr_actor']
        self.lr_critic = cfg['DDPG']['lr_critic']
        self.tau = cfg['DDPG']['tau']                    
        self.gamma = cfg['RL']['reward_gamma']
        # self.actor_NN = cfg['actor_NN']

        self.sess = sess

        self.actor = ActorNetwork(sess, self.s_dim, self.a_dim, self.a_bound, self.lr_actor, self.tau, cfg['actor_NN'])
        self.critic = CriticNetwork(sess, self.s_dim, self.a_dim,
                            self.lr_critic, self.tau, self.actor.get_num_trainable_vars(),  cfg['critic_state_NN'],  cfg['critic_action_NN'])

        # noise = Noise(DELTA, SIGMA, OU_A, OU_MU)
        # reward = Reward(REWARD_FACTOR, GAMMA)
        # replay memory 
        assert self.memory_capacity > self.memory_train_min, "Replay memory: memory_capacity < memory_train_min"
        self.mem = ReplayMemory(self.memory_capacity)

        self.sum_q_max = 0
        self.train_count = 0
        self.notify_ep_done()  # for reset variable

    def choose_action(self, s):
        # print('self.actor.s_dim = ', self.actor.s_dim, ', type = ', type(self.actor.s_dim))
        # print('choose_action() shape = ', np.shape(s), ', type = ', type(s),', s=', s)
        a = self.actor.predict(np.reshape(s, (1, self.actor.s_dim)))
        action = a[0]
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
            # print('DDPG in train')
            s_batch, a_batch, r_batch, d_batch, s2_batch = self.mem.sample_batch(self.batch_size)

            # Calculate targets
            q_target = self.critic.predict_target(s2_batch, self.actor.predict_target(s2_batch))

            
            y_i = []
            for k in range(self.batch_size):
                if d_batch[k]:
                    y_i.append(r_batch[k])
                else:
                    y_i.append(r_batch[k] + self.gamma * q_target[k])

            
            # print('shape(s_batch) = ', np.shape(s_batch) )
            # print('shape(a_batch) = ', np.shape(a_batch) )
            # print('shape(y_i) = ', np.shape(y_i) )
            # print(" np.reshape(y_i, (self.batch_size, 1)=",  np.reshape(y_i, (self.batch_size, 1)).shape)
            
            # Update the critic given the targets
            predicted_q_value, c_loss, _ = self.critic.train(s_batch, a_batch, np.reshape(y_i, (self.batch_size, 1)))

            # print('predicted_q_value=', predicted_q_value)
            # ep_ave_max_q += np.amax(predicted_q_value)

            # Update the actor policy using the sampled gradient
            a_outs = self.actor.predict(s_batch)
            grads = self.critic.action_gradients(s_batch, a_outs)
            self.actor.train(s_batch, grads[0])

            # Update target networks
            self.actor.update_target_network()
            self.critic.update_target_network()

            # return  np.amax(predicted_q_value)
            self.max_q =  np.amax(predicted_q_value)
            self.sum_q_max += np.amax(predicted_q_value)
            self.train_count += 1

            self.train_sum_critic_loss += c_loss

     # override
    def _build_net(self):
        pass

    def get_avg_q(self):
        return  self.sum_q_max /self.train_count if self.train_count!=0 else 0


    def notify_ep_done(self):
        self.train_count = 0
        self.sum_q_max = 0
        self.train_sum_critic_loss = 0

    def get_log_dic(self):
        log_dic = {'ep_avg_Q':self.get_avg_q() }
        log_dic['critic avg loss'] = self.train_sum_critic_loss /self.train_count if self.train_count!=0 else 0
        return log_dic

class ActorNetwork(object):
    def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau, actor_NN):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.tau = tau
        self.actor_NN = actor_NN

        # Actor Network
        with tf.variable_scope('actor'):
            self.inputs, self.out, self.scaled_out = self.create_actor_network()
        # self.inputs, self.out, self.scaled_out = self.create_actor_network('actor')

        self.network_params = tf.trainable_variables()

        # Target Network
        with tf.variable_scope('actor_target'):
            self.target_inputs, self.target_out, self.target_scaled_out = self.create_actor_network()
        #self.target_inputs, self.target_out, self.target_scaled_out = self.create_actor_network('actor_target')

        self.target_network_params = tf.trainable_variables()[len(self.network_params):]

        # Op for periodically updating target network with online network weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) + \
                                                  tf.multiply(self.target_network_params[i], 1. - self.tau))
             for i in range(len(self.target_network_params))]

        # This gradient will be provided by the critic network
        self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])

        # Combine the gradients here
        self.actor_gradients = tf.gradients(self.scaled_out, self.network_params, -self.action_gradient)

        # Optimization Op by applying gradient, variable pairs
        self.optimize = tf.train.AdamOptimizer(self.learning_rate). \
            apply_gradients(zip(self.actor_gradients, self.network_params))

        self.num_trainable_vars = len(self.network_params) + len(self.target_network_params)

    def create_actor_network(self):
        # print(' self.s_dim=',  self.s_dim, ' self.s_dim.shape=',  np.shape(self.s_dim) )
        # print('type( self.s_dim = ', type( self.s_dim),'self.s_dim = ', self.s_dim)
        # print('self.a_dim = ' , self.a_dim)
       
        inputs = tf.placeholder(tf.float32, [None, self.s_dim], name =  'state')
        # type 1
        # fc1 = FC(inputs, n_hidden_1, name_prefix = 'fc_1', op='relu', initializer = 'truncated_normal', bias_const=0.03)
        # fc2 = FC(fc1   , n_hidden_2, name_prefix = 'fc_2', op='relu', initializer = 'truncated_normal', bias_const=0.03)
        # out = FC(fc2   , self.a_dim, name_prefix = 'output', op='tanh', initializer = 'truncated_normal', bias_const=0.03)
        # type 2
        nn = NNcomponent(self.actor_NN, inputs)
        out = FC(nn, self.a_dim, name_prefix = 'output', op='tanh', initializer = 'truncated_normal', bias_const=0.03)

        scaled_out = tf.multiply(out, self.action_bound)  # Scale output to -action_bound to action_bound
        return inputs, out, scaled_out

    def train(self, inputs, a_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient
        })

    def predict(self, inputs):
        # out = self.sess.run(self.out, feed_dict={
        #     self.inputs: inputs
        # })
        # print('actor.out = ', out)

        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs: inputs
        })

    def predict_target(self, inputs):
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.target_inputs: inputs
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars



class CriticNetwork(object):
    """
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.
    """

    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, num_actor_vars, critic_state_NN, critic_action_NN):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau

        self.critic_state_NN  = critic_state_NN
        self.critic_action_NN = critic_action_NN

        # Create the critic network
        with tf.variable_scope('critic'):
            self.inputs, self.action, self.out = self.create_critic_network()
        # self.inputs, self.action, self.out = self.create_critic_network('critic')

        self.network_params = tf.trainable_variables()[num_actor_vars:]

        # Target Network
        with tf.variable_scope('critic_target'):
            self.target_inputs, self.target_action, self.target_out = self.create_critic_network()
        # self.target_inputs, self.target_action, self.target_out = self.create_critic_network('critic_target')

        self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]

        # Op for periodically updating target network with online network weights with regularization
        self.update_target_network_params = \
            [self.target_network_params[i].assign(
                tf.multiply(self.network_params[i], self.tau) + tf.multiply(self.target_network_params[i], 1. - self.tau))
             for i in range(len(self.target_network_params))]

        # Network target (y_i)
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        # Define loss and optimization Op
        self.loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.predicted_q_value, self.out))))
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        # Get the gradient of the net w.r.t. the action
        self.action_grads = tf.gradients(self.out, self.action)

    def create_critic_network(self):
        
        state = tf.placeholder(tf.float32, [None, self.s_dim], name="state")
        action = tf.placeholder(tf.float32, [None, self.a_dim], name= "action")

        # n_hidden_1 = 200
        # n_hidden_2 = 100
        # type 1
        # # state -> fc1
        # fc1 = FC(state, n_hidden_1, name_prefix = critic_name + '_fc_1', op='relu', initializer = 'truncated_normal', bias_const=0.03)
        # # fc1*w2 +  fc_action (action-> fc_action)
        # w2 = weight_variable([n_hidden_1, n_hidden_2] , initializer='truncated_normal',normal_mean=0.0, normal_stddev=0.01, name=critic_name + '_w2')
        # fc_action = FC(action, n_hidden_2, name_prefix = critic_name + '_fc_action', op='none', initializer = 'truncated_normal', bias_const=0.03)
        # add_fc_s_a = tf.nn.relu(tf.matmul(fc1, w2) + fc_action)

        # type 2
        # state -> fc1
        # state_fc1 = FC(state, n_hidden_1, name_prefix = 'fc_1', op='relu', initializer = 'truncated_normal', bias_const=0.03)
        # state_fc2 = FC(state_fc1, n_hidden_2, name_prefix =  'fc_2', op='relu', initializer = 'truncated_normal',normal_mean=0.0, normal_stddev=0.01,  bias_const=0.0)
        # # fc1*w2 +  fc_action (action-> fc_action)
        # # w2 = weight_variable([n_hidden_1, n_hidden_2] , initializer='truncated_normal',normal_mean=0.0, normal_stddev=0.01, name=critic_name + '_w2')
        # action_fc = FC(action, n_hidden_2, name_prefix ='fc_action', op='none', initializer = 'truncated_normal', bias_const=0.03)
        # add_fc_s_a = tf.nn.relu(state_fc2 + action_fc)

        #type 3
        state_nn = NNcomponent(self.critic_state_NN, state)
        action_nn = NNcomponent(self.critic_action_NN, action)
        add_fc_s_a = tf.nn.relu(state_nn + action_nn)

        out = FC(add_fc_s_a, 1, name_prefix =  'out', op='none', initializer = 'truncated_normal', bias_const=0.03)
        
        return state, action, out
    
    def train(self, inputs, action, predicted_q_value):
        # return self.sess.run([self.out, self.optimize], feed_dict={
        return self.sess.run([self.out, self.loss, self.optimize], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.predicted_q_value: predicted_q_value
        })

    def predict(self, inputs, action):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def predict_target(self, inputs, action):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action
        })

    def action_gradients(self, inputs, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)




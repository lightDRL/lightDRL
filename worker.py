
import tensorflow as tf
import numpy as np
import yaml
import os
import time

# for connect to server
# from flask_socketio import Namespace, emit
# DRL import 
from DRL.Base import RL, DRL
from DRL.DDPG import DDPG
from DRL.DQN import DQN
from DRL.A3C import A3C
from DRL.Qlearning import Qlearning
from DRL.component.reward import Reward
from DRL.component.noise import Noise
# for debug 
from DRL.component.utils import print_tf_var

class WorkerBase(object):
    def base_init(self, cfg, graph, sess, model_log_dir, net_scope = None):
        self.graph = graph
        # RL or DRL Init
        np.random.seed( cfg['misc']['random_seed'])
        #----------------------------setup RL method--------------------------#
        method_class = globals()[cfg['RL']['method'] ]
        self.method_class = method_class

        #--------------setup var---------------#
        self.model_log_dir = model_log_dir
        self.var_init(cfg)

        if issubclass(method_class, DRL):
            '''Use DRL'''
            self.sess = sess
            # with tf.variable_scope(client_id): 
            with self.graph.as_default():
                if cfg['RL']['method']=='A3C':  # for multiple worker
                    self.RL = method_class(cfg, model_log_dir, self.sess, net_scope)
                else:
                    self.RL = method_class(cfg, model_log_dir, self.sess )
                    #self.RL.init_or_restore_model(self.sess)    # init or check model
                    self.init_or_restore_model(self.sess)
                # print_tf_var('worker after init DRL method')
        elif issubclass(method_class, RL):
            '''Use RL'''
            self.RL = method_class(cfg, model_log_dir, None)
            pass
        else:
            print('E: Worker::__init__() say error method name={}'.format(cfg['RL']['method'] ))

        self.tf_writer = tf.summary.FileWriter(model_log_dir)
        

    def var_init(self, cfg):
        self.frame_count = 0
        
        self.state_buf  = []
        self.action_buf = []
        self.reward_buf = []
        self.done_buf = []
        self.next_state_buf  = []

        self.start_time = time.time()
        self.ep_s_time = time.time()
        self.train_s_time = self.ep_s_time
        self.ep = 1#0
        self.ep_use_step = 0
        self.ep_reward = 0
        self.ep_max_reward = -9999
        self.all_step = 0
        self.all_ep_reward = 0   # sum of all ep reward
        self._is_max_ep =  False
        self.max_ep = cfg['misc']['max_ep']
        self.ep_max_step =  cfg['misc']['ep_max_step']                 # default: None -> unlimit steps, max steps in one episode
        self.a_discrete = cfg['RL']['action_discrete'] 
        self.train_multi_steps = cfg['RL']['train_multi_steps']        # default: 1, param: 'if_down', 'steps(int)'-> train if down or train after steps
        self.add_data_steps = cfg['RL']['add_data_steps']              # default: 1, param: 'if_down', 'steps(int)'-> add data if down or add data after steps    

        if type(self.train_multi_steps) == int:
            self.train_multi_count =  0

        if type(self.add_data_steps) == int:
            self.add_data_count =  0

        

        # setup reward
        if cfg['RL']['reward_reverse'] == True:
            self.reward_reverse = cfg['RL']['reward_reverse']
            self.reward_process = Reward(cfg['RL']['reward_factor'], cfg['RL']['reward_gamma'])
        # self.one_step_buf = {'s':None, 'a':None, 'r':None, 'd':None, 's_':None} 


        self.exploration_step = cfg['RL']['exploration']
        self.exploration_action_noise = cfg['RL']['exploration_action_noise']

        self.action_epsilon = cfg['RL']['action_epsilon']
        self.action_epsilon_add = cfg['RL']['action_epsilon_add']
        self.action_noise = cfg['RL']['action_noise']
        # setup noise
        if self.action_noise!=None:
            if self.action_noise =='epsilon-greedy':
                self.epsilon_greedy_value = cfg['epsilon-greedy']['value']
                self.epsilon_greedy_discount = cfg['epsilon-greedy']['discount']
            elif self.action_noise =='Uhlenbeck':
                self.noise = None
                # print('--------in cofigure noise-----')
                self.noise = Noise( cfg['Uhlenbeck']['delta'], cfg['Uhlenbeck']['sigma'], cfg['Uhlenbeck']['ou_a'], cfg['Uhlenbeck']['ou_mu'])
                self.noise_max_ep= cfg['Uhlenbeck']['max_ep']
                self.ou_level = 0
    # def tf_summary_init(self, model_log_dir )
    #     self.tf_writer = tf.summary.FileWriter(model_log_dir)

        self.none_over_pos_count = 0 

        self.worker_nickname = cfg['misc']['worker_nickname']

        # about model save
        self.model_save_cycle = cfg['misc']['model_save_cycle'] if ('misc' in cfg) and ('model_save_cycle' in cfg['misc']) else None
        self.model_retrain = cfg['misc']['model_retrain'] #if ('misc' in cfg) and ('model_retrain' in cfg['misc']) else True
        print('[I] worker_nickname = ' +  self.worker_nickname + 'ready')

    def train_process(self, data):
        
        state = data['s']
        action = data['a']
        reward = data['r']
        done = data['d']
        next_state = data['s_']

        self.all_step += 1
        self.ep_use_step += 1
        self.ep_reward += reward
        self.ep_max_reward = reward if reward > self.ep_max_reward else self.ep_max_reward

        if np.isscalar(state):
            state = np.array([state])

        if np.isscalar(next_state):
            next_state = np.array([next_state])

        # print('-------in train_process-------')
        # print('I: train get state.shape={}, type(state)={}'.format(np.shape(state), type(state)))
        # print('I: train get action.shape={}, type(action)={}'.format(np.shape(action), type(action)))
        # print('I: train get reward.shape={}, type(reward)={}'.format(np.shape(reward), type(reward)))
        # print('I: train get done = {}'.format(done)) 
        train_done = done
        if done == True and self.ep_use_step >= self.ep_max_step:
            # if the env has end position which could get reward, try to get the end position (like maze) as the done
            # else like cartpole which is no end position, so it use the default done
            train_done = False 

        # print('done = {}, train_done={}'.format(done, train_done))
        self.train_add_data(state, action, reward, train_done, next_state )

        if self.all_step > self.exploration_step:
            if self.train_multi_steps == 1:     # usually do this
                self.train()
            elif type(self.train_multi_steps) == int:
                self.train_multi_count += 1
                if self.train_multi_count >= self.train_multi_steps:
                    self.train()
                    self.train_multi_count = 0
            elif self.train_multi_steps=='if_down' and done:
                self.train()
            else:
                assert False, 'Error train_multi_steps'


        if done:
            ep_time_str = self.time_str(self.ep_s_time,min=True)
            all_time_str = self.time_str(self.train_s_time)
            more_log = ''
            if self.action_noise =='epsilon-greedy':
                more_log += ' | epsilon: %5.4f' % self.epsilon_greedy_value
            # print('more_log = ' ,more_log)
            log_str = '(%s) EP%5d | EP_Step: %5d | EP_Reward: %8.2f | MAX_R: %4.2f %s | EP_Time: %s | All_Time: %s ' % \
                    (self.worker_nickname, self.ep, self.ep_use_step, self.ep_reward, self.ep_max_reward, more_log, ep_time_str, all_time_str )
            print(log_str)
            # if issubclass(self.method_class, DRL):
            #      log_str = '%s| Avg_Q: %.4f' % (log_str, self.RL.get_avg_q())
            log_dict = self.RL.get_log_dic()
            # if self.action_epsilon !=None:
            #     log_dict = dict() if log_dict == None else log_dict
            #     log_dict['epsilon'] = self.action_epsilon

            if self.action_noise =='epsilon-greedy':
                log_dict = dict() if log_dict == None else log_dict
                log_dict['epsilon'] = self.epsilon_greedy_value
            # ep_time = time.time() - self.ep_s_time
            # log_dict['EP Time'] = ep_time
            
            if self.action_epsilon!=None and self.ep_max_reward > 0:
                self.action_epsilon += self.action_epsilon_add

            
            self.all_ep_reward+= self.ep_reward
            self.tf_summary(self.ep , self.ep_reward,self.ep_max_reward,self.ep_use_step, self.ep_s_time, log_dict)
            
            self.RL.notify_ep_done()
            self.ep_use_step = 0
            self.ep_reward = 0
            self.ep_max_reward = -9999
            
            if self.ep >= self.max_ep:
                # summary result
                avg_ep_reward = self.all_ep_reward / float(self.ep)
                avg_ep_step   = self.all_step   / float(self.ep) 
                avg_ep_reward_str =  'average ep reward = ' + str(avg_ep_reward)
                avg_ep_step_str =    'average ep step = ' + str(avg_ep_step)
                self.tf_summary_text('EP Average', avg_ep_reward_str, self.ep)
                self.tf_summary_text('EP Average', avg_ep_step_str, self.ep)
                self.tf_writer.flush()
                self._is_max_ep = True
            # else:
            self.ep+=1
            self.ep_s_time = time.time()  # update episode start time

            if self.ep % self.model_save_cycle==0 and issubclass(self.method_class, DRL):
                if not self.method_class == A3C:
                    self.save_model(self.model_log_dir, self.ep)
               
    def tf_summary_text(self, tag, text, ep):
        text_tensor = tf.make_tensor_proto(text, dtype=tf.string)
        meta = tf.SummaryMetadata()
        meta.plugin_data.plugin_name = "text"
        summary = tf.Summary()
        summary.value.add(tag=tag, metadata=meta, tensor=text_tensor)
        self.tf_writer.add_summary(summary, ep)


    def tf_summary(self, ep, ep_r, ep_max_r, ep_use_step, ep_s_time, log_dict):
        summary = tf.Summary()
        summary.value.add(tag='EP Reward', simple_value=int(ep_r))
        summary.value.add(tag='EP Max_Reward', simple_value=int(ep_max_r))
        summary.value.add(tag='EP Use Steps', simple_value=int(ep_use_step))
        summary.value.add(tag='EP Time', simple_value=float(time.time() - ep_s_time))
        summary.value.add(tag='All Time', simple_value=float(time.time() - self.train_s_time))
        if log_dict != None:
            for key, value in log_dict.items(): #iteritems():
                # print('{} -> {} '.format(key, value))
                summary.value.add(tag=key, simple_value=float(value))

        # summary.value.add(tag='Perf/Qmax', simple_value=float(ep_ave_max_q / float(j)))
        self.tf_writer.add_summary(summary, ep)


    def train_add_data(self, state, action, reward, done, next_state ):
        # print('self.add_data_steps  =', self.add_data_steps )
        if self.add_data_steps == 1:
            # print('in add_data_steps ==1')
            self.RL.add_data( state, action, reward, done, next_state)
        else:
            # print('USE multi_add')
            go_multi_add = False
            self.state_buf.append(state)
            self.action_buf.append(action)
            self.reward_buf.append(reward)
            self.done_buf.append(done)
            self.next_state_buf.append(next_state)
            if type(self.add_data_steps) == int:
                self.add_data_count += 1
                if self.add_data_count >= self.add_data_steps:
                    go_multi_add = True
                    self.add_data_count = 0
            elif self.add_data_steps=='if_down' and done:
                go_multi_add = True

            # print(' self.add_data_steps = ',  self.add_data_steps,', done = ', done)


            if go_multi_add:
                tmp_reward_buf = self.reward_buf
                if self.reward_reverse:
                    # print('before reward process, len=', len(self.reward_buf), ',data =', self.reward_buf )
                    tmp_reward_buf = self.reward_process.discount(self.reward_buf)
                    # print('afeter reward process, len=', len(tmp_reward_buf), ',data =', tmp_reward_buf )

                states = np.array(self.state_buf) 
                actions = np.array(self.action_buf) 
                rewards = np.array(tmp_reward_buf) 
                dones = np.array(self.done_buf) 
                next_states = np.array(self.next_state_buf) 

                self.RL.add_data(states, actions, rewards, dones, next_states, add_multiple = True)

                self.state_buf  = []
                self.action_buf = []
                self.reward_buf = []
                self.done_buf = []
                self.next_state_buf  = []

    def predict(self, state):
        state = np.array(state)
        a =  self.RL.choose_action(state)

        return a

    def add_action_noise(self, a, reward = None):
        # print('--------------%03d-%03d----------------' % ( self.ep, self.ep_use_step))
        # print('before a = ', a)
        if self.a_discrete==True:
            a_dim = self.RL.a_discrete_n
            if self.action_noise=='epsilon-greedy':
                if np.random.rand() < self.epsilon_greedy_value:
                    # print('in random choose')
                    a = np.zeros(a_dim)
                    a[ np.random.randint(self.RL.a_discrete_n) ] =1
                    # print('self.RL.a_discrete_n = ', self.RL.a_discrete_n)
                # print('self.epsilon_greedy_discount = ', self.epsilon_greedy_discount)
                if self.epsilon_greedy_discount > 0.0:
                # if self.epsilon_greedy_discount > 0.0 and reward > 0:  # maybe happen in done, and not return
                    self.epsilon_greedy_value -= self.epsilon_greedy_discount


        if self.action_noise=='Uhlenbeck' and self.ep < self.noise_max_ep:
            self.ou_level = self.noise.ornstein_uhlenbeck_level(self.ou_level)
            a = a + self.ou_level
            # print('in ornstein_uhlenbeck ou_level = ' , self.ou_level)
        # print('after a = ', a)
        return a
        

    def train(self):
        if self.method_class == Qlearning:
            return
        
        with self.graph.as_default():
            # with self.sess.as_default():
            self.RL.train()

    def to_py_native(self, obj):
        if type(obj) == np.ndarray:
            return obj.tolist()
        if isinstance(obj, np.generic):
            return np.asscalar(obj)

    def time_str(self, start_time, min=False):
        use_secs = time.time() - start_time
        if min:
            return '%3dm%2ds' % (use_secs/60, use_secs % 60 )
        return  '%3dh%2dm%2ds' % (use_secs/3600, (use_secs%3600)/60, use_secs % 60 )

    
    def avg_ep_reward_show(self):
        avg_ep_reward = float(self.all_ep_reward) / float(self.ep-1)
        print('(%s) EP%5d | all_ep_reward: %lf | avg_ep_reward: %lf' % \
                    (self.worker_nickname, self.ep-1, self.all_ep_reward, avg_ep_reward) )
        return avg_ep_reward

    @property
    def is_max_ep(self):
        return self._is_max_ep
        #return (self.ep >= (self.max_ep) )

    def log_time(self, s):
        
        if hasattr(self, 'ts'):
            print(s + ' use time = ' + str( time.time() - self.ts  ))
        self.ts = time.time()

    # -----------------model save or restore------------------#

    def save_model(self, save_model_dir, g_step):
        with self.graph.as_default():
            saver = tf.train.Saver()
            save_path = os.path.join(self.model_log_dir , 'model.ckpt') 
            save_ret = saver.save(self.sess, save_path, global_step = g_step)

            print('Save Model to ' + save_ret  + ' !')


    def init_or_restore_model(self, sess,  model_dir = None ):
        if model_dir == None:
            model_dir = self.model_log_dir
        assert model_dir != None, 'init_or_restore_model model_dir = None'
        model_file = tf.train.latest_checkpoint(model_dir)
        print(f'model_file = {model_file}')
        start_ep = 1
        if model_file is None or self.model_retrain:
            print('[I] Initialize all variables')
            sess.run(tf.global_variables_initializer())
            print('[I] Initialize all variables Finish')
        else:
            #-------restore------#e 
            ind1 = model_file.index('model.ckpt')
            start_ep = int(model_file[ind1 +len('model.ckpt-'):]) + 1
            self.ep = start_ep
            saver = tf.train.Saver()
            saver.restore(sess, model_file)

            print('[I] Use model_file = ' + str(model_file) + ' ! Train from epoch = %d' % start_ep )
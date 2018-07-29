
import tensorflow as tf
import numpy as np
import yaml
import threading
import time

# for connect to server
from flask_socketio import Namespace, emit
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
        if issubclass(method_class, DRL):
            '''Use DRL'''
            self.sess = sess
            # with tf.variable_scope(client_id): 
            with self.graph.as_default():
                if cfg['RL']['method']=='A3C':  # for multiple worker
                    self.RL = method_class(cfg, model_log_dir, self.sess, net_scope)
                else:
                    self.RL = method_class(cfg, model_log_dir, self.sess )
                    self.RL.init_or_restore_model(self.sess)    # init or check model

                # print_tf_var('worker after init DRL method')
        elif issubclass(method_class, RL):
            '''Use RL'''
            self.RL = method_class(cfg, model_log_dir, None)
            pass
        else:
            print('E: Worker::__init__() say error method name={}'.format(cfg['RL']['method'] ))

        # print("({}) Worker Ready!".format(self.client_id))
        
        
        #--------------setup var---------------#
        self.var_init(cfg)

        # tf_summary_init
        print('model_log_dir = ', model_log_dir)
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
        self.ep_max_reward = 0
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
        print('worker_nickname = ', self.worker_nickname)

    def train_process(self, data):
        
        state = data['state']
        action = data['action']
        reward = data['reward']
        done = data['done']
        next_state = data['next_state']

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
            self.ep_max_reward = -99999
            
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
            for key, value in log_dict.iteritems():
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

                self.RL.add_data(states, actions, rewards, dones, next_states)
                # if self.ep_max_reward > 0:
                #     self.RL.add_data(states, actions, rewards, dones, next_states)
                #     self.none_over_pos_count = 0 
                # else:
                #     self.none_over_pos_count += 1
                #     if self.none_over_pos_count <= 2:
                #         self.RL.add_data(states, actions, rewards, dones, next_states)

                # print('self.none_over_pos_count = ', self.none_over_pos_count)

                self.state_buf  = []
                self.action_buf = []
                self.reward_buf = []
                self.done_buf = []
                self.next_state_buf  = []

    def predict(self, state):
        # print('--------------%03d-%03d----------------' % ( self.ep, self.ep_use_step))
        # print("I: predict() state.shape: {}, type(state)= {}, state={} ".format(np.shape(state), type(state), state) )
        state = np.array(state)
        a =  self.RL.choose_action(state)
       
        # if self.noise!=None and self.ep < self.noise_max_ep:
        #     self.ou_level = self.noise.ornstein_uhlenbeck_level(self.ou_level)
        #     a = a + self.ou_level
            # print('ou_level = ' , self.ou_level)
        # action = a[0]

        # print('send action = ', action)

        return a

    def add_action_noise(self, a):
        # print('--------------%03d-%03d----------------' % ( self.ep, self.ep_use_step))
        # print('before a = ', a)
        if self.a_discrete==True:
            a_dim = self.RL.a_discrete_n
            if self.action_noise=='epsilon-greedy':
                if np.random.rand() < self.epsilon_greedy_value:
                    a = np.zeros(a_dim)
                    a[ np.random.randint(self.RL.a_discrete_n) ] =1
                    self.epsilon_greedy_value -= self.epsilon_greedy_discount


        if self.action_noise=='Uhlenbeck' and self.ep < self.noise_max_ep:
            self.ou_level = self.noise.ornstein_uhlenbeck_level(self.ou_level)
            a = a + self.ou_level
            # print('in ornstein_uhlenbeck ou_level = ' , self.ou_level)
        # print('after a = ', a)
        return a
        '''
        if self.noise!=None and self.ep < self.noise_max_ep:
            self.ou_level = self.noise.ornstein_uhlenbeck_level(self.ou_level)
            a = a + self.ou_level
            # print('in ornstein_uhlenbeck ou_level = ' , self.ou_level)
        
        a_dim = a.shape[0]
        #------replace original action-----#
        if self.all_step < self.exploration_step:
            if self.action_discrete==True:
                
                if self.exploration_action_noise=='np_random':
                    # print('in np_random')
                    a = np.zeros(a_dim)
                    a[ np.random.randint(a_dim) ] =1
                elif self.exploration_action_noise=='dirichlet':
                    # print('in dirichlet')
                    a = np.random.dirichlet(np.ones(a_dim) )       # random, and sum is 1 
            
        
        if  self.action_epsilon!=None and self.all_step > self.exploration_step:
            if np.random.rand(1)[0] > self.action_epsilon:
                a = np.zeros(a_dim)
                a[ np.random.randint(a_dim) ] =1

                
        # if self.ep_max_reward > 0: 
            # print('iniiniin  self.ep_max_reward > 0')
        
        # print('self.action_epsilon = ' , self.action_epsilon)
        # print('after a=', a )
        # print('worker use action = ', np.argmax(a))
        '''
        

    def train(self):
        with self.graph.as_default():
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
        print('(%s) EP%5d | all_ep_reward: %lf ' % \
                    (self.worker_nickname, self.ep-1, self.all_ep_reward) )
        return float(self.all_ep_reward) / float(self.ep)

    @property
    def is_max_ep(self):
        return self._is_max_ep
        #return (self.ep >= (self.max_ep) )

class WorkerStandalone(WorkerBase):
    def __init__(self, cfg = None, model_log_dir = None, 
                        graph = None, sess = None, net_scope = None):

        # self.main_queue = main_queue
        self.lock = threading.Lock()
        self.client_id = "standalone"

        self.base_init(cfg, graph, sess, model_log_dir, net_scope)

        import Queue
        self.main_queue = Queue.Queue()
        
    def get_callback_queue(self):
        return self.main_queue 


    def on_predict(self, data):
         
        self.lock.acquire()
        
        action = self.predict(data['state'])
        if self.action_noise != None:
            action = self.add_action_noise(action)
        self.main_queue.put(action)
        self.lock.release()
    
    def on_train_and_predict(self, data):
        self.lock.acquire()
        self.train_process(data)
        self.lock.release()

        # if not data['done']:
        #     action = self.predict(data['next_state'])
        #     action = self.add_action_noise(action)
        #     # print('worker on_train_and_predict action = ' , action,', thread=' ,threading.current_thread().name )
        #     self.main_queue.put(action)

        action = self.predict(data['next_state'])
        action = self.add_action_noise(action)
        # print('worker on_train_and_predict action = ' , action,', thread=' ,threading.current_thread().name )
        # Becareful here !!!
        if not data['done']:
            self.main_queue.put(action)
        else:
            self.main_queue.put('WORKER_GET_DONE')

        


class WorkerConn(WorkerBase, Namespace):  # if you want to standalone, you could use  Worker(object)

    def __init__(self, ns = "", client_id="", cfg = None, model_log_dir = None, 
                                graph = None, sess = None, net_scope = None):

        super(WorkerConn, self).__init__(ns)
        self.client_id = client_id
            
        self.base_init(cfg, graph, sess, model_log_dir, net_scope)

    def on_connect(self):
        print('{} Worker Connect'.format(self.client_id))

    def on_disconnect(self):
        print('{} Worker Disconnect'.format(self.client_id))
        

    def on_predict(self, data):
        action = self.predict(data['state'])
        # print('worker on_predict action = ' , action,', type=', type(action))
        # for socketio_client (if dont use, error: is not JSON serializable)
        action = action.tolist() if type(action) == np.ndarray else action
        # print('worker on_predict action after = ' , action,', type=', type(action))
        
        emit('predict_response', action)

    

    def on_train_and_predict(self, data):
        self.train_process(data)
        if not data['done']:
            action = self.predict(data['next_state'])
            action = action.tolist() if type(action) == np.ndarray else action
            emit('predict_response', action)

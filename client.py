
import sys, os, time
import numpy as np
import json
from config import cfg
from threading import Thread
import signal                   # for ctrl+C
import sys                      # for ctrl+C
from socketIO_client import SocketIO, BaseNamespace

# import six
# from abc import ABCMeta,abstractmethod
# from socketIO_client import SocketIO, BaseNamespace

class EnvBase(object):
    def envbase_init(self):
        self.start_time = time.time()
        self.ep_s_time = time.time()
        self.ep = 0
        self.ep_use_step = 0
        self.ep_reward = 0

        self.env_name =''
    
    def on_predict_response(self, callback_action):
        self.server_action =  callback_action
        
        # print('server_action: {},type ={}, shape={}'.format(callback_action, type(callback_action), np.shape(callback_action)))
        action  = np.argmax(callback_action) if  cfg['RL']['action_discrete'] else callback_action
        # print('use action: ', action)
        # action = self.to_py_native(action)
        self.on_action_response(action)

    def send_state_get_action(self, state):
        state       = state.tolist() if type(state) != list else state
        dic ={'state': state}

        self.emit('predict',dic)
        
    
    def send_train_get_action(self, state, action, reward, done,next_state):
        self.ep_use_step += 1
        self.ep_reward += reward
        state      = state.tolist() if type(state) == np.ndarray else state
        # sever_action = self.server_action.tolist() if type(self.server_action) == np.ndarray else self.server_action
        next_state = next_state.tolist() if type(next_state) == np.ndarray else next_state

        dic ={'state': state, 
                        'action': self.server_action, #action, 
                        'reward': reward, 
                        'done'  : done,
                        'next_state': next_state}
        self.emit('train_and_predict',dic)

        if done:
            # self.log()
            self.ep+=1
            self.ep_use_step = 0
            self.ep_reward = 0
            self.ep_s_time = time.time()  # update episode start time
    
    def log(self):
        print('EP:%5d, STEP:%4d, r: %7.2f, ep_t:%s, all_t:%s' % \
            ( self.ep,  self.ep_use_step, self.ep_reward, self.time_str(self.ep_s_time, True), self.time_str(self.start_time)))
        
    def set_name(self,name):
        # print('set_name  = ', name)
        self.env_name = name

    def time_str(self, start_time, min=False):
        use_secs = time.time() - start_time
        if min:
            return '%3dm%2ds' % (use_secs/60, use_secs % 60 )
        return  '%3dh%2dm%2ds' % (use_secs/3600, (use_secs%3600)/60, use_secs % 60 )

    # @abstractmethod
    # def emit(self, cmd, dic):
    #     pass   


#--------------Alread have id, connect again -------------#
class EnvSpace(EnvBase, BaseNamespace):

    def on_connect(self):
        print('EnvSpace say connect')
        self.envbase_init()
        self.env_init()
       

    def on_disconnect(self):
        print('{} env say disconnect'.format(self.env_name))

        
    # def env_init(self):
    #     pass


class Client:
    def __init__(self, target_env_class, project_name=None,i_cfg = None, retrain_model = False):
        # Thread.__init__(self)
        self.target_env_class = target_env_class
        self.env_name = project_name
        self.socketIO = SocketIO('127.0.0.1', 5000)
        self.socketIO.on('connect', self.on_connect)
        self.socketIO.on('disconnect', self.on_disconnect)
        self.socketIO.on('reconnect', self.on_reconnect)
        self.socketIO.on('session_response', self.on_session_response)

        # for ctrl+C
        signal.signal(signal.SIGINT, self.signal_handler)

        send_cfg = cfg if i_cfg == None else cfg
        #self.socketIO.emit('session', project_name, cfg)  
        self.socketIO.emit('session', project_name, send_cfg, retrain_model)  
        self.socketIO.wait()
        

    def signal_handler(self, signal, frame):
        #print(signal)
        print('You pressed Ctrl+C!')
        self.target_env_class.close()
        self.socketIO.disconnect()
        
        sys.exit(0)

    def on_connect(self):
        print('[I] Client connect')

    def on_reconnect(self):
        print('[I] Client reconnect')

    def on_disconnect(self):
        print('[I] Client disconnect')

    def on_session_response(self, new_id):
        print('[I] Get id = {}'.format(new_id ))
        new_ns = '/' + str(new_id)  + '/rl_session'
        self.connect_with_ns(new_ns)

    def connect_with_ns(self,ns):
        # print('get ns ={}'.format(ns))
        new_env = self.socketIO.define(self.target_env_class, ns)
        new_env.set_name(self.env_name)
        # method_to_call = getattr(new_env, self.env_call_fun)
        # result = method_to_call()

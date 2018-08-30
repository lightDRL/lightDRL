
import sys, os, time
import numpy as np
import json
# from config import cfg
from threading import Thread
import signal                   # for ctrl+C
import sys                      # for ctrl+C
from socketIO_client_kbe import SocketIO, BaseNamespace

class EnvBase(object):
    def envbase_init(self):
        self.start_time = time.time()
        self.ep_s_time = time.time()
        self.ep = 1#0
        self.ep_use_step = 0
        self.ep_reward = 0

        self.env_name =''
        
    def set_cfg(self, cfg):
        self.cfg = cfg
    
    def on_predict_response(self, callback_action):
        self.server_action =  callback_action
        
        # print('server_action: {},type ={}, shape={}'.format(callback_action, type(callback_action), np.shape(callback_action)))
        action  = np.argmax(callback_action) if  self.cfg['RL']['action_discrete'] else callback_action
        # print('use action: ', action)
        # action = self.to_py_native(action)
        self.on_action_response(action)

    def send_state_get_action(self, state):
        state       = state.tolist()  if type(state) == np.ndarray else state # if type(state) != list else state
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

        self.log_time('[H] client emit ')
        self.emit('train_and_predict',dic)
        self.log_time('after client emit ')

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

    def set_socketIO(self, i_socketIO):
        self.socketIO = i_socketIO

    def emit(self, cmd, dic):
        self.socketIO.emit(cmd, dic)
        


class Client:
    def __init__(self, target_env_class, i_cfg , project_name=None, retrain_model = False):
        # Thread.__init__(self)
        
        self.env_name = project_name
        self.socketIO = SocketIO('127.0.0.1', 5000)
        self.socketIO.on('connect', self.on_connect)
        self.socketIO.on('disconnect', self.on_disconnect)
        self.socketIO.on('reconnect', self.on_reconnect)
        self.socketIO.on('predict_response', self.on_predict_response)
        
        # for ctrl+C
        signal.signal(signal.SIGINT, self.signal_handler)

        # send_cfg = cfg if i_cfg == None else cfg
        
        self.send_cfg = i_cfg
        #self.target_env_class = target_env_class
        self.target_env = target_env_class()
        self.target_env.set_cfg(self.send_cfg)
        self.target_env.set_name(self.env_name)

        self.target_env.set_socketIO(self.socketIO)
        self.target_env.envbase_init()
        self.target_env.env_init()

        print('[I] Client Init finish')
        self.socketIO.wait()

    def on_connect(self):
        print('[I] Client connect')
        # self.target_env.envbase_init()
        # self.target_env.env_init()

    def on_reconnect(self):
        print('[I] Client reconnect')

    def on_disconnect(self):
        print('[I] Client disconnect')

    def on_predict_response(self, callback_action):
        self.target_env.on_predict_response(callback_action)

    def signal_handler(self, signal, frame):
        #print(signal)
        print('You pressed Ctrl+C!')
        self.target_env.close()
        self.socketIO.disconnect()
        
        sys.exit(0)
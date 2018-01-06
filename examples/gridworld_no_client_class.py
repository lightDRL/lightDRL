
# Run With SARSA
#   python server.py config/gridworld_SARSA.yaml
#   python examples/gridworld_no_client_class.py config/gridworld_SARSA.yaml
# Run with QLearning
#   python server.py config/gridworld_QLearning.yaml
#   python examples/gridworld_no_client_class.py config/gridworld_QLearning.yaml
# Run With A3C
#   python server.py config/gridworld_A3C.yaml
#   python examples/gridworld_no_client_class.py config/gridworld_A3C.yaml



from socketIO_client import SocketIO, BaseNamespace
import sys, os, time
import numpy as np
import json
# append this repo's root path
sys.path.append(os.path.abspath(os.path.dirname(__file__)+'/../'))
import envs
import gym
from config import cfg

TRAIN_MODE = True

#--------------Alread have id, connect again -------------#
class ClientSpace(BaseNamespace):
    def on_connect(self):
        print('ClientSpace say connect')
        self.env_init()

    def on_disconnect(self):
        print('ClientSpace say disconnect')


    def on_predict_response(self, action):
        # print('client get action data = {}'.format(action))

        next_state, r, d, _ = self.env.step(action)
        self.ep_use_step += 1
        if self.ep_use_step >= self.EP_MAXSTEP:
            d = True

        if TRAIN_MODE:
            dic ={'state': self.state, 
                 'action': action, 
                 'reward': r, 
                 'done':d,
                 'next_state': next_state}
            self.emit('train_and_predict',dic)
        else:
            dic ={'state': next_state}
            self.emit('predict',dic)

        self.state = next_state
        

        if self.ep % 50 == 25:
            self.show()

        if d:
            self.log(r)
            self.ep+=1
            self.ep_use_step = 0
            self.state =  self.env.reset()
            
            dic ={'state': self.state}
            self.emit('predict',dic)
        
    def env_init(self):
        self.start_time = time.time()
        self.ep = 0
        self.ep_use_step = 0
        self.EP_MAXSTEP = 1000
        # env setting
        self.env = gym.make('gridworld-v0')
        self.state = self.env.reset()
        
        dic ={'state': self.state}
        self.emit('predict',dic)
        
    
    def log(self, reward):
        use_secs = time.time() - self.start_time
        time_str = '%3dh%3dm%3ds' % (use_secs/3600, (use_secs%3600)/60, use_secs % 60 )
        # print('%s -> EP:%4d, STEP:%3d, r: %4.2f, t:%s' % (self.client.client_id,  self.ep,  self.ep_use_step, self.reward, time_str))
        print('EP:%4d, STEP:%3d, r: %4.2f, t:%s' % (  self.ep,  self.ep_use_step, reward, time_str))
            
        

    def show(self):
        self.env._render(title = 'Episode: %3d, Step: %3d' % (self.ep,self.ep_use_step))

def connect_with_ns(ns):
    print('defins ns ={}'.format(ns))
    new_client = socketIO.define(ClientSpace, ns)

#--------------Get id -------------#
def on_connect():
    print('client say connect')

def on_reconnect():
    print('client say connect')

def on_disconnect():
    print('disconnect')

def on_session_response(new_id):
    print('Get id = {}'.format(new_id ))
    new_ns = '/' + str(new_id)  + '/predict'
    connect_with_ns(new_ns)

socketIO = SocketIO('127.0.0.1', 5000)
socketIO.on('connect', on_connect)
socketIO.on('disconnect', on_disconnect)
socketIO.on('reconnect', on_reconnect)
# Listen
socketIO.on('session_response', on_session_response)
socketIO.emit('session')
socketIO.wait()



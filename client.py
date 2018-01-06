from socketIO_client import SocketIO, BaseNamespace
import sys, os, time
import numpy as np
import json
# append this repo's root path
sys.path.append(os.path.abspath(os.path.dirname(__file__)+'/../'))
import envs
import gym
from config import cfg
from threading import Thread

TRAIN_MODE = True

#--------------Alread have id, connect again -------------#
class EnvSpace(BaseNamespace):
    def on_connect(self):
        self.frame_count = 0
        self.reward_buf = []
        self.state_buf  = []
        self.action_buf = []
        # print('EnvSpace say connect')
        self.start_time = time.time()
        self.ep = 0
        self.ep_use_step = 0
        self.ep_reward = 0
        self.env_name =''
        self.env_init()

    def on_disconnect(self):
        print('{} env say disconnect'.format(self.env_name))

        
    def env_init(self):
        pass

    def send_state_get_action(self, state):
        state       = state.tolist() if type(state) != list else state
        dic ={'state': state}
        self.emit('predict',dic)
    
    def send_train_get_action(self, state, action, reward, done,next_state):
        self.ep_use_step += 1
        self.ep_reward += reward
        
        state      = state.tolist() if type(state) == np.ndarray else state
        next_state = next_state.tolist() if type(next_state) == np.ndarray else next_state

        if not cfg['RL']['train_multi_steps']:
            # train_multi_steps = no, like using Q-Learning, SARSA 
            dic ={'state': state, 
                        'action': action, 
                        'reward': reward, 
                        'done':done,
                        'next_state': next_state}
            self.emit('train_and_predict',dic)
        else:
            # train_multi_steps = yes, like using DRL, A3C, DQN...etc.
            self.frame_count+=1
            self.state_buf.append(state)
            self.action_buf.append(action)
            self.reward_buf.append(reward)
            if self.frame_count >= cfg['RL']['train_run_steps'] or done:

                # print('I: send for train state_buf.shape={}, type(state_buf)={}, state_buf={}'.format(np.shape(self.state_buf), type(self.state_buf),self.state_buf))
                # print('I: send for train action_buf.shape={}, type(action_buf)={}, action_buf={}'.format(np.shape(self.action_buf), type(self.action_buf), self.action_buf))
                # print('I: send for train reward_buf.shape={}, type(reward_buf)={}, reward_buf={}'.format(np.shape(self.reward_buf), type(self.reward_buf), self.reward_buf))
                # print('I: send for train done = {}, type(done)={}'.format(done,type(done)) )

                dic ={'state': self.state_buf, 
                        'action': self.action_buf, 
                        'reward': self.reward_buf, 
                        'done':done,
                        'next_state': next_state}
                self.emit('train_and_predict',dic)

                self.frame_count = 0
                self.reward_buf = []
                self.state_buf  = []
                self.action_buf = []
            else:
                self.send_state_get_action(state)


        if done:
            self.ep+=1
            self.log()
            self.ep_use_step = 0


    def log(self):
        use_secs = time.time() - self.start_time
        time_str = '%3dh%3dm%3ds' % (use_secs/3600, (use_secs%3600)/60, use_secs % 60 )
        # print('%s -> EP:%4d, STEP:%3d, r: %4.2f, t:%s' % (self.client.client_id,  self.ep,  self.ep_use_step, self.reward, time_str))
        print('(%s) EP:%5d, STEP:%4d, r: %7.2f, t:%s' % ( self.env_name, self.ep,  self.ep_use_step, self.ep_reward, time_str))

    def set_name(self,name):
        # print('set_name  = ', name)
        self.env_name = name

class Gridworld_EX(EnvSpace):

    def env_init(self):
        self.EP_MAXSTEP = 1000
        self.env = gym.make('gridworld-v0')
        self.state = self.env.reset()

        self.send_state_get_action(self.state)

    def on_predict_response(self, action):
        # print('client get action data = {}'.format(action))

        next_state, reward, done, _ = self.env.step(action)
        if TRAIN_MODE:
            self.send_train_get_action(self.state, action, reward, done, next_state)
        else:
            self.send_state_get_action(self.state)
        if self.ep_use_step >= self.EP_MAXSTEP: done = True
        if self.ep % 50 == 25:
            self.env._render(title = 'Episode: %4d, Step: %4d' % (self.ep,self.ep_use_step))
        self.state = next_state
        if done:
            self.state =  self.env.reset()
            self.send_state_get_action(self.state)



class Client(Thread):
    def __init__(self, target_env_class, env_name=''):
        Thread.__init__(self)
        self.target_env_class = target_env_class
        self.env_name = env_name
        self.socketIO = SocketIO('127.0.0.1', 5000)
        self.socketIO.on('connect', self.on_connect)
        self.socketIO.on('disconnect', self.on_disconnect)
        self.socketIO.on('reconnect', self.on_reconnect)
        self.socketIO.on('session_response', self.on_session_response)
        # self.socketIO.emit('session')
        # self.socketIO.wait()
    
    def run(self):
        self.socketIO.emit('session')
        self.socketIO.wait()

    def on_connect(self):
        print('client say connect')

    def on_reconnect(self):
        print('client say connect')

    def on_disconnect(self):
        print('disconnect')

    def on_session_response(self, new_id):
        print('Get id = {}'.format(new_id ))
        new_ns = '/' + str(new_id)  + '/rl_session'
        self.connect_with_ns(new_ns)

    def connect_with_ns(self,ns):
        print('defins ns ={}'.format(ns))
        new_env = self.socketIO.define(self.target_env_class, ns)
        new_env.set_name(self.env_name)
        # method_to_call = getattr(new_env, self.env_call_fun)
        # result = method_to_call()

if __name__ == '__main__':
    # Client(EnvSpace) 
    Client(Gridworld_EX)
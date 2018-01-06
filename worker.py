import os 
from flask_socketio import Namespace, emit
import numpy as np
import cv2


from config import cfg
from DRL.Base import RL, DRL
from DRL.A3C import A3C
from DRL.TD import SARSA, QLearning

#------ Dynamic Namespce Predict -------#
class Worker(Namespace):
    def __init__(self,ns, client_id, sess = None, main_net = None ):
        super(Worker, self).__init__(ns)
        self.client_id = client_id

        
        # RL or DRL Init
        self.nA = cfg['RL']['action_num']
        method_class = globals()[cfg['RL']['method'] ]

        if issubclass(method_class, DRL):
            '''Use DRL'''
            self.sess = sess
            self.RL = method_class(self.sess, self.client_id, main_net)
        elif issubclass(method_class, RL):
            '''Use RL'''
            self.RL = method_class()
            pass
        else:
            print('E: Worker::__init__() say error method name={}'.format(cfg['RL']['method'] ))

        print("{}'s Worker Ready!".format(self.client_id))


    def on_connect(self):
        print('{} Worker Connect'.format(self.client_id))
        self.data_dir = 'data_pool/{}/'.format(self.client_id)
        if not os.path.isdir(self.data_dir):
            os.mkdir(self.data_dir)

        self.state_count = 0

    def on_disconnect(self):
        print('{} Worker Disconnect'.format(self.client_id))

    
    def on_predict(self, data):
        action = self.predict(data['state'])
        action = action.tolist() if type(action) == np.ndarray else action
        emit('predict_response', action)
        # self.save_data(state)

    def on_train(self, data):
        self.train(data['state'],data['action'],data['reward'],data['next_state'],data['done'])

    def on_train_and_predict(self, data):
        self.train(data['state'],data['action'],data['reward'],data['next_state'],data['done'])
        if not data['done']:
            action = self.predict(data['next_state'])
            action = action.tolist() if type(action) == np.ndarray else action

            emit('predict_response', action)

    def predict(self, state):
        # print("I: predict() state.shape: {}, type(state)= {} ".format(np.shape(state), type(state)) )
        state = np.array(state)
        return self.RL.choose_action(state)

    def train(self, states, actions, rewards, next_state, done ):
        # print('I: train get states.shape={}, type(states)={}'.format(np.shape(states), type(states)))
        # print('I: train get actions.shape={}, type(actions)={}'.format(np.shape(actions), type(actions)))
        # print('I: train get rewards.shape={}, type(rewards)={}'.format(np.shape(rewards), type(rewards)))
        # print('I: train get done = {}'.format(done)) 

        # more detail
        # print('I: train get states.shape={}, type(states)={}, states={}'.format(np.shape(states), type(states),states))
        # print('I: train get actions.shape={}, type(actions)={}, actions={}'.format(np.shape(actions), type(actions), actions))
        # print('I: train get rewards.shape={}, type(rewards)={}, rewards={}'.format(np.shape(rewards), type(rewards), rewards))
        # print('I: train get done = {}, type(done)={}'.format(done,type(done)) )
        states = np.array(states) if type(states) == list else states
        actions = np.array(actions) if type(actions) == list else actions
        next_state = np.array(next_state) if type(next_state) == list else next_state

        self.RL.train(states, actions, rewards, next_state, done)
        
    def save_data(self, data):
        state = data['state']
        state = np.array(state)
        print('train state.shape={}'.format(state.shape) )
        # pic_path =  train_dir + tag_id +'+r'+ str(reward) + '_a'+  str(action) + '_0.'+ self.identity +'.jpg'
        self.state_count += 1
        pic_path = '%s/%06d.jpg' % (self.data_dir, self.state_count)
        # cv2.imwrite(pic_path, state[:,:,0])
        print('pic_path = %s' % self.data_dir)
        cv2.imwrite(pic_path, state)
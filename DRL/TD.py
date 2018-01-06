"""
First Version:
https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/3_Sarsa_maze/RL_brain.py
"""

import numpy as np
import pandas as pd
from Base import RL
import six
from abc import ABCMeta,abstractmethod
from config import cfg

class TD(RL):
    def __init__(self):
        self.actions = list(range(cfg['RL']['action_num']))
        m = cfg['RL']['method']
        self.lr      = cfg[m]['LR']
        self.gamma   = cfg[m]['gamma']  # reward_decay
        self.epsilon = cfg[m]['epsilon-greedy']

        print('I: Use Learning Rate={}, Gamma={}, Epsilon={}'.\
            format(self.lr, self.gamma, self.epsilon))

        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def check_state_exist(self, state):
        # print('state={}, type = {}'.format(state, type(state)))
        state = str(state)
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )

    def choose_action(self, state):
        if type(state) != str:
            state = np.squeeze(state)
            # state = np.squeeze(state) if  len(np.shape(state)) > 1 else state
            # print("I: choose_action()  state={}, state.shape: {}, len(np.shape(s)): {}, type(state)= {} ".\
            #             format(state, np.shape(state),len(np.shape(state)), type(state)) )
            assert len(np.shape(state)) == 1,  'TD.choose_action() say state  dimention != 1'

        observation = str(state)
        # print("in TD choose_action")
        self.check_state_exist(observation)
        # action selection
        if np.random.rand() < self.epsilon:
            # choose best action
            state_action = self.q_table.loc[observation, :]
            state_action = state_action.reindex(np.random.permutation(state_action.index))     # some actions have same value
            action = state_action.argmax()
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return action

    def show_qtalbe(self):
        print(self.q_table)

    @abstractmethod
    def train(self, states, actions, rewards, next_state, done):
        pass


# off-policy
# Q-Learning
class QLearning(TD):
    def __init__(self):
        super(QLearning, self).__init__()

    # def learn(self, s, a, r, s_, done):
    def train(self, s, a, r, s_, done):
        s = np.squeeze(s) if  len(np.shape(s)) > 1 else s
        s_ = np.squeeze(s_) if  len(np.shape(s_)) > 1 else s_
        
        assert len(np.shape(s)) == 1,  'QLearning.train() say s  dimention != 1'
        assert len(np.shape(s_)) == 1, 'QLearning.train() say s_ dimention != 1'
        
        s = str(s)
        s_ = str(s_)
        
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        # if s_ != 'terminal':
        if not done:
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update


# on-policy
class SARSA(TD):

    def __init__(self):
        super(SARSA, self).__init__()
        self.next_action = None
    
    
    def choose_action(self, observation):
        # print("in SARSA choose_action, self.next_action={}".format(self.next_action))
        if self.next_action == None:
            return super(SARSA,self).choose_action(observation)  
        else:
            return self.next_action

    # def learn(self, s, a, r, s_,  done):
    def train(self, s, a, r, s_, done):

        # print("I: Before squeeze s={}, s.shape: {}, len(np.shape(s)): {}, type(s)= {} ".\
        #             format(s, np.shape(s),len(np.shape(s)), type(s)) )

        s  = np.squeeze(s)  
        s_ = np.squeeze(s_) 

        # print("I: After s={}, squeeze s.shape: {}, len(np.shape(s)): {}, type(s)= {} ".\
        #             format(s, np.shape(s), len(np.shape(s)), type(s)) )
        
        assert len(np.shape(s)) == 1,  'SARSA.train() say s  dimention != 1'
        assert len(np.shape(s_)) == 1, 'SARSA.train() say s_ dimention != 1'
        
        s = str(s)
        s_ = str(s_)

        self.check_state_exist(s_)
        self.next_action = super(SARSA,self).choose_action(s_)  
        q_predict = self.q_table.loc[s, a]

        if not done:
            q_target = r + self.gamma * self.q_table.loc[s_, self.next_action]  # next state is not terminal
        else:
            # if r > 0 and self.epsilon < 0.99:
            #     self.epsilon = self.epsilon + 0.001 if self.epsilon < 0.99 else self.epsilon
            q_target = r  # next state is terminal
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update

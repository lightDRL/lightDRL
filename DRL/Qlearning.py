
import numpy as np
from Base import RL


class Qlearning(RL):
    def __init__(self, cfg, model_log_dir="", sess = None):
        super(Qlearning, self).rl_init(cfg, model_log_dir)
        self.check_config()
        self.lr = cfg['Q-learning']['lr']             #learning rate  
        self.discount_factor = cfg['Q-learning']['discount_factor']  #gamma 
        
        self.q_table = np.zeros((self.s_discrete_n, self.a_discrete_n))

        print('self.q_table.shape = ', self.q_table.shape )

    def check_config(self):
        assert self.a_discrete==True, '[Warning] Q-Learning not support continuous action'
        assert self.s_discrete==True, '[Warning] Q-Learning not support continuous state'
        # print('self.s_dim[0] =', self.s_dim[0])
        # print('self.s_dim = ', self.s_dim,'type(self.s_dim) = ' ,type(self.s_dim), 'len(self.a_dim) = ', len(self.s_dim))
        # print('self.s_dim = ', self.a_dim, 'len(self.a_dim) = ', len(self.a_dim))
        assert len(self.s_shape) == 1, '[Warning] Q-Learning only support state -> len(state.shape)=1'
        assert len(self.a_shape) == 1, '[Warning] Q-Learning only support action  -> len(action.shape)=1 '

        

    def choose_action(self, state):
        # a = np.zeros(self.a)
        # a[ np.argmax(self.q_table[state])] =1
        a = self.q_table[state]
        return a
        '''
        return env.action_space.sample()
        if random.random() < EPSILON:
            return  np.argmax( np.random.randint(6, size=6))
            # return env.action_space.sample()
        else:
        '''

    def train(self):
        pass

    def add_data(self, s, a, r, d, s_):
        # print('add_data s=', s ,',a=', a ,',r=', r ,',d=', d ,',s_=',s_)
        s = s[0]
        # a = a[0]
        s_=s_[0]
        a = np.argmax(a) 
    
        # print('s=', s ,',a=', a ,',r=', r ,',d=', d ,',s_=',s_)
        self.q_table[s][a] += self.lr * ( r + self.discount_factor*max(self.q_table[s_]) -  self.q_table[s][a])
    
    def notify_ep_done(self):
        pass

    def get_log_dic(self):
        return None

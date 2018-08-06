
import numpy as np
from Base import RL


class Qlearning(RL):
    def __init__(self, cfg, model_log_dir="", sess = None):
        super(Qlearning, self).rl_init(cfg, model_log_dir)
        self.check_config()
        self.lr = cfg['Q-learning']['lr']             #learning rate  
        self.discount_factor = cfg['Q-learning']['discount_factor']  #gamma 
        
        # self.q_table = np.zeros((self.s_discrete_n, self.a_discrete_n))
        self.q_table = dict()

    def check_config(self):
        assert self.a_discrete==True, '[Warning] Q-Learning not support continuous action'
        # assert self.s_discrete==True, '[Warning] Q-Learning not support continuous state'
        # print('self.s_dim[0] =', self.s_dim[0])
        # print('self.s_dim = ', self.s_dim,'type(self.s_dim) = ' ,type(self.s_dim), 'len(self.a_dim) = ', len(self.s_dim))
        # print('self.s_dim = ', self.a_dim, 'len(self.a_dim) = ', len(self.a_dim))
        assert len(self.s_shape) == 1, '[Warning] Q-Learning only support state -> len(state.shape)=1'
        assert len(self.a_shape) == 1, '[Warning] Q-Learning only support action  -> len(action.shape)=1 '

        

    def choose_action(self, state):
        state = self.to_dic_key(state)
        if state  in self.q_table:
            a = self.q_table[state]
        else:
            a = np.zeros(self.a_discrete_n)
            a[ np.random.randint(self.a_discrete_n) ] =1
        return a

    def train(self):
        pass

    def add_data(self, s, a, r, d, s_):
        # print('add_data s=', s ,',a=', a ,',r=', r ,',d=', d ,',s_=',s_)
        # s = s[0]
        # a = a[0]
        # s_=s_[0]
        a = np.argmax(a) 
        
        # print('after a = ', a )
        # print('add_data -> before s = {}, type(s)={}, s_={}, type(_s)={}'.format( s, type(s), s_, type(s_)) )


        s = self.to_dic_key(s)
        s_ = self.to_dic_key(s_)
        # print('add_data -> after s = {}, type(s)={}, s_={}, type(_s)={}'.format( s, type(s), s_, type(s_)) )


        self.q_table[s]  = np.zeros(self.a_discrete_n) if not s  in self.q_table else self.q_table[s]
        self.q_table[s_] = np.zeros(self.a_discrete_n) if not s_ in self.q_table else self.q_table[s_]
        
        
        # print('after s=', s ,',a=', a ,',r=', r ,',d=', d ,',s_=',s_)
        if not d:
            self.q_table[s][a] += self.lr * ( r + self.discount_factor*max(self.q_table[s_]) -  self.q_table[s][a])
        else:
            self.q_table[s][a] = r

        # print('self.q_table[{}]={}'.format(s, self.q_table[s] ))
    
    def notify_ep_done(self):
        pass

    def get_log_dic(self):
        return None

    def to_dic_key(self, obj):
        # print('type(obj) = ', type(obj))
        if type(obj) == np.ndarray:
            obj = np.squeeze(obj)
            if (len(obj.shape)==0):
                return  np.asscalar(obj)
            else:   
                return tuple(obj) # tuple(map(tuple, obj))#str(obj)
        elif isinstance(obj, np.generic):
            return np.asscalar(obj)
        elif type(obj) == int or type(obj) == bool or type(obj)==float or type(obj)==tuple:
            return obj
        elif type(obj) == list:
            return tuple(obj)
        else:  # include type list
            return str(obj)
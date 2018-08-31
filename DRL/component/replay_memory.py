import numpy as np
from collections import deque
import random

import numpy as np


class ReplayMemory(object):

    def __init__(self, memory_capacity, random_seed=1234):
        self.memory_capacity = memory_capacity
        self.memory = deque()
        # print('ReplayMemory seed = ' + str(random_seed) )
        random.seed(random_seed)

    def add(self, s, a, r, d, s2):
        # self.log_time('replay memory before add ')
        # print('type(s) = ', type(s))
        # if type(s) == np.ndarray and s.shape[0]>1:
        #     assert s.shape[0] == a.shape[0] == r.shape[0] == d.shape[0] == s2.shape[0], 's, a, r, d, s2 not all length same'
        #     for ind in range(s.shape[0]):
        #         transition = (s[ind], a[ind], r[ind], d[ind], s2[ind])
        #         self.memory.append(transition)
        # else:
        transition = (s, a, r, d, s2)
        self.memory.append(transition)

        while len(self.memory) > self.memory_capacity:
            self.memory.popleft()

        # self.log_time('replay memory add ')

        
    def sample_batch(self, batch_size):
        batch = []

        if self.size() < batch_size:
            batch = random.sample(self.memory, self.size() )
        else:
            batch = random.sample(self.memory, batch_size)

        # self.log_time('random.sample')

        # batch_ary = np.array(batch)
        # self.log_time('batch_ary')

        
        # batch_ary_s = batch_ary[:,0]
        # self.log_time('batch_ary_s')
        # print('I: batch_ary_s.shape={}'.format(batch_ary_s.shape))

        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch])
        t_batch = np.array([_[3] for _ in batch])
        s2_batch = np.array([_[4] for _ in batch])
        # s_batch  = ([_[0] for _ in batch])
        # a_batch  = ([_[1] for _ in batch])
        # r_batch  = ([_[2] for _ in batch])
        # t_batch  = ([_[3] for _ in batch])
        # s2_batch = ([_[4] for _ in batch])

        # s_batch  = batch_ary[:,0]
        # a_batch  = batch_ary[:,1]
        # r_batch  = batch_ary[:,2]
        # t_batch  = batch_ary[:,3]
        # s2_batch = batch_ary[:,4]

        # self.log_time('to diff batch')

        # print('I: s_batch.shape={}, type(s_batch)={}'.format(np.shape(s_batch), type(s_batch)))
        # print('I: a_batch.shape={}, type(a_batch)={}'.format(np.shape(a_batch), type(a_batch)))
        # print('I: r_batch.shape={}, type(r_batch)={}'.format(np.shape(r_batch), type(r_batch)))
        # print('I: t_batch.shape={}, type(t_batch)={}'.format(np.shape(t_batch), type(t_batch)))
        # print('I: s2_batch.shape={}, type(s2_batch)={}'.format(np.shape(s2_batch), type(s2_batch)))
        

        return s_batch, a_batch, r_batch, t_batch, s2_batch

    def size(self):
        return len(self.memory)

    def clear(self):
        self.memory.clear()


    def log_time(self, s):
        import time
        if hasattr(self, 'ts'):
            print(s + ' use time = ' + str( time.time() - self.ts  ))
        self.ts = time.time()
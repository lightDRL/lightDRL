import numpy as np
from collections import deque
import random

class SimpleBuffer(object):

    def __init__(self, memory_capacity, random_seed=1234):
        self.memory_capacity = memory_capacity
        self.memory = deque()
        random.seed(random_seed)

    def add(self, s, a, r, d, s2, add_multiple = False):
        # print('type(s) = ', type(s))
        # print('add_multiple =', add_multiple)
        if add_multiple:
            assert s.shape[0] == a.shape[0] == r.shape[0] == d.shape[0] == s2.shape[0], 's, a, r, d, s2 not all length same'
            for ind in range(s.shape[0]):
                transition = (s[ind], a[ind], r[ind], d[ind], s2[ind])
                self.memory.append(transition)
        else:
            transition = (s, a, r, d, s2)
            self.memory.append(transition)

        while len(self.memory) > self.memory_capacity:
            self.memory.popleft()

        
    def sample_batch(self, batch_size):

        s_batch = np.array([_[0] for _ in self.memory])
        a_batch = np.array([_[1] for _ in self.memory])
        r_batch = np.array([_[2] for _ in self.memory])
        t_batch = np.array([_[3] for _ in self.memory])
        s2_batch = np.array([_[4] for _ in self.memory])

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

    def get_last_done(self):
        return self.memory[self.size() -1][3]
# import from https://github.com/hiwonjoon/tf-a3c-gpu/blob/master/history_buffer.py

import numpy as np
from collections import deque

class HistoryBuffer():
    def __init__(self,image_shape,frames_for_state) :
        self.buf = deque(maxlen=frames_for_state)
        self.expend_axis = len(image_shape)
        self.image_shape = list(image_shape)+[1]
        # print('self.expend_axis={}'.format(self.expend_axis) )
        # print('np.shape(image_shape)={}'.format(np.shape(self.image_shape)) )
        self.clear()

    def clear(self) :
        for i in range(self.buf.maxlen):
            self.buf.append(np.zeros(self.image_shape,np.float32))

    def add(self,o) : 
        self.buf.append(np.expand_dims(o.astype(np.float32),axis=self.expend_axis))
        
        state = np.concatenate([img for img in self.buf], axis=self.expend_axis)
        return state

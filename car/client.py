from socketIO_client import SocketIO, LoggingNamespace
import numpy as np
import json


socketIO = SocketIO('localhost', 8000, LoggingNamespace)


def print_something(msg):
    print(msg['data'])
    
   
def motorplan(msg):
    b=msg['data']
    print(b[0]+1)
socketIO.on('print_',print_something)
socketIO.on('action',motorplan) 


while True:

    socketIO.emit('online')
    d=np.random.rand(42)
    d=json.loads(json.dumps(d.tolist()))
    socketIO.emit('state', {'data':d})
    socketIO.emit('state_on_train', {'data':d})
    socketIO.wait(1)
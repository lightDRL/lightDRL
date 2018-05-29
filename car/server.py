from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from flask_restful import  Api
import numpy as np
from DDPG import DDPG
from car_env import CarEnv
import json
from dashboard import Dashboard

env = CarEnv()
s_dim = 2+env.O_LC
a_dim = 1
a_bound = env.action_bound[1]
MEMORY_CAPACITY = 3
RENDER=True
ddpg = DDPG(a_dim, s_dim, a_bound)
ddpg.restore()

app = Flask(__name__,static_folder='static', static_url_path='')
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

@socketio.on('online')
def connect():
    print("Robot is connect")
    emit('print_', {'data': 'welcome'})
    
@socketio.on('state')
def state(msg):
    state=np.array(msg['data'])
    a=ddpg.choose_action(state)
    a=json.loads(json.dumps(a.tolist()))
    emit('action', {'data': a})


@socketio.on('state_on_train')
def state_train(msg):
    if RENDER:
        env.render()
    state=np.array(msg['data'])
    a=ddpg.choose_action(state)
    s,r,s_,crash=env.state_reward(state,a,400)
    ddpg.store_transition(s, a, r / 10, s_)
    if ddpg.pointer > MEMORY_CAPACITY:
        print('learning')
        ddpg.learn()
    a=json.loads(json.dumps(a.tolist()))
    emit('action', {'data': a})

            

if __name__ == '__main__':

#    api = Api(app)
    # you can try on http://localhost:5000/dashboard
#    api.add_resource(Dashboard,'/dashboard')

    socketio.run(app, port=8000)
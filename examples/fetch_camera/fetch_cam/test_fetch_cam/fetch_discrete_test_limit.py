import numpy as np
import gym
import time
from matplotlib import pyplot as plt
from fetch_cam import FetchDiscreteEnv
from fetch_cam import FetchCameraEnv
from fsm import FSM

# start_pos = [1.3419437, 0.74910047 0.53471723] 
# limit_max = [1.50, 1.23, 0.42]          # not sure z
# limit_min = [1.00, 0.26, 0.42 ]

# because the limit of arm is sphere
limit_max = [1.50, 0.92, 0.42]          # not sure z
limit_min = [1.00, 0.58, 0.42 ]




def test_one_axis():
    env = FetchDiscreteEnv(dis_tolerance = 0.001, step_ds=0.005)
    obs = env.reset()

    ori_pos = (obs['eeinfo'][0]).copy()
    print('---ori_pos = ' , obs['eeinfo'][0],'----')

    for _ in range(2000):
        env.render()
        a = [0, 0, 0, 1, 0]
        s,r, d, info =  env.step(a) 
        print('now pos = ', env.pos)

def test_limit(go_pos):
    step_ds = 0.005
    env = FetchDiscreteEnv(dis_tolerance = 0.001, step_ds=0.005)
    obs = env.reset()
    done = False

    sum_r = 0

    while True:
        env.render()
        diff_x = go_pos[0] - env.pos[0]
        diff_y = go_pos[1] - env.pos[1]
        if diff_x > step_ds:
            a = [1, 0, 0, 0, 0]
        elif diff_x < 0 and abs(diff_x) >  step_ds:
            a = [0, 0 , 1, 0, 0]
        elif diff_y > step_ds:
            a = [0, 1, 0, 0, 0]
        elif diff_y < 0 and abs(diff_y) >  step_ds:
            a = [0, 0 , 0, 1, 0]
        else:
            break
        
        s,r, d, info =  env.step(a) 
        sum_r += r

        print('now pos = ', env.pos)

    print('Sum reward = ', sum_r)
    print('Final pos = ', env.pos)
test_limit(limit_max)
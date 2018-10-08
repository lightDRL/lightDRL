import numpy as np
import gym
import time
from matplotlib import pyplot as plt
from fetch_cam import FetchDiscreteEnv
from fetch_cam import FetchCameraEnv
from fsm import FSM


def basic_info():
    env = FetchDiscreteEnv(dis_tolerance = 0.001, step_ds=0.005)
    obs = env.reset()

    print("Start position ->", env.pos)
    print("Start (Close) gripper state ->", env.gripper_state)
    
    env.gripper_close(False)
    print("Open gripper state ->", env.gripper_state)
    # Open gripper state -> [0.0500507 0.0500507]

    env.gripper_close(True)
    print("Close gripper state ->", env.gripper_state)
    # Close gripper state -> [0.00184229 0.00184229]

    env.measure_obj_reward()
    
 
# run specific discrete (like [1,0,0,0,0]  2000 times)
def basic_test():
    env = FetchDiscreteEnv(dis_tolerance = 0.001, step_ds=0.005)
    obs = env.reset()
    done = False

    ori_pos = (obs['eeinfo'][0]).copy()
    print('---ori_pos = ' , obs['eeinfo'][0],'----')
    step  = 0
    robot_step = 0
    # env.render()
    s_time = time.time()

    for _ in range(2000):
        env.render()
        a = [1, 0, 0, 0, 0]
        s,r, d, info =  env.step(a) 
        print('now pos = ', env.pos)
        # env.render()
        # a = [0, 0, 1, 0, 0]
        # s,r, d, info =  env.step(a)

    time.sleep(1)
    # env.gripper_close()
    a = [0, 0, 0, 0, 1]
    s,r, d, info =  env.step(a)

    env.render()
    time.sleep(1)


    print('---final_pos = ' , obs['eeinfo'][0],'----')
    pos_diff = obs['eeinfo'][0] - ori_pos
    formattedList = ["%.2f" % member for member in pos_diff]
    print('---pos_diff = ' ,formattedList ,'----')

    print('use time = {:.2f}'.format(time.time()-s_time))



basic_info()
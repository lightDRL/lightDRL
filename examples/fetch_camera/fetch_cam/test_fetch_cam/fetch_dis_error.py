import numpy as np
import gym
import time
from matplotlib import pyplot as plt
from fetch_cam import FetchCameraEnv

from fsm import FSM


dis_tolerance  = 0.0001     # 1mm

env = FetchCameraEnv()
obs = env.reset()
done = False

want_pos = (obs['eeinfo'][0]).copy()
ori_pos = (obs['eeinfo'][0]).copy()
print('---ori_pos = ' , obs['eeinfo'][0],'----')
step  = 0
robot_step = 0

s_time = time.time()

while True:
    # env.render()
    
    now_pos = obs['eeinfo'][0]
    dis = np.linalg.norm(now_pos - want_pos)
    print('dis = ',dis)
    if dis < dis_tolerance:
        
        x, y, z, g = 0.01, 0.01, 0.01, 0.

        want_pos = obs['eeinfo'][0] + np.array([x, y, z]) 
        print('want_pos =' , want_pos)
        step +=1
        if step>=11:
            break
    else:
        x, y, z, g = 0., 0.0, 0., 0.
    a = np.array([x, y, z, g])
    obs, r, done, info = env.step(a)
    robot_step +=1

    if abs(x) > 0  or abs(y) > 0 or abs(z) > 0 :
        diff_x = obs['eeinfo'][0] - want_pos
        # print("pre_obs['eeinfo'][0] = ", pre_x)
        print("obs['eeinfo'][0] = {}, diff_x={}".format( obs['eeinfo'][0],  diff_x) )

    # time.sleep(0.5)
    

print('---final_pos = ' , obs['eeinfo'][0],'----')
print('---pos_diff = ' , obs['eeinfo'][0] - ori_pos,'----')

print('step = {}, robot_step={}'.format(step, robot_step))
print('use time = {:.2f}'.format(time.time()-s_time))
import numpy as np
import gym
import time
from matplotlib import pyplot as plt
import sys, os
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)),'../../'))
from fetch_cam import FetchDiscreteEnv
# from fetch_cam import FetchCameraEnv
from fetch_cam import FetchDiscreteCamEnv
from fsm import FSM
import cv2
from PIL import Image

def create_dir(dir_name):
    import os
    if os.path.exists(dir_name):
       import shutil
       shutil.rmtree(dir_name) 
    os.makedirs(dir_name)

def go_obj():
    # dis_tolerance  = 0.0001     # 1mm
    step_ds = 0.005
    env = FetchDiscreteEnv(dis_tolerance = 0.001, step_ds=0.005)
    obs = env.reset()
    
    done = False

    ori_pos = (obs['eeinfo'][0]).copy()
    print('---ori_pos = ' , obs['eeinfo'][0],'----')
    step  = 0
    robot_step = 0
    # env.render()
    s_time = time.time()
    sum_r = 0

    while True:
        env.render()
        diff_x = env.obj_pos[0] - env.pos[0]
        diff_y = env.obj_pos[1] - env.pos[1]
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
        
    

    a = [0, 0, 0, 0, 1]
    s,r, d, info =  env.step(a)
    sum_r +=r
    env.render()
    print('epsoide sum_r = ', sum_r)

    print('---final_pos = ' , obs['eeinfo'][0],'----')
    pos_diff = obs['eeinfo'][0] - ori_pos
    formattedList = ["%.2f" % member for member in pos_diff]
    print('---pos_diff = ' ,formattedList ,'----')

    print('use time = {:.2f}'.format(time.time()-s_time))




def go_obj_savepic(is_render = True):
    # dis_tolerance  = 0.0001     # 1mm
    step_ds = 0.005
    env = FetchDiscreteEnv(dis_tolerance = 0.001, step_ds=0.005)

    print('---ori_pos = ' , env.pos,'----')
    # step  = 0
    # robot_step = 0
    # # env.render()
    s_time = time.time()
    # env.render()
    # step_count = 0

    for i in range(5):
        obs = env.reset()
        env.gripper_close(False)
        env.render()
        save_dir = 'tmp/fetch_run_pic_%02d' %i 
        create_dir(save_dir)
        step_count = 0
        print('------start ep %03d--------' % i)
        sum_r = 0
        while True:
            if is_render:
                env.render()
            diff_x = env.obj_pos[0] - env.pos[0]
            diff_y = env.obj_pos[1] - env.pos[1]
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
            step_count +=1
            s,r, d, info =  env.step(a)
            sum_r += r  
            # rgb_external = env.sim.render(width=256, height=256, camera_name="external_camera_0", depth=False,
            #         mode='offscreen', device_id=-1)
            # rgb_gripper = env.sim.render(width=256, height=256, camera_name="gripper_camera_rgb", depth=False,
            #     mode='offscreen', device_id=-1)
            rgb_external = env.sim.render(width=256, height=256, camera_name="external_camera_0", depth=False,
                    mode='offscreen', device_id=-1)
            rgb_gripper = env.sim.render(width=256, height=256, camera_name="gripper_camera_rgb", depth=False,
                mode='offscreen', device_id=-1)

            # print('type(rgb_gripper) = ', type(rgb_gripper),', shape=', np.shape(rgb_gripper))
            img = Image.fromarray(rgb_gripper, 'RGB')
            # img.save(save_dir + '/%03d.jpg' % step_count)
            img.save(save_dir + '/%03d_r%3.2f.jpg' % (step_count,r ))
        


        a = [0, 0, 0, 0, 1]
        s,r, d, info =  env.step(a)
        sum_r += r  
        print('sum_r = ', sum_r)
        print("use step = ", step_count)

        env.render()


    # print('---final_pos = ' , obs['eeinfo'][0],'----')
    # pos_diff = obs['eeinfo'][0] - ori_pos
    # formattedList = ["%.2f" % member for member in pos_diff]
    # print('---pos_diff = ' ,formattedList ,'----')

    print('use time = {:.2f}'.format(time.time()-s_time))




def go_obj_savepic_with_camenv(is_render = True):
    save_dir = 'z_fetch_run_pic'
    create_dir(save_dir)
    # dis_tolerance  = 0.0001     # 1mm
    step_ds = 0.005
    env = FetchDiscreteCamEnv(dis_tolerance = 0.001, step_ds=0.005)
    # obs = env.reset()
    # done = False

    # ori_pos = (obs['eeinfo'][0]).copy()
    print('---ori_pos = ' , env.pos,'----')
    # step  = 0
    # robot_step = 0
    # # env.render()
    s_time = time.time()
    # env.render()
    # step_count = 0

    for i in range(5):
        obs = env.reset()
        # env.gripper_close(False)
        env.render()
        save_dir = 'tmp/fetch_run_pic_%02d' %i 
        create_dir(save_dir)
        step_count = 0
        print('------start ep %03d--------' % i)
        sum_r = 0
        while True:
            if is_render:
                env.render()
            diff_x = env.obj_pos[0] - env.pos[0]
            diff_y = env.obj_pos[1] - env.pos[1]
            if diff_x > step_ds:
                a = 0 # [1, 0, 0, 0, 0]
            elif diff_x < 0 and abs(diff_x) >  step_ds:
                a = 2 # [0, 0 , 1, 0, 0]
            elif diff_y > step_ds:
                a = 1 # [0, 1, 0, 0, 0]
            elif diff_y < 0 and abs(diff_y) >  step_ds:
                a = 3 # [0, 0 , 0, 1, 0]
            else:
                break
            step_count +=1
            s,r, d, info =  env.step(a)
            sum_r += r  
            
            cv2.imwrite(save_dir + '/%03d_r%3.2f.jpg' % (step_count,r ),  s[:,:,0])
        


        a = 4 # [0, 0, 0, 0, 1]
        s,r, d, info =  env.step(a)
        sum_r += r  
        print('sum_r = ', sum_r)
        print("use step = ", step_count)

        env.render()


    # print('---final_pos = ' , obs['eeinfo'][0],'----')
    # pos_diff = obs['eeinfo'][0] - ori_pos
    # formattedList = ["%.2f" % member for member in pos_diff]
    # print('---pos_diff = ' ,formattedList ,'----')

    print('use time = {:.2f}'.format(time.time()-s_time))

go_obj_savepic()
# go_obj()

# go_obj_savepic_with_camenv()
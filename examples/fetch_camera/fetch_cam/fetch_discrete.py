from gym import utils
from fetch_cam import fetch_env
import numpy as np 
from fetch_cam.utils import robot_get_obs

class FetchDiscreteEnv(fetch_env.FetchEnv, utils.EzPickle):
    '''
        [1, 0, 0, 0, 0]: (+step_ds,        0)
        [0, 1, 0, 0, 0]: (       0, +step_ds)
        [0, 0, 1, 0, 0]: (-step_ds,        0)
        [0, 0, 0, 1, 0]: (       0, -step_ds)
        [0, 0, 0, 0, 1]: gripper down and close gripper

    '''
    def __init__(self, reward_type='sparse', dis_tolerance = 0.001, step_ds=0.005, is_render=False):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
        }
        fetch_env.FetchEnv.__init__(
            self, 'fetch/pick_and_place.xml', has_object=True, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)

        self.dis_tolerance = dis_tolerance
        self.ds = step_ds     # each step run distance, [1,0,0,0] run dx = 0.01
        self.is_render = is_render

    @property
    def pos(self):
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        return grip_pos

    @property
    def obj_pos(self):
        return self.sim.data.get_site_xpos('object0')

    @property
    def gripper_state(self):
        robot_qpos, robot_qvel = robot_get_obs(self.sim)
        return robot_qpos[-2:]

    def go_diff_pos(self, diff_pos, gripper_state = 1):
        # diff_pos = [dx, dy, dz]
        this_action_use_step = 0
        for_set_action =  np.array( [  diff_pos[0], diff_pos[1], diff_pos[2], gripper_state ]  )

        want_pos = self.pos + diff_pos
        self._set_action(for_set_action)
        while True:
            self.sim.step()
            self._step_callback()
            this_action_use_step+=1

            if self.is_render:
                self.render()

            dis = np.linalg.norm(self.pos - want_pos)
            # print('dis = ',dis)
            if dis < self.dis_tolerance:
                return True    

            if this_action_use_step>10:
                # use too much, maybe limit of arm
                # print('this_action_use_step >= 10, maybe limitation position of arm -> break')
                return False
            
    def gripper_close(self, close = True):
        # set close action, close = True -> close,  close = False -> Open

        while True:
            robot_qpos, robot_qvel = robot_get_obs(self.sim)
            pre_gripper_state = robot_qpos[-2:]
    
             
            num = -1 if close else  1
            
            # do close 
            for_set_action =  np.array( [ 0, 0, 0, num ]  )
            self._set_action(for_set_action)
            self.sim.step()
            self._step_callback()

            if self.is_render:
                self.render()

            # new grip state
            robot_qpos, robot_qvel = robot_get_obs(self.sim)
            gripper_state = robot_qpos[-2:]

            dis = np.linalg.norm(gripper_state - pre_gripper_state)

            # print('gripper_state = ', gripper_state,', dis = ', dis,', num = ', num)
            if dis < 0.0001 : #self.dis_tolerance  :
                # gripper_result= 1. if (gripper_state >= 0.048).all() else -1.  ,
                # print('gripper_state = ', gripper_state)
                return gripper_state

        
    def measure_obj_reward(self):
        obj_name = 'object0'
        object_pos = self.sim.data.get_site_xpos(obj_name)
        pos_xy = self.pos[:2]
        obj_xy = object_pos[:2]
        
        arm_2_obj_xy = np.linalg.norm(pos_xy -obj_xy)
        r = -0.001
        max_dis = 0.1000
        if arm_2_obj_xy  < max_dis:
            r =  (max_dis - arm_2_obj_xy ) / max_dis *0.01
            # print('r = %.2f' % r ,'pos_xy = ', pos_xy,', obj_xy = ', obj_xy,', arm_2_obj_xy = ', arm_2_obj_xy)

        return r
        
    # override robot_env:step()
    def step(self, action):
        reward = 0
        self.use_step += 1
        done = True  if self.use_step >= 100 else False
        
        if action[4]==1:
            
            object_pos = self.sim.data.get_site_xpos('object0')
            dz = object_pos[2] - self.pos[2]
            self.go_diff_pos([0, 0, dz])
            # print('DOWN object_pos = ', object_pos,'self.pos = ', self.pos, ',dz = ', dz)

            final_gripper_state = self.gripper_close()
            # print("final final_gripper_state = ", final_gripper_state)
            
            # up
            self.go_diff_pos([0, 0, -dz], gripper_state = -1)

            if self.gripper_state[0] > 0.01:
                reward = 1
            else:
                reward = -1
            
            done = True

            if self.is_render:
                if final_gripper_state[0] > 0.01:
                    grip_close_reward = 1
                else:
                    grip_close_reward = -1
                diff_reward = 'DIFF' if grip_close_reward!=reward else 'SAME' 
                print(f'grip_close_reward = {grip_close_reward}, reward={reward}, {diff_reward}')
                import time
                for i in range(1, 100):
                    # print('in render 0.01, i = ', i)
                    # time.sleep(0.01)
                    # self.gripper_close()
                    self.sim.step()
                    self._step_callback()
                    self.render()    
        else:
            
            dx = action[0] *  self.ds - action[2] * self.ds
            dy = action[1] *  self.ds - action[3] * self.ds
            dz = 0
            # print('run dx=%0.2f, dy=%0.2f, dz=%0.2f' % (dx, dy, dz))
            
            go_diff_reulst = self.go_diff_pos([dx, dy, dz])
            if not go_diff_reulst:
                done = True
                reward = -1
            else:
                reward = self.measure_obj_reward() # 0
            


        obs = self._get_obs()

        return obs, reward, done, None

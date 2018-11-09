from gym import utils
from fetch_cam import fetch_env
import numpy as np 
from fetch_cam.utils import robot_get_obs
from mujoco_py.modder import TextureModder
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
import random

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

    def get_obj_pos(self, obj_id ):
        obj_name  = 'object%d' % obj_id
        return self.sim.data.get_site_xpos(obj_name)

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
            
            object_pos = self.sim.data.get_site_xpos('object0').copy()
            dz = object_pos[2] - self.pos[2]
            self.go_diff_pos([0, 0, dz])
            # print('DOWN object_pos = ', object_pos,'self.pos = ', self.pos, ',dz = ', dz)

            final_gripper_state = self.gripper_close()
            # print("final final_gripper_state = ", final_gripper_state)
            
            # up
            self.go_diff_pos([0, 0, -dz], gripper_state = -1)
            new_object_pos = self.sim.data.get_site_xpos('object0')
            if self.gripper_state[0] > 0.01 and (new_object_pos[2]-object_pos[2])>=0.2: # need to higher than 20cm    
                reward = 1
                ori_xy = object_pos[:2]
                new_xy = new_object_pos[:2]
                
                diff_xy = np.linalg.norm(new_xy -ori_xy)    
                diff_xy = diff_xy / 0.01  # to cm

                # print('diff_xy = ', diff_xy)

                reward-= diff_xy * 0.1

                # reward = 1

                

            else:
                reward = -1
            
            done = True

            if self.is_render:
                if final_gripper_state[0] > 0.01:
                    grip_close_reward = 1
                else:
                    grip_close_reward = -1
                diff_reward = 'DIFF' if grip_close_reward!=reward else 'SAME' 
                # print(f'grip_close_reward = {grip_close_reward}, reward={reward}, {diff_reward}')
                import time
                for i in range(1, 10):
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
                reward = 0 # self.measure_obj_reward() # 0
            


        obs = self._get_obs()

        return obs, reward, done, None

    # ------------------------for siamese network------------------------
    def rand_obj0_hide_obj1_obj2(self):
        # self.sim.set_state(self.initial_state)
        # self.gripper_to_init()
        # self.env.rand_objs_color()
        # rand obj1 pose
        self.backup_objs_xpos = []
        for i in range(3):
            obj_joint_name = 'object%d' % i
            pos = self.sim.data.get_site_xpos(obj_joint_name).copy()
            # print('[%d] pos = ' % i , pos)
            self.backup_objs_xpos.append(pos)
        
        object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range*0.5, self.obj_range*0.5, size=2)
        object_qpos = self.sim.data.get_joint_qpos('object0:joint')
        assert object_qpos.shape == (7,)
        object_qpos[:2] = object_xpos
        self.sim.data.set_joint_qpos('object0:joint', object_qpos)
        
        # print('object_qpos = ', object_qpos)
        # hide obj1, obj2
        obj_z =  object_qpos[2]
        obj_hide_z = obj_z - 0.15

        # print('obj0 z ', obj_z)
        for i in range(1,3):
            obj_joint_name = 'object%d:joint' % i
            # print("modify ", obj_joint_name)
            object_qpos = self.sim.data.get_joint_qpos(obj_joint_name)
            assert object_qpos.shape == (7,)
            object_qpos[2] = obj_hide_z

            self.sim.data.set_joint_qpos(obj_joint_name, object_qpos)

        self.sim.forward()
        self._step_callback()

        # self.render()
        # self.gripper_to_init()

    def hide_obj1_obj2(self):
       
        # hide obj1, obj2
        obj_z =  self.obj_pos[2]
        obj_hide_z = obj_z - 0.15

        for i in range(1,3):
            obj_joint_name = 'object%d:joint' % i
            # print("modify ", obj_joint_name)
            object_qpos = self.sim.data.get_joint_qpos(obj_joint_name)
            assert object_qpos.shape == (7,)
            object_qpos[:3] = [1.3, 0.6, obj_hide_z]
            # object_qpos[2] = obj_hide_z
            print('object_qpos = ', object_qpos)
            self.sim.data.set_joint_qpos(obj_joint_name, object_qpos)

        self.sim.forward()
        self._step_callback()

        # self.render()
        # self.gripper_to_init()

    # def rand_objs_color(self):
    #     for i in range(3):
    #         obj_name = 'object%d' % i
    #         # it will fail in first time
    #         try:
    #             modder = TextureModder(self.sim)
    #             # color = np.array([0, 0,255 ]
    #             color = np.array(np.random.uniform(size=3) * 255, dtype=np.uint8) 
    #             modder.set_rgb(obj_name, color )

                
    #         except Exception as e :
    #             pass
    #             # print('[E] fail to set color to ' , obj_name,', becase e -> ', e )

    #     self.sim.forward()

    def set_obj_color(self, geom_name, rgba):
        geom_id = self.sim.model.geom_name2id(geom_name)
        mat_id = self.sim.model.geom_matid[geom_id]
        # print('mat_id = ', mat_id)
        # print('self.model.mat_rgba = ' , self.sim.model.mat_rgba)
        # self.model.mat_rgba[mat_id, :] = 1.0
        self.sim.model.mat_rgba[mat_id] = rgba # [0., 1., 0.,1.]



    def rand_objs_color(self, exclude_obj0 = False):
        start_obj = 0 if exclude_obj0==False else 1
        for i in range(start_obj, 3):
            obj_name = 'object%d' % i
            # it will fail in first time
            try:
                color =  np.random.rand(4)
                color[3] = 1.0
                # print('!!color = ', color) 
                self.set_obj_color(obj_name, color)

            except Exception as e :
                # pass
                print('[E] fail to set color to ' , obj_name,', becase e -> ', e )

        self.sim.forward()


    def rand_objs_hsv(self):
        
        if not hasattr(self, 'color_space'):
            # only catch 3603 colors
            self.color_space = []
            HUE_MAX = 180
            for h in np.linspace(0,1,HUE_MAX,endpoint=False):
                for s in np.linspace(0.1, 1.0, 10,endpoint=True):
                    for v in [0.5,1.0]:
                        self.color_space.append([h, s, v])
            self.color_space.append([0,0,  0])
            self.color_space.append([0,0,0.5])
            self.color_space.append([0,0,1.0])

        try:
            hsv_3 = random.sample(self.color_space, 3)

            for i in range(3):
                obj_name = 'object%d' % i
                '''
                rgb = hsv_to_rgb( hsv_3[i] )* 255.0
                rgb = rgb.astype(int)
                # print('hsv = ', hsv_3[i], ', rgb=', rgb)
                
                modder = TextureModder(self.sim)
                modder.set_rgb(obj_name, rgb )
                '''
                rgb = hsv_to_rgb( hsv_3[i] )
                rgba = [rgb[0], rgb[1], rgb[2], 1.0]
                self.set_obj_color(obj_name, rgba)
        except Exception as e :
            # pass
            print('[E] fail to set color to ' , obj_name,', becase e -> ', e )


                
    def recover_obj0_obj1_obj2_pos(self):
        # print('len(self.backup_objs_xpos) = ', len(self.backup_objs_xpos))

        for i in range(len(self.backup_objs_xpos)):
            obj_joint_name = 'object%d:joint' % i
            object_qpos = self.sim.data.get_joint_qpos(obj_joint_name)
            assert object_qpos.shape == (7,)
            object_qpos[:3] = self.backup_objs_xpos[i]
            # print('self.backup_objs_xpos[%d] = ' % i , self.backup_objs_xpos[i])
            self.sim.data.set_joint_qpos(obj_joint_name, object_qpos)

        self.sim.forward()
        self._step_callback()

            
from gym import utils
from fetch_cam import fetch_env
import numpy as np 
from fetch_cam.utils import robot_get_obs
from mujoco_py.modder import TextureModder
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
import random
import cv2 

class FetchDiscreteEnv(fetch_env.FetchEnv, utils.EzPickle):
    '''
        [1, 0, 0, 0, 0]: (+step_ds,        0)
        [0, 1, 0, 0, 0]: (       0, +step_ds)
        [0, 0, 1, 0, 0]: (-step_ds,        0)
        [0, 0, 0, 1, 0]: (       0, -step_ds)
        [0, 0, 0, 0, 1]: gripper down and close gripper

    '''
    def __init__(self, reward_type='sparse', dis_tolerance = 0.001, step_ds=0.005, use_tray=True,  is_render=False):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
        }

        env_xml_name = 'fetch/pick_and_place_tray.xml' if use_tray else 'fetch/pick_and_place.xml'

        fetch_env.FetchEnv.__init__(
            # self, 'fetch/pick_and_place.xml', has_object=True, block_gripper=False, n_substeps=20,
            self, env_xml_name , has_object=True, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)

        self.dis_tolerance = dis_tolerance
        self.ds = step_ds     # each step run distance, [1,0,0,0] run dx = 0.01
        self.is_render = is_render
        self.use_tray = use_tray

        # self.hold_gripper_close = False

    @property
    def pos(self):
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        return grip_pos

    @property
    def obj_pos(self):
        return self.sim.data.get_site_xpos('object0')

    @property
    def red_tray_pos(self):
        # return self.sim.data.get_site_xpos('red_tray')
        return self.sim.data.get_body_xpos('red_tray')

    def get_obj_pos(self, obj_id ):
        obj_name  = 'object%d' % obj_id
        return self.sim.data.get_site_xpos(obj_name)

    @property
    def gripper_state(self):
        robot_qpos, robot_qvel = robot_get_obs(self.sim)
        return robot_qpos[-2:]

    @property
    def is_gripper_close(self):
        # about grip a cube,open -> [0.05004456 0.05004454], close ->  [0.02422073 0.02435405]
        g = self.gripper_state
        if g[0] <= 0.045 : #and  g[1] <= 0.45:
            return True
        else:
            return False

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
            r =  (max_dis - arm_2_obj_xy ) / max_dis *0.001
            # print('r = %.2f' % r ,'pos_xy = ', pos_xy,', obj_xy = ', obj_xy,', arm_2_obj_xy = ', arm_2_obj_xy)

        return r

    def measure_tray_reward(self):
        tray_pos =  self.red_tray_pos #self.sim.data.get_site_xpos(obj_name)
        pos_xy = self.pos[:2]
        tray_xy = tray_pos[:2]
        
        arm_2_tray_xy = np.linalg.norm(pos_xy -tray_xy)
        r = -0.001
        max_dis = 0.1000
        if arm_2_tray_xy  < max_dis:
            r =  (max_dis - arm_2_tray_xy ) / max_dis *0.001
            # print('r = %.2f' % r ,'pos_xy = ', pos_xy,', obj_xy = ', obj_xy,', arm_2_obj_xy = ', arm_2_obj_xy)

        return r
        


    def pick_place(self, pick = True):
        object_pos = self.sim.data.get_site_xpos('object0').copy()

        if pick:
            dz = object_pos[2] - self.pos[2]
        else:
            dz = self.red_tray_pos[2] - self.pos[2] + 0.01


        down_gripper_state = 1 if pick else -1
        self.go_diff_pos([0, 0, dz], down_gripper_state)
        # print('DOWN object_pos = ', object_pos,'self.pos = ', self.pos, ',dz = ', dz)

        
        final_gripper_state = self.gripper_close(pick)
        # print("final final_gripper_state = ", final_gripper_state)
        
        # for i in range(1, 10):
        #     self.sim.step()
        #     self._step_callback()
        #     self.render()  

        # up
        up_gripper_state = -1 if pick else 1
        self.go_diff_pos([0, 0, -dz], gripper_state = up_gripper_state)
        new_object_pos = self.sim.data.get_site_xpos('object0')
        
        
        if pick:
            if  self.gripper_state[0] > 0.01 and (new_object_pos[2]-object_pos[2])>=0.2: # need to higher than 20cm    
                reward = 0.5 if self.use_tray else 1.0  #0.5
                ori_xy = object_pos[:2]
                new_xy = new_object_pos[:2]
                
                diff_xy = np.linalg.norm(new_xy -ori_xy)    
                diff_xy = diff_xy / 0.01  # to cm

                # print('diff_xy = ', diff_xy)

                reward-= diff_xy * 0.01

            else:
                reward = -1
        else:
            assert self.use_tray==True,'Strange!' 
            # diff_tray_xy = np.linalg.norm( new_object_pos[:2] - self.red_tray_pos[:2]) 
            diff_tray_xy = np.linalg.norm( self.pos[:2] - self.red_tray_pos[:2]) 
            # print('diff_tray_xy = ', diff_tray_xy)   
            if diff_tray_xy < 0.03:
                reward = 1
            else:
                reward = -1
        

        if self.is_render:
            if final_gripper_state[0] > 0.01:
                grip_close_reward = 1
            else:
                grip_close_reward = -1
            diff_reward = 'DIFF' if grip_close_reward!=reward else 'SAME' 
            print(f'grip_close_reward = {grip_close_reward}, reward={reward}, {diff_reward}')
            import time
            for i in range(1, 10):
                # print('in render 0.01, i = ', i)
                # time.sleep(0.01)
                # self.gripper_close()
                self.sim.step()
                self._step_callback()
                self.render()  
        return reward
    # override robot_env:step()
    def step(self, action):
        reward = 0.0
        self.use_step += 1
        done = True  if self.use_step >= 150 else False
        
        if action[4]==1:
            reward = self.pick_place(True)
            # self.hold_gripper_close = True
            if reward == -1 or not self.use_tray:
                done = True
            
        elif action[5]==1:
            assert self.use_tray==True,'Strange!' 
            if self.is_gripper_close:
                reward = self.pick_place(False)
            else:
                reward = -1
            # self.is_gripper_close = False
            done = True
        else:
            
            dx = action[0] *  self.ds - action[2] * self.ds
            dy = action[1] *  self.ds - action[3] * self.ds
            dz = 0
            # print('run dx=%0.2f, dy=%0.2f, dz=%0.2f' % (dx, dy, dz))
            hold = -1 if self.is_gripper_close else 1
            go_diff_reulst = self.go_diff_pos([dx, dy, dz], hold)
            # print('go_diff_reulst = ', go_diff_reulst)
            if not go_diff_reulst:
                done = True
                reward = -1
            else:
                if self.use_tray and self.is_gripper_close:
                    reward = self.measure_tray_reward() # 0
                else:
                    reward =self.measure_obj_reward() # 0
            

        # print('action = ', action,', reward = ', reward)
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
            object_qpos[2] = obj_hide_z

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


    def one_hsv_2_rgb(self, h, s, v):
        hsv_one_color = np.uint8([[[h,s,v ]]]) 
        hsv2rgb = cv2.cvtColor(hsv_one_color, cv2.COLOR_HSV2RGB)
        return hsv2rgb[0][0][0], hsv2rgb[0][0][1], hsv2rgb[0][0][2]


    def rand_red_or_not(self, obj_name, use_red_color):
        if use_red_color:
            h = np.random.randint(174,179)
            s = np.random.randint(211,255)
            v = np.random.randint(150 ,255)
        else:
            h = np.random.randint(0,150)
            s = np.random.randint(0,255)
            v = np.random.randint(0 ,255)
        
        r, g, b = self.one_hsv_2_rgb(h, s, v)
        color = [r/255.0, g/255.0, b/255.0, 1.0]
        try:
            self.set_obj_color(obj_name, color)
        except Exception as e :
            # pass
            print('[E] fail to set color to ' , obj_name,', becase e -> ', e )

        self.sim.forward()


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


    def generate_3_obj_pos(self):
        assert self.has_object == True, 'self.has_object != True'
        

        obj_pos_ary =[]
        if self.use_tray:
            red_tray_pos = self.sim.data.get_body_xpos('red_tray')

            
            for i in range(3):
                obj_gripper_dis = 0
                object_xpos = self.initial_gripper_xpos[:2]

                while  obj_gripper_dis < 0.1 or self.check_dis_any_small_threshold(obj_pos_ary, object_xpos , 0.1) or obj_tray_dis < 0.05 :
                    object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
                    obj_gripper_dis = np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2])
                    obj_tray_dis = np.linalg.norm(object_xpos -red_tray_pos[:2])

                obj_pos_ary.append(object_xpos)
        else:
            for i in range(3):
                obj_gripper_dis = 0
                object_xpos = self.initial_gripper_xpos[:2]
                while  obj_gripper_dis < 0.1 or self.check_dis_any_small_threshold(obj_pos_ary, object_xpos , 0.1) :
                    object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
                    obj_gripper_dis = np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2])

                obj_pos_ary.append(object_xpos)

        return obj_pos_ary

            
'''
roslaunch manipulator_h_manager dual_arm.launch
rosrun rosserial_python serial_node.py /dev/wrs/arduino
'''
import cv2
import numpy as np
from arm_control import ArmTask, SuctionTask
import rospy


IMG_W_H = 84
# because thread bloack the image catch (maybe), so create the shell class 
class WRSEnv:
    def __init__(self, dis_tolerance = 0.001, step_ds=0.005, gray_img = False,  is_render = True):
        self.step_ds = step_ds
        self.gray_img = gray_img
        self.is_render = is_render

        rospy.init_node('test_arm_task')
        print("Test arm task script")
        # cam init
        self.cap = cv2.VideoCapture(0)

        print('Camera status: ', self.cap)
        # arm init
        self.arm = ArmTask('left_arm')
        rospy.sleep(0.3)

        self.arm.set_speed(30)

        # sucktion init
        SuctionTask.switch_mode(True)
        self.suck = SuctionTask(_name='left')

    def get_img(self):
        for i in range(5):
            # clear all image
            self.cap.grab()
        ret, rgb_gripper = self.cap.read()
        # rgb_gripper = cv2.cvtColor(rgb_gripper, cv2.COLOR_RGB2BGR)
        if ret==False:
            return None

        if self.is_render:           
            self.render_gripper_img(rgb_gripper)
            # save image
            import time
            import os
            if not os.path.exists('tmp_img_path'):
                os.makedirs('tmp_img_path')

            img_path = 'tmp_img_path/{}.jpg'.format(time.time())
            save_rgb_gripper = cv2.cvtColor(rgb_gripper, cv2.COLOR_BGR2RGB)
            cv2.imwrite(img_path, save_rgb_gripper)

        # s = self.state_preprocess(rgb_gripper)
        resize_img = cv2.resize(rgb_gripper, (IMG_W_H, IMG_W_H), interpolation=cv2.INTER_AREA)
        if self.gray_img:
            gray_img = cv2.cvtColor(resize_img, cv2.COLOR_RGB2GRAY)
            gray_img = np.reshape(gray_img,(IMG_W_H,IMG_W_H,1))
            return gray_img
        else:
            return resize_img

    def step(self,action):
        print('Use action = ', action)
        done  = False  
        reward = 0
        if action==0:
            self.arm.relative_move_pose('line', [self.step_ds, 0, 0] )
            self.arm.wait_busy()
        elif action==1:
            self.arm.relative_move_pose('line', [0, self.step_ds, 0] )
            self.arm.wait_busy()
        elif action==2:
            self.arm.relative_move_pose('line', [-self.step_ds, 0, 0] )
            self.arm.wait_busy()
        elif action==3:
            self.arm.relative_move_pose('line', [0, -self.step_ds, 0] )
            self.arm.wait_busy()
        elif action==4:
            self.suck.gripper_vaccum_on()
            # rospy.sleep(5)
            self.arm.relative_move_pose('line', [0, 0,-0.305] )
            self.arm.wait_busy()
            rospy.sleep(1)

            is_suck = self.suck.gripped 
            print('Is sucked: {}'.format(is_suck))

            self.arm.relative_move_pose('line', [0, 0, 0.305] )
            self.arm.wait_busy()
            rospy.sleep(1)
            done = True

            reward = 1 if is_suck else -1


        return self.get_img(), reward, done, None


    def reset(self):
        self.suck.gripper_vaccum_off()
        # self.arm.back_home()
        # self.arm.ikMove('p2p', (0.4, 0.1, -0.2), (0, 0, 0), 30) 
        self.arm.ikMove('p2p', (0.4, 0.1, -0.26), (0, 0, 0), 30) 
        self.arm.wait_busy()
        
        return self.get_img()

    def render(self):
        pass

    
    def render_gripper_img(self, gripper_img):
        if self.gray_img:
            gray_img = cv2.cvtColor(rgb_gripper, cv2.COLOR_RGB2GRAY)
            cv2.imshow('Gripper Gray Image',gray_img)
            cv2.waitKey(1)

        else:
            gripper_img = cv2.cvtColor(gripper_img, cv2.COLOR_BGR2RGB)
            cv2.imshow('Gripper Image',gripper_img)
            cv2.waitKey(1)


if __name__ == '__main__':
    env = WRSEnv()

    env.reset()
    # env.step(4) # suck
    # for i in range(20):
    #     env.step(1)
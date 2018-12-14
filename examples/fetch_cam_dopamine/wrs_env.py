'''
sudo bash
roslaunch manipulator_h_manager dual_arm.launch

rosrun rosserial_python serial_node.py /dev/wrs/arduino
'''
import cv2
import numpy as np
from arm_control import ArmTask, SuctionTask
import rospy
import os, sys

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__) )+ '/../fetch_camera/'))
from fetch_cam.img_process import ImgProcess, IMG_TYPE, IMG_SHOW
import time

def cut_2_same_wh(img):
    # print('img.shape=', img.shape) 
    shape = np.shape(img)
    print('shape =', shape)
    print('img.shape[0] = ', shape[0], ', img.shape[1] = ', shape[1] )
    if shape[0] > shape[1]:
        left = (shape[0] - shape[1])/2
        right= left + (shape[0] - shape[1]) 
        return img[ left:right,:,: ]
    elif shape[0] < shape[1]:
        left = int(  (shape[1] - shape[0])/2 )
        right= left +  shape[0]
        # print('left = ', left, ', right = ', right)
        return img[:,left:right,: ]
    else:
        return img

# because thread bloack the image catch (maybe), so create the shell class 
class WRSEnv:
    def __init__(self, dis_tolerance = 0.001, step_ds=0.005, 
                        img_type = IMG_TYPE.SEMANTIC,  img_show_type = IMG_SHOW.HIDE):
        self.step_ds = step_ds
        self.img_type  = img_type

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

        self.imp = ImgProcess(img_type, flip=False)
        self.imp.show_type = IMG_SHOW.RAW_PROCESS

    def get_img(self):
        for i in range(5):
            # clear all image
            self.cap.grab()
        ret, rgb_gripper = self.cap.read()
        # rgb_gripper = cv2.cvtColor(rgb_gripper, cv2.COLOR_RGB2BGR)
        if ret==False:
            return None


        cut_img  = cut_2_same_wh(rgb_gripper)
        
        process_img = self.imp.preprocess(cut_img)
        
        return process_img


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
            self.arm.relative_move_pose('line', [0, 0,-0.335] )
            # self.arm.relative_move('line', euler=[90, 0,0], pos = [0, 0,-0.335])
            self.arm.wait_busy()
            rospy.sleep(1)

            is_suck = self.suck.gripped 
            print('Is sucked: {}'.format(is_suck))

            # self.arm.relative_move('line', euler=[90, 0,0], pos = [0, 0,0.335])
            self.arm.relative_move_pose('line', [0, 0, 0.335] )
            self.arm.wait_busy()
            rospy.sleep(1)
            done = True

            reward = 1 if is_suck else -1


        return self.get_img(), reward, done, None


    def reset(self):
        self.suck.gripper_vaccum_off()
        # self.arm.back_home()
        # self.arm.ikMove('p2p', (0.4, 0.1, -0.2), (0, 0, 0), 30) 
        print('reset before robot move')
        self.arm.ikMove('p2p', (0.3, 0.1, -0.26), (90, 0, 0), 30) 
        print('reset after robot move')
        self.arm.wait_busy()
        print('reset after robot move wait busy')
        
        return self.get_img()

    def render(self):
        pass

if __name__ == '__main__':
    env = WRSEnv(img_type = IMG_TYPE.RAW)

    env.reset()
    # env.step(4) # suck
    # for i in range(20):
    #     env.step(1)

    # show img
    while True:
        time.sleep(0.5)
        env.get_img()
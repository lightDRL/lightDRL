import numpy as np
from car_env import CarEnv
from DDPG import DDPG
import matplotlib.pyplot as plt
import time



###ROS library
#import rospy
#import math
#from std_msgs.msg import Float32
#from std_msgs.msg import Int32MultiArray

DISPLAY_REWARD_THRESHOLD=200
env = CarEnv()
s_dim = 2+env.O_LC
a_dim = 1
a_bound = env.action_bound[1]
ddpg = DDPG(a_dim, s_dim, a_bound)
ON_TRAIN=False
t1_ = time.time()
GLOBAL_RUNNING_R=[]

def train():
 
    RENDER=False
    var = 1
    MEMORY_CAPACITY = 10000
    MAX_EPISODES = 1000
    MAX_EP_STEPS = 100
    goood_job=0
    ddpg.restore() ########important
    for i in range(MAX_EPISODES):
        t1 = time.time()
        s=env.reset()
        ep_reward = 0
        for j in range(MAX_EP_STEPS):
            if RENDER:
                env.render()
    
            # Add exploration noise
            a = ddpg.choose_action(s)
            a = np.clip(np.random.normal(a, var), *env.action_bound) 
#            s_, r, done  = env.step(a)
            s_, r, done  = env.step_state(a)
            ddpg.store_transition(s, a, r / 10, s_)
            if ddpg.pointer > MEMORY_CAPACITY:
                var *= .9995    # decay the action randomness
                ddpg.learn()
    
            s = s_.copy()
            
            ep_reward += r
            if j == MAX_EP_STEPS-1 or done==True :
                print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var,'Running time: ', time.time() - t1 )
                if len(GLOBAL_RUNNING_R) == 0: GLOBAL_RUNNING_R.append(ep_reward)
                else: GLOBAL_RUNNING_R.append(GLOBAL_RUNNING_R[-1]*0.9+ep_reward*0.1)
                if ep_reward>30 and i>200 :
                    RENDER = False
                    var=0
                    goood_job+=1
                    print(goood_job)
                else:
                    if goood_job>0:
                        goood_job-=1
                break
        if goood_job>40:
            break
    plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
    plt.xlabel('Episode'); plt.ylabel('Moving reward'); plt.ion(); plt.show()
    print('Running time: ', time.time() - t1_)
    ddpg.save()

"""
def callback(data):
     global state
     line=np.array(data.data,dtype=np.float32)/1000
     main_vec=rospy.get_param('/AvoidChallenge/GoodAngle')
     num_change=main_vec*3-180;
     go_where_x=math.cos(num_change*math.pi/180);
     go_where_y=math.sin(num_change*math.pi/180);
     print(main_vec)
#     line=np.array(data.data)
     #print(line[0:120:6])
     state=[]
     state[0:1]=np.array([go_where_x,go_where_y])
     state[2:11]=line[60:120:6]
     state[12:21]=line[0:60:6]
     state=np.array(state)
     print(state)
def ros_robot():
    pub = rospy.Publisher('/motor_plan', Float32, queue_size=100)
    rospy.Subscriber("/vision/BlackRealDis", Int32MultiArray, callback)
    rospy.init_node('car_strage', anonymous=True)
    rate = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():
        a = ddpg.choose_action(state)
        pub.publish(a)

"""


def eval_():
    ddpg.restore()
    env.render()
    env.viewer.set_vsync(True)
    s = env.reset()
    while True:
        s=env.reset()/400
        ep_reward = 0
        for j in range(100):
            env.render()
            a = ddpg.choose_action(s)
#            s, r, done  = env.step(a)
            s, r, done  = env.step_state(a)
            ep_reward += r
            if j == 99 or done==True :
                    print(' Reward: %i' % int(ep_reward) )
                    break

if ON_TRAIN:
    train()
else:
    eval_()
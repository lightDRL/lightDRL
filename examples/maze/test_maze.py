import gym
import getch
import random
import math
import numpy as np
from maze_env import Maze


env = Maze()    
s = env.reset()
for _ in range(1000):

    char = getch.getch() 
    
    i = int(char)
    # for easy keyboard
    # action: ["South", "North", "East", "West", "Pickup", "Dropoff"]
    #            1       0        2       3       4         5
    print('char = ' + str(char) +', i = ' + str(i))
    a = 0
    if i==2:
        a = 1
    elif i==8:
        a = 0
    elif i==6:
        a = 2
    elif i==4:
        a = 3
    
    s_, r, d, _ = env.step(a)

    env.render()
    print('s={}, a = {}, r = {}, d = {}, s_ = {}'.format(s, a, r, d, s_))

    s = s_
    if d:
        break



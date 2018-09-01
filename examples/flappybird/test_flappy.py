from getch import getch, pause
import random
import math
import numpy as np
import sys
sys.path.append("flappybird_env/")
import wrapped_flappy_bird as game

env = game.GameState()
for _ in range(1000):

    char = getch() 
    
    print('char = ', char)
    i = int(char)
    # for easy keyboard
    # action: ["South", "North", "East", "West", "Pickup", "Dropoff"]
    #            1       0        2       3       4         5
    print('char = ' + str(char) +', i = ' + str(i))
    a = 0

    if i==8:
        a = 1
    else:
        a = 0
    
    s_, r, d, _ = env.step(a)

    # env.render()
    print('a = {}, r = {}, d = {}'.format( a, r, d))

    s = s_
    if d:
        break



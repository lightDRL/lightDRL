# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 15:40:27 2018

# Author:  allengun2000  <https://github.com/allengun2000/>
"""


import numpy as np
import time
import sys
import math
import pyglet


UNIT = 10   # pixels
MAZE_H = 4  # grid height
MAZE_W = 4  # grid width
car_r=20
O_LC=20

###################################################################
class CarEnv(object):
    viewer = None
    action_bound = [-math.pi, math.pi]
    point_bound =[30,370]
    v = 10
    goal_point=np.array([370,200])
    o1_point=np.array([200,280])
    o2_point=np.array([150,150])
    def __init__(self):
        self.car_info=np.zeros(2, dtype=[('a', np.float32), ('i', np.float32)])
        self.car_info['a']=(20,300) 
        self.car_info['i']=(10,20)  #R=20 V=10
        self.ran=math.pi
        self.obs_l=np.zeros(O_LC,dtype=np.float32)
        self.O_LC=O_LC
    def step(self, action):
        done = False
        r = 0.
        action = np.clip(action, *self.action_bound)
        self.car_info['a'][0]+=self.v*math.cos(action)
        self.car_info['a'][1]+=self.v*math.sin(action)

        self.car_info['a']=np.clip(self.car_info['a'],*self.point_bound)
        
        # state


        self.obs_l[:]=self.obs_line()
        s=np.hstack((self.car_info['a']/400,self.obs_l/400))
#        s = self.car_info['a']/400
        # done and reward
        goal_car_dis=np.hypot(*(self.car_info['a']-self.goal_point)) 
        ob1_car_dis=np.hypot(*(self.car_info['a']-self.o1_point)) 
        ob2_car_dis=np.hypot(*(self.car_info['a']-self.o2_point)) 
        r=-goal_car_dis/4000
        if goal_car_dis<25+car_r:
#            done = True
            r += 1.
        if ob1_car_dis<25+car_r:
#            done = True
            r += -1.
        if ob2_car_dis<25+car_r:
#            done = True
            r += -1.
        if self.car_info['a'][0]==30 or self.car_info['a'][0]==370 \
             or self.car_info['a'][1]==30 or self.car_info['a'][1]==370:
#                done = True
                r += -1.
                
            
        return s, r, done, None

    def obs_line(self,car_=None):
        obs_line=[]
        for i in np.linspace(-math.pi, math.pi,O_LC,endpoint=False):
            if car_ is None:
                car_point=self.car_info['a'].copy()
                car_=self.car_info['a'].copy()
            else:
                car_point=car_.copy()
                
            for j in np.linspace(1,400,20):
                car_point[0]=car_[0]+j*math.cos(i)
                car_point[1]=car_[1]+j*math.sin(i)
                car_point=np.clip(car_point,1,399)
                ob1_car_dis=np.hypot(*(car_point-self.o1_point)) 
                ob2_car_dis=np.hypot(*(car_point-self.o2_point)) 
                if ob1_car_dis<25 or ob2_car_dis<25 \
                    or car_point[0]==1 or car_point[0]==399 \
                    or car_point[1]==1 or car_point[1]==399:
                        break
            obs_line.append(j)
        
        return obs_line
            
    def reset(self):
#        self.car_info['a']=(30,300)
        self.car_info['a']=np.random.rand(2)*(1,380)+30
        self.obs_l[:]=self.obs_line()
        s=np.hstack((self.car_info['a']/400,self.obs_l/400))
        return s
    
    def render(self):
        if self.viewer is None:
            self.viewer = Viewer(self.goal_point,self.car_info['a'],self.o1_point,self.o2_point,self.obs_l)
        self.viewer.render()
    def sample_action(self):
        self.ran-=0.1
        if self.ran<-math.pi:
            self.ran=math.pi
#        return np.random.rand(2)+2  # two radians
        return self.ran
        

class Viewer(pyglet.window.Window):
    bar_thc = 5

    def __init__(self,goal_point,car_point,o1_point,o2_point,obs_line):
        # vsync=False to not use the monitor FPS, we can speed up training
        super(Viewer, self).__init__(width=400, height=400, resizable=True, caption='gooood_car', vsync=False)
        self.car_point_=car_point
        self.obs_line=obs_line

        o1_v=np.hstack([o1_point-25,o1_point[0]-25,o1_point[1]+25,o1_point+25,o1_point[0]+25,o1_point[1]-25])
        o2_v=np.hstack([o2_point-25,o2_point[0]-25,o2_point[1]+25,o2_point+25,o2_point[0]+25,o2_point[1]-25])
        goal_v=np.hstack([goal_point-25,goal_point[0]-25,goal_point[1]+25,goal_point+25,goal_point[0]+25,goal_point[1]-25])
        pyglet.gl.glClearColor(1, 1, 1, 1)
        #        GL_POINTS  GL_LINES GL_LINE_STRIP GL_LINE_LOOP GL_POINTS
        self.batch = pyglet.graphics.Batch()    # display whole batch at once
        self.goal = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,    # 4 corners
            ('v2f', goal_v),
            ('c3B', (86, 109, 249) * 4))    # color
        self.obs_1 = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', o1_v),
            ('c3B', (249, 86, 86) * 4,))
        self.obs_2 = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', o2_v ),
            ('c3B', (249, 86, 86) * 4,))    # color
        car_dot=self.makeCircle(200,car_r,*self.car_point_)
        self.car = self.batch.add(
            int(len(car_dot)/2), pyglet.gl.GL_LINE_LOOP, None,
            ('v2f', car_dot), ('c3B', (0, 0, 0) * int(len(car_dot)/2)))

        line_dot=self.linedot()
        self.o_line=self.batch.add(
            int(len(line_dot)/2), pyglet.gl.GL_LINES, None,
            ('v2f', line_dot), ('c3B', (0, 0, 0) * int(len(line_dot)/2)))

    def makeCircle(self,numPoints,r,c_x,c_y):
        verts = []
        for i in range(numPoints):
            angle = math.radians(float(i)/numPoints * 360.0)
            x = r*math.cos(angle) + c_x
            y = r*math.sin(angle) + c_y
            verts += [x,y]
        return verts
    
    def linedot(self):
        line_dot_v=[]
        for i, j in zip(np.linspace(-math.pi, math.pi,O_LC,endpoint=False),range(O_LC)):
            l_dot=self.car_point_.copy()
            line_dot_v.append(l_dot.copy())
            l_dot[0]+=self.obs_line[j]*math.cos(i)
            l_dot[1]+=self.obs_line[j]*math.sin(i)
            line_dot_v.append(l_dot)
        return np.hstack(line_dot_v)
    
    def render(self):
        self._update_car()
        self.switch_to()
        self.dispatch_events()
        self.dispatch_event('on_draw')
        self.flip()
        
    def on_draw(self):
        self.clear()
        self.batch.draw()
        

    def _update_car(self):
        car_dot=self.makeCircle(200,car_r,*self.car_point_)
        self.car.vertices = car_dot 
        line_dot=self.linedot()
        self.o_line.vertices=line_dot
    
    def on_close(self):
        self.close()
        
if __name__ == '__main__':
    env = CarEnv()
    s=env.reset()
    while True:
        env.render()
        env.step(env.sample_action())
#    pyglet.on_close()
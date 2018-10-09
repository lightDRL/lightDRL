import numpy as np


max_dis = 0.1
less_step = 20
sum_r = 0
dis_range=  np.linspace(max_dis, 0.005, less_step)
print(dis_range)
for arm_2_obj_xy in dis_range:
    if arm_2_obj_xy  <= max_dis:
        max_dis = max_dis
        tmp = (max_dis - arm_2_obj_xy ) / max_dis
        
        r =  (max_dis - arm_2_obj_xy ) / max_dis *0.01
        # print('tmp = ', tmp,', arm_2_obj_xy=',arm_2_obj_xy,',(max_dis - arm_2_obj_xy ) =', (max_dis - arm_2_obj_xy ) ,',r=',r)
        print(f'dis = {arm_2_obj_xy:5.4f}, r={r:5.4f}')
        sum_r+=r




discout= (100-less_step)*(0.001)

print('sum_r = ', sum_r)

print('discout = ', discout)


'''
sum_r = 0
dis_range=  np.linspace(0.1, 0., 21)
print(dis_range)
for arm_2_obj_xy in dis_range:
    if arm_2_obj_xy  <= 0.1:
        max_dis = 0.1
        tmp = (max_dis - arm_2_obj_xy ) / max_dis
        print('tmp = ', tmp,', arm_2_obj_xy=',arm_2_obj_xy,',(max_dis - arm_2_obj_xy ) =', (max_dis - arm_2_obj_xy ) )
        r =  (max_dis - arm_2_obj_xy ) / max_dis *0.01
        sum_r+=r

'''

'''
sum_r = 0
dis_range=  np.linspace(0.075, 0., 16)
print(dis_range)
for arm_2_obj_xy in dis_range:
    if arm_2_obj_xy  <= 0.075:
        max_dis = 0.075
        tmp = (max_dis - arm_2_obj_xy ) / max_dis
        print('tmp = ', tmp,', arm_2_obj_xy=',arm_2_obj_xy,',(max_dis - arm_2_obj_xy ) =', (max_dis - arm_2_obj_xy ) )
        r =  (max_dis - arm_2_obj_xy ) / max_dis *0.01
        sum_r+=r


'''
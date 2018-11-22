import cv2
import numpy as np
import os


MARGIN = 30
IMG_W_H = 256
# because thread bloack the image catch (maybe), so create the shell class 
class AugmentationImage:

    save_id = 0

    def __init__(self, table_path = 'table.png', cube_path = 'cube_shadow.png', cube_label_path = 'cube_shadow_label.png'):
        # self.table = cv2.imread(table_path)
        # self.cube = cv2.imread(cube_path)
        # self.table_label = np.zeros(self.table.shape)
        # self.cube_label = cv2.imread(cube_label_path)
        # print("self.table.shape = ", self.table.shape)
        # print("self.cube.shape = ", self.cube.shape)

        self.set_image(table_path = table_path, cube_path = cube_path, cube_label_path = cube_label_path)

    def set_image(self, table_path = 'table.png', cube_path = 'cube_shadow.png', cube_label_path = 'cube_shadow_label.png'):
        self.table = cv2.imread(table_path)
        self.cube = cv2.imread(cube_path)
        self.table_label = np.zeros(self.table.shape)
        self.cube_label = cv2.imread(cube_label_path)

    def paste_cube(self, cube_img_x, cube_img_y, rotate_deg= 30, scale= 1):
        
        cube_new = self.cube.copy()

        cube_new = cv2.resize(cube_new, \
                            (int(cube_new.shape[0]*scale),int(cube_new.shape[1]*scale)), \
                            interpolation=cv2.INTER_AREA)        
        rows = cube_new.shape[0]
        cols = cube_new.shape[1]
        M = cv2.getRotationMatrix2D((cols/2,rows/2),rotate_deg,1)
        cube_new = cv2.warpAffine(cube_new,M,(cols,rows))

        
        # print('self.cube.shape = ', self.cube.shape)
        # print('cube_new.shape = ', cube_new.shape)

        cube_h = cube_new.shape[0]
        cube_w = cube_new.shape[1]
        self.state = self.table.copy()
        if cube_img_x >= self.table.shape[0] or cube_img_y >= self.table.shape[1] or \
           (cube_img_x+cube_h) < 0 or  (cube_img_y+cube_w) < 0 :
            return 
        
        paste_low_h =  -cube_img_x  if cube_img_x < 0 else 0
        paste_low_w =  -cube_img_y  if cube_img_y < 0 else 0
        cube_img_x = 0 if cube_img_x < 0 else cube_img_x
        cube_img_y = 0 if cube_img_y < 0 else cube_img_y
        paste_h = cube_new.shape[0] if (cube_img_x +  cube_new.shape[0]) <= self.table.shape[0] else ( self.table.shape[0]- cube_img_x)  
        paste_w = cube_new.shape[1] if (cube_img_y +  cube_new.shape[1]) <= self.table.shape[1] else ( self.table.shape[1]- cube_img_y)  

        s_min_x = cube_img_x
        s_max_x = cube_img_x+(paste_h-paste_low_h)
        s_min_y = cube_img_y
        s_max_y = cube_img_y + (paste_w-paste_low_w)
        self.state[s_min_x:s_max_x, s_min_y:s_max_y, :]  = cube_new[paste_low_h:paste_h,paste_low_w:paste_w,:] #self.cube[:,:,:] 

        for i in range(s_min_x, s_max_x):
            for j in range(s_min_y, s_max_y):
                if sum(self.state[i][j]) == 0:
                    self.state[i,j,:] = self.table[i,j,:] 

        # self.label = self.table_label.copy()
        # self.label[cube_img_x:cube_img_x+(paste_h-paste_low_h), cube_img_y:cube_img_y + (paste_w-paste_low_w), :]  =self.cube_label[paste_low_h:paste_h,paste_low_w:paste_w,:] #self.cube[:,:,:] 
        
        cube_label_new = cv2.warpAffine(self.cube_label,M,(cols,rows))

        self.label = self.table_label.copy()
        self.label[cube_img_x:cube_img_x+(paste_h-paste_low_h), cube_img_y:cube_img_y + (paste_w-paste_low_w), :]  = cube_label_new[paste_low_h:paste_h,paste_low_w:paste_w,:] #self.cube[:,:,:] 

        if self.state.shape[0]!=IMG_W_H or self.state.shape[1]!=IMG_W_H:
            self.state = cv2.resize(self.state, (IMG_W_H, IMG_W_H), interpolation=cv2.INTER_AREA)
            self.label = cv2.resize(self.label, (IMG_W_H, IMG_W_H), interpolation=cv2.INTER_AREA)


    def save(self, img_dir, label_dir, save_id):
        f_path = f'{img_dir}/{save_id:03d}.png'
        f_label_path = f'{label_dir}/{save_id:03d}.png'
        cv2.imwrite(f_path,self.state)
        cv2.imwrite(f_label_path,self.label)




    def recreate_dir(self, d):
        import shutil,os
        if os.path.isdir(d):
            shutil.rmtree(d)
        os.makedirs(d)

    def run_gen(self, img_dir, label_dir, s_id = 0,  num = 300):
        if s_id == 0:
            self.recreate_dir(img_dir)
            self.recreate_dir(label_dir)
        for i in range(s_id, s_id+num):
            rand_cube_x = np.random.randint(MARGIN, self.table_label.shape[0] -MARGIN) 
            rand_cube_y = np.random.randint(MARGIN, self.table_label.shape[1] -MARGIN)
            rotate_deg =  np.random.randint(0, 90)
            scale = np.random.uniform(0.8,1.2)
            print('Generate x={:3d},y={:3d}, roate={:2d}, scale={:3.2f} '.format( rand_cube_x, rand_cube_y, rotate_deg, scale) + f', save -> {img_dir}/{i:03d}.png' )
            self.paste_cube(rand_cube_x, rand_cube_y, rotate_deg, scale)
            self.save(img_dir, label_dir, i)



# AugmentationImage().run_gen(img_dir='OneCube/red_cube2', label_dir='OneCube/red_cube_label')

# train 0~199
aug = AugmentationImage(table_path = 'table.png', cube_path = 'cube_shadow.png', cube_label_path = 'cube_shadow_label.png')
aug.run_gen(img_dir='OneCube/train', label_dir='OneCube/trainannot', num=200)

# train 200~399
aug.set_image(table_path = 'sim_table.png', cube_path = 'sim_cube_2.png', cube_label_path = 'sim_cube_2_label.png')
aug.run_gen(img_dir='OneCube/train', label_dir='OneCube/trainannot', s_id = 200, num=200)

# test 0~50
aug.set_image(table_path = 'sim_table.png', cube_path = 'sim_cube_2.png', cube_label_path = 'sim_cube_2_label.png')
aug.run_gen(img_dir='OneCube/test', label_dir='OneCube/testannot', num=100)


# test 50~99
aug.set_image(table_path = 'table.png', cube_path = 'cube_shadow.png', cube_label_path = 'cube_shadow_label.png')
aug.run_gen(img_dir='OneCube/test', label_dir='OneCube/testannot', s_id = 100, num=100)


# val 0~50
aug.set_image(table_path = 'sim_table.png', cube_path = 'sim_cube_2.png', cube_label_path = 'sim_cube_2_label.png')
aug.run_gen(img_dir='OneCube/val', label_dir='OneCube/valannot', num=100)


# val 50~99
aug.set_image(table_path = 'table.png', cube_path = 'cube_shadow.png', cube_label_path = 'cube_shadow_label.png')
aug.run_gen(img_dir='OneCube/val', label_dir='OneCube/valannot', s_id = 100, num=100)


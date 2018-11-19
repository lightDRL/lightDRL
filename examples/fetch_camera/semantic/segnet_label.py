import argparse

from .VGGSegnet import VGGSegnet

from keras.models import load_model
import glob
import cv2
import numpy as np
import random
import os

'''
python segnet_label.py --input=test.png --output=test_label.png
'''


class SegnetLabel:
	def __init__(self, n_classes, input_height, input_width, save_weights_path, epoch_number):

		self.limit_gpu_memory()

		self.m = VGGSegnet( n_classes , input_height=input_height, input_width=input_width   )
		self.m.load_weights(  save_weights_path + "." + str(  epoch_number )  )
		self.m.compile(loss='categorical_crossentropy',
			optimizer= 'adadelta' ,
			metrics=['accuracy'])

		self.n_classes = n_classes
		self.input_height = input_height
		self.input_width = input_width


	def limit_gpu_memory(self):
		# ---- limit  GPU memory resource-----#
		import tensorflow as tf
		from keras.backend.tensorflow_backend import set_session
		config = tf.ConfigProto()
		config.gpu_options.per_process_gpu_memory_fraction = 0.4
		set_session(tf.Session(config=config))

	def predict(self, inName, img = None):
		X = self.getImageArr(inName , self.input_width  , self.input_height , ordering='None', img=img )
		pr = self.m.predict( np.array([X]) )[0]
		pr = pr.reshape(( self.m.outputHeight ,  self.m.outputWidth , self.n_classes ) ).argmax( axis=2 )
		return pr

	def getImageArr(self, path , width , height , imgNorm="sub_mean" , ordering='channels_first', img=None):

		try:
			if img==None:
				img = cv2.imread(path, 1)
			
			if imgNorm == "sub_and_divide":
				img = np.float32(cv2.resize(img, ( width , height ))) / 127.5 - 1
			elif imgNorm == "sub_mean":
				img = cv2.resize(img, ( width , height ))
				img = img.astype(np.float32)
				img[:,:,0] -= 103.939
				img[:,:,1] -= 116.779
				img[:,:,2] -= 123.68
			elif imgNorm == "divide":
				img = cv2.resize(img, ( width , height ))
				img = img.astype(np.float32)
				img = img/255.0

			if ordering == 'channels_first':
				img = np.rollaxis(img, 2, 0)

			# print('img -> ', img.shape)
			return img
		except Exception as e:
			print(path , e)
			img = np.zeros((  height , width  , 3 ))
			if ordering == 'channels_first':
				img = np.rollaxis(img, 2, 0)
			return img

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--save_weights_path", type = str, default='weights/ex1'  )
	parser.add_argument("--epoch_number", type = int, default = 10 )
	parser.add_argument("--input", type = str , default = "test.png")
	parser.add_argument("--output", type = str , default = "test_label.png")
	parser.add_argument("--input_height", type=int , default = 224  )
	parser.add_argument("--input_width", type=int , default = 224 )
	parser.add_argument("--n_classes", type=int , default = 2)

	args = parser.parse_args()


	s = SegnetLabel(args.n_classes, args.input_height , args.input_width, args.save_weights_path, args.epoch_number )
	output = s.predict(args.input)

	print('output.shape = ', output.shape)

	seg_img = output*255.0
	cv2.imwrite(  args.output , seg_img )
	cv2.imshow('label', seg_img)


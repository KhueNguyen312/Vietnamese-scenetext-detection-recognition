import time
from typing import Text
import torch
import subprocess
import os
from model import EAST
from PIL import Image, ImageDraw
from detect import detect, plot_boxes
import numpy as np
import shutil

import sys
import cv2
sys.path.insert(1, '../SEED')

from predict import predict_text_by_img, get_model

#read img
def drop_img_by_boudingbox(img_path, cord):
	img = cv2.imread(img_path, cv2.IMREAD_COLOR)
	img_height, img_width, _ = img.shape
	x_min = np.int(round(min(cord[0], cord[2], cord[4], cord[6])))
	x_max = np.int(round(max(cord[0], cord[2], cord[4], cord[6])))
	y_min = np.int(round(min(cord[1], cord[3], cord[5], cord[7])))
	y_max = np.int(round(max(cord[1], cord[3], cord[5], cord[7])))
	if len(img.shape) == 3:
		img_cropped = img[ y_min:y_max:1, x_min:x_max:1, :]
	else:
		img_cropped = img[ y_min:y_max:1, x_min:x_max:1]
	
	#cv2.imshow(img_cropped)
	# try:
	# 	cv2.imshow('',img_cropped)
	# except:
	# 	print('Invalid Cropped Images')
	# 	img_cropped = img
	return img_cropped

def detect_dataset(model,reg_model ,device, test_img_path, submit_path):
	'''detection on whole dataset, save .txt results in submit_path
	Input:
		model        : detection model
		device       : gpu if gpu is available
		test_img_path: dataset path
		submit_path  : submit result for evaluation
	'''
	img_files = os.listdir(test_img_path)
	img_files = sorted([os.path.join(test_img_path, img_file) for img_file in img_files])
	
	for i, img_file in enumerate(img_files):
		print('evaluating {} image'.format(i), end='\r')
		boxes = detect(Image.open(img_file), model, device)
		seq = []
		# plot_img = plot_boxes(Image.open(img_file), boxes)	
		# plot_img.save('./'+ os.path.basename(img_file))
		if boxes is not None:
			for box in boxes:
				box = [1 if i <= 0 else i for i in box]
				cropped_img = drop_img_by_boudingbox(img_file, box)
				predicted_label = predict_text_by_img(cropped_img, reg_model)
				if not predicted_label:
					predicted_label = '###'
				seq.extend([','.join([str(int(b)) for b in box[:-1]]) + ',' + predicted_label.upper() + '\n'])
			#seq.extend([','.join([str(int(b)) for b in box[:-1]]) + '\n' for box in boxes])
		with open(os.path.join(submit_path, os.path.basename(img_file) + '.txt'), 'w',encoding="utf-8") as f:
			f.writelines(seq)

def eval_model(detect_model_name, reg_model_name ,test_img_path, submit_path, save_flag=True):
	if os.path.exists(submit_path):
		shutil.rmtree(submit_path) 
	os.mkdir(submit_path)

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model = EAST(False).to(device)
	if torch.cuda.is_available():
		model.load_state_dict(torch.load(detect_model_name))
	else:
		model.load_state_dict(torch.load(detect_model_name, map_location ='cpu'))
	model.eval()

	reg_model = get_model(reg_model_name)
	reg_model.eval()
	
	start_time = time.time()
	detect_dataset(model,reg_model ,device, test_img_path, submit_path)
	os.chdir(submit_path)
	res = subprocess.getoutput('zip -q submit.zip *.txt')
	res = subprocess.getoutput('mv submit.zip ../')
	os.chdir('../')
	res = subprocess.getoutput('python ./evaluate/script.py –g=./evaluate/gt.zip –s=./submit.zip')
	print(res)
	# os.remove('./submit.zip')
	print('eval time is {}'.format(time.time()-start_time))	

	if not save_flag:
		shutil.rmtree(submit_path)


if __name__ == '__main__': 
	detect_model_name = '/content/drive/MyDrive/Colab Notebooks/Trained Models/pretrained_east/model_epoch_14.pth'
	reg_model_name = '/content/drive/MyDrive/Colab Notebooks/logs/se_aster/model_best.pth.tar'
	#test_img_path = os.path.abspath('../vietnamese/test_imgs')
	test_img_path = '/content/drive/MyDrive/Colab Notebooks/Dataset/vietnamese/TestA'
	submit_path = './submit'
	eval_model(detect_model_name, reg_model_name ,test_img_path, submit_path)

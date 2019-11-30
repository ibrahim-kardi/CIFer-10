"""
Author: Ibrahim, rayhan

Desc: configuration file for kaggle cifer-10
"""
import os

train_sample_num = 50000
test_sample_num = 300000
nb_class = 10
img_size = 32
img_channel = 3
img_shape = (img_size, img_size, img_channel)
lr = 0.001
batch_size = 64
nb_epochs = 50

def root_path():
	return os.path.dirname(__file__)

def checkpoint_path():
	return os.path.join(root_path(),"checkpoint")

def dataset_path():
	return os.path.join(root_path(),"dataset")

def src_path():
	return os.path.join(root_path(),"src")

def submission_path():
	return os.path.join(root_path(),"submission")

def output_path():
	return os.path.join(root_path(),"output")

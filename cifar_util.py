import pickle
import os
import numpy as np
import cv2

from PIL import Image

import random

def SaveImage(image_array,im_output_path):
	save_im = image_array
	save_im = (255.0 / save_im.max() * (save_im - save_im.min())).astype(np.uint8)
	im = Image.fromarray(save_im)
	im.save(im_output_path)

def LabelToOneHot(label,num_classes):
	y = [0]*num_classes
	y[label] = 1

	return np.array(y)

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict


def FlatCifarToImageArray(flat_image_array):
	channel_size = flat_image_array.shape[0] / 3

	r = flat_image_array[:channel_size].reshape(32,32)
	g = flat_image_array[channel_size:2*channel_size].reshape(32,32)
	b = flat_image_array[2*channel_size:].reshape(32,32)

	return np.dstack([r,g,b])



def LoadCifarData(num_im_to_load):
	cifar_dir = "cifar_data"

	data_set = "cifar-10-batches-py"

	training_dir = "training"
	test_dir = "test"

	training_path = os.path.join(cifar_dir,data_set,training_dir)
	test_path = os.path.join(cifar_dir,data_set,test_dir)

	training_batch_files = os.listdir(training_path)

	cifar_images = []
	cifar_labels = []

	for input_i in range(num_im_to_load):
		pass
	

def LoadCifarDataFromImages(num_im_to_load, train = False):

	cifar_dir = "cifar_data"

	if(train):
		images_path = os.path.join(cifar_dir,"cifar_10_images","train")
	else:
		images_path = os.path.join(cifar_dir,"cifar_10_images","test")

	image_names = os.listdir(images_path)

	selected_images = random.sample(image_names,num_im_to_load)

	cifar_images = []
	cifar_labels = []

	for selected_image in selected_images:
		image_path = os.path.join(images_path,selected_image)
		cifar_images.append(cv2.imread(image_path)[...,::-1])

		cifar_labels.append( LabelToOneHot(int(selected_image.split("_")[-1].replace(".jpg","")),10) )


	return np.array(cifar_images), np.array(cifar_labels)



def LoadGSCifarData(num_im_to_load, train = False):
	xs, ys, = LoadCifarDataFromImages(num_im_to_load,train)

	gs_xs = []

	for image_i in range(num_im_to_load):
		image = xs[image_i,:,:,:]

		gs_image = np.dot(image[...,:3],[0.299,0.587,0.114])/255
		gs_image = gs_image.reshape(32,32,1)
		gs_xs.append(gs_image)

	return np.array(gs_xs), ys








if __name__ == '__main__':
	training = False
	test = True

	cifar_dir = "cifar_data"

	data_set = "cifar-10-batches-py"

	training_dir = "training"
	test_dir = "test"

	training_path = os.path.join(cifar_dir,data_set,training_dir)
	test_path = os.path.join(cifar_dir,data_set,test_dir)

	training_batch_files = os.listdir(training_path)
	test_batch_files = os.listdir(test_path)
	

	output_dir = os.path.join(cifar_dir,"cifar_10_images")

	if(not os.path.exists(output_dir)):
		os.mkdir(output_dir)

	if(training):
		output_save_path = os.path.join(output_dir,"training")
		if(not os.path.exists(output_save_path)):
			os.mkdir(output_save_path)

		for batch_i in range(len(training_batch_files))[:]:
			training_batch_file = training_batch_files[batch_i]
			batch_file_path = os.path.join(training_path,training_batch_file)
			training_batch_dict = unpickle(batch_file_path)

			
			batch_size = training_batch_dict["data"].shape[0]
			for image_i in range(batch_size)[:]:
				gt_label = training_batch_dict["labels"][image_i]
				flat_image_array = training_batch_dict["data"][image_i,:]
				
				image_array = FlatCifarToImageArray(flat_image_array)

				im_output_path = os.path.join(output_save_path,str(batch_i)+"_"+str(image_i)+"_"+str(gt_label)+".jpg")
				SaveImage(image_array,im_output_path)
	if(test):
		output_save_path = os.path.join(output_dir,"test")
		if(not os.path.exists(output_save_path)):
			os.mkdir(output_save_path)

		for batch_i in range(len(test_batch_files))[:]:
			test_batch_file = test_batch_files[batch_i]
			batch_file_path = os.path.join(test_path,test_batch_file)
			test_batch_dict = unpickle(batch_file_path)

			
			batch_size = test_batch_dict["data"].shape[0]
			for image_i in range(batch_size)[:]:
				gt_label = test_batch_dict["labels"][image_i]
				flat_image_array = test_batch_dict["data"][image_i,:]
				
				image_array = FlatCifarToImageArray(flat_image_array)

				im_output_path = os.path.join(output_save_path,str(batch_i)+"_"+str(image_i)+"_"+str(gt_label)+".jpg")
				SaveImage(image_array,im_output_path)

	# images, labels = LoadCifarDataFromImages(2)

	# print(images.shape)
	# print(labels.shape)

	# print(labels[0])

	# #SaveImage(images[0,:,:,:],"test_output.jpg")


	# images, labels = LoadGSCifarData(2)

	# print(images.shape)
	# print(labels.shape)

	# print(labels[0])

	# SaveImage(images[0,:,:],"gs_test_output.jpg")
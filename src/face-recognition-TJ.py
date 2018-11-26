from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import glob

from scipy import misc
import sys
import os
import copy
import argparse
import time
import cv2
import tensorflow as tf
import facenet
import align.detect_face
from os import listdir
import os
from os.path import isfile, join

# load_and_align_data(['/Users/jintao01/Documents/facenet/dataset/lfw/raw/Aaron_Eckhart/Aaron_Eckhart_0001.jpg'], 160, 44, 1.0)
def load_and_align_data(image_paths, image_size = 160, margin = 44, gpu_memory_fraction = 1.0):
    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor
    
    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
  
    tmp_image_paths=copy.copy(image_paths)
    img_list = []
    t0 = time.time()
    for image in tmp_image_paths:
        img = misc.imread(os.path.expanduser(image), mode='RGB')
        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        if len(bounding_boxes) < 1:
          image_paths.remove(image)
          print("can't detect face, remove ", image)
          continue
        det = np.squeeze(bounding_boxes[0,0:4])
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0]-margin/2, 0)
        bb[1] = np.maximum(det[1]-margin/2, 0)
        bb[2] = np.minimum(det[2]+margin/2, img_size[1])
        bb[3] = np.minimum(det[3]+margin/2, img_size[0])
        print(bb)
        cv2.rectangle(img, (bb[0], bb[1]), (bb[2], bb[3]), (0,255,0), 2)
        RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imshow(os.path.splitext(os.path.basename(image))[0], RGB_img)   # show the detected image in RGB
        k = cv2.waitKey(0)
        cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
        aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
        prewhitened = facenet.prewhiten(aligned) #It subtracts the average and normalizes the range of the pixel values of input images.
        img_list.append(prewhitened)
    t1 = time.time()
    total = t1-t0
    print("align face running time = " + str(total) + " sec") #in seconds
    images = np.stack(img_list)
    return images  # list of normalized cropped images

# database = initialize('/Users/jintao01/Documents/facematch/20180204-160909', '/Users/jintao01/Documents/facenet/data/images', 160, 44, 1.0)
def initialize(model_path, path, image_size = 160, margin = 44, gpu_memory_fraction = 1.0):
    # load all the images of verified employees into the database
    # database is a dictionary {person:face embedding}
	database = {}   
	person_list = os.listdir(path) # path contains images for each person 
	print(person_list)
	person_files = [join(path, f) for f in listdir(path) if isfile(join(path, f)) and f != '.DS_Store']
	print(person_files)
	images = load_and_align_data(person_files, image_size, margin, gpu_memory_fraction)
	with tf.Graph().as_default():
		with tf.Session() as sess:
			# Load the model
			facenet.load_model(model_path)   
			# Get input and output tensors
			images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
			embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
			phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")           
			t0 = time.time()
			# Run forward pass to calculate embeddings           
			feed_dict = { images_placeholder: images, phase_train_placeholder:False }
			emb = sess.run(embeddings, feed_dict=feed_dict)
			print(len(emb[0]))
			t1 = time.time()
			total = t1-t0
			print("get embedding running time = " + str(total) + " sec") #in seconds      
	for idx, person_file in enumerate(person_files):
		print(idx)
		print(person_file)
		identity = os.path.splitext(os.path.basename(person_file))[0]
		database[identity] = emb[idx]
	return database

# given a new image, tell who is he/she or unknown person 
def recognize_face(unknown_face_embedding, database):
	min_dist = 1000
	identity = None
	# Loop over the database dictionary's names and encodings.
	for (name, emb) in database.items():
		# Compute L2 distance between the target "encoding" and the current "emb" from the database.
		dist = np.sqrt(np.sum(np.square(np.subtract(unknown_face_embedding, emb))))
		print('distance for %s is %1.4f' % (name, dist))
		# If this distance is less than the min_dist, then set min_dist to dist, and identity to name
		if dist < min_dist:
			min_dist = dist
			identity = name
	# set the threshold
	if(min_dist > 1.1): 
		similar = identity
		identity = "unknown but is most similar to " + similar
	return identity,min_dist

# person_file is ['/full/path/****.jpg']
def get_single_embedding(person_file, model_path, image_size = 160, margin = 44, gpu_memory_fraction = 1.0):
	images = load_and_align_data(person_file, image_size, margin, gpu_memory_fraction)
	with tf.Graph().as_default():
		with tf.Session() as sess:
			# Load the model
			facenet.load_model(model_path)   
			# Get input and output tensors
			images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
			embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
			phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")           
			t0 = time.time()
			# Run forward pass to calculate embeddings           
			feed_dict = { images_placeholder: images, phase_train_placeholder:False }
			emb = sess.run(embeddings, feed_dict=feed_dict)
	print(emb[0])
	return emb[0]

# '/Users/jintao01/Documents/Arm-Buddies/Yue_Without_Glasses.jpg'
def rotate(image_path):
	#loading the image into a numpy array
	img = cv2.imread(image_path)
 
	#rotating the image
	rotated_90_clockwise = np.rot90(img) #rotated 90 deg once
	rotated_180_clockwise = np.rot90(rotated_90_clockwise)
	rotated_270_clockwise = np.rot90(rotated_180_clockwise)
 
	#displaying all the images in different windows(optional)
	cv2.imshow('Original', img)
	cv2.imshow('90 deg', rotated_90_clockwise)
	cv2.imshow('Inverted', rotated_180_clockwise)
	cv2.imshow('270 deg', rotated_270_clockwise)
 
	k = cv2.waitKey(0)
	if (k == 27): #closes all windows if ESC is pressed
		cv2.destroyAllWindows()

# file = '/Users/jintao01/Documents/Arm-Buddies/too/Yue_Without_Glasses.jpg'
# file = '/Users/jintao01/Documents/Arm-Buddies/unknown_pics/6.jpg'
# image = cv2.imread(file)
# image = image_resize(image, height = 600)
# cv2.imwrite("/Users/jintao01/Documents/Arm-Buddies/face-yue.jpg", image)

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
	# initialize the dimensions of the image to be resized and
	# grab the image size
	dim = None
	(h, w) = image.shape[:2]
	# if both the width and height are None, then return the
	# original image
	if width is None and height is None:
		return image
	# check to see if the width is None
	if width is None:
		# calculate the ratio of the height and construct the
		# dimensions
		r = height / float(h)
		dim = (int(w * r), height)
	# otherwise, the height is None
	else:
		# calculate the ratio of the width and construct the
		# dimensions
		r = width / float(w)
		dim = (width, int(h * r))
	# resize the image
	resized = cv2.resize(image, dim, interpolation = inter)
	# return the resized image
	return resized

# for iphone high-resolution image preprocessing (use the same-quality images for database and testing)
def simplify(file, target):
	image = cv2.imread(file)
	image = image_resize(image, height = 600)
	cv2.imwrite(target, image)


if __name__ == '__main__':
#	simplify('/Users/jintao01/Documents/Arm-Buddies/Vikas.jpg','/Users/jintao01/Documents/Arm-Buddies/Vikas1.jpg')
#	simplify('/Users/jintao01/Documents/Arm-Buddies/Saina.jpg','/Users/jintao01/Documents/Arm-Buddies/Saina1.jpg')
#	simplify('/Users/jintao01/Documents/Arm-Buddies/unknown_pics/5.jpg','/Users/jintao01/Documents/Arm-Buddies/5.jpg')
	database = initialize('/Users/jintao01/Documents/facematch/20180204-160909', '/Users/jintao01/Documents/Arm-Buddies', 160, 44, 1.0)
	# print(database)
	path = '/Users/jintao01/Documents/Arm-Buddies/unknown_pics'
	for i in range(1, 10):
		person_file = [join(path, str(i) + '.jpg')] #[join(path, f) for f in listdir(path) if isfile(join(path, f)) and f != '.DS_Store']
		unknown_face_embedding = get_single_embedding(person_file, '/Users/jintao01/Documents/facematch/20180204-160909', 160, 44, 1.0) #20181119-002059   #20180204-160909
		print(recognize_face(unknown_face_embedding, database))
		time.sleep(5)

#same resolution quality, frontal (instead of tilted face), clear background should work
# for better result, train from scatch using customer's dataset
# for conveinience, using the pretrained model means no need to retrain the model when new faces are added

# add the tag to the detected face if needed

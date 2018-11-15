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
        cv2.imshow(os.path.splitext(os.path.basename(image))[0], img)   # show the detected image
        k = cv2.waitKey(1)
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
	if(min_dist > 1.05): 
		identity = "unknown"
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

if __name__ == '__main__':
	database = initialize('/Users/jintao01/Documents/facematch/20180204-160909', '/Users/jintao01/Documents/Arm-Buddies', 160, 44, 1.0)
	# print(database)
	person_file = ['/Users/jintao01/Documents/Arm-Buddies/unknown_pics/1.jpg']
	unknown_face_embedding = get_single_embedding(person_file, '/Users/jintao01/Documents/facematch/20180204-160909', 160, 44, 1.0)
	print(recognize_face(unknown_face_embedding, database))



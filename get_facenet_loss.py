from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import argparse
import importlib
import time

import os
import facenet
import numpy as np
from sklearn.datasets import load_files
import tensorflow as tf
from scipy.misc import imsave, imread, imresize
from six.moves import xrange
# output_dir="output1/"
batch_size=64
trained_model_dir="~/error_concealment_GAN/Divya/real-face-recog/facenet/20170512-110547/"
# data_dir="output/"


def main():

	with tf.Graph().as_default():

		with tf.Session() as sess:

			# create output directory if it doesn't exist
			#if not os.path.isdir(output_dir):
			#	os.makedirs(output_dir)

			# load the model
			print("Loading trained model...\n")
			meta_file, ckpt_file = facenet.get_model_filenames(os.path.expanduser(trained_model_dir))
			facenet.load_model(trained_model_dir)

			# Get input and output tensors
			images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
			embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
			phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

			# a=np.random.rand(8,160,160,3)  #feed the images here
			# Run forward pass to calculate embeddings

			i_list = ['facenet_i/' + x for x in os.listdir('facenet_i/')]
			o_list = [x.replace('facenet_i', 'facenet_o') for x in i_list]

			i_ims, o_ims = [], []
			# batch
			loss = []
			for i in range(len(i_list)):
				i_ims.append(imresize(imread(i_list[i]), (160, 160, 3)))
				o_ims.append(imresize(imread(o_list[i]), (160, 160, 3)))

				if i%batch_size == batch_size - 1:
					
					i_emb = sess.run(embeddings,feed_dict={images_placeholder:i_ims,phase_train_placeholder:False})
					o_emb = sess.run(embeddings,feed_dict={images_placeholder:o_ims,phase_train_placeholder:False})		

					loss.append(np.mean((i_emb - o_emb)**2))
					i_ims, o_ims = [], []			

			print ('loss is :', np.mean(loss))

			# print(sess.run(embeddings,feed_dict={images_placeholder:a,phase_train_placeholder:False}))
			

if __name__ == "__main__":
	main()

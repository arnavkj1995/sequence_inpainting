from __future__ import division
import os
import pickle
import sys
import time
import reader
import random
from ops import *
import scipy.misc
import numpy as np
import poissonblending
import tensorflow as tf
from six.moves import xrange
import vggface
from skimage.measure import compare_psnr
from skimage.measure import compare_mse
from tensorflow.python.platform import gfile
from tensorflow.python.tools import inspect_checkpoint

F = tf.app.flags.FLAGS

class DCGAN(object):
	def __init__(self, sess):
		self.sess = sess
		self.ngf = 128
		self.ndf = 64
		self.nt = 128
		self.k_dim = 16
		self.image_shape = [F.output_size, F.output_size, 3]
		self.build_model()
		if F.output_size == 64:
			self.is_crop = True
		else:
			self.is_crop = False

	def build_model(self):
		# main method for training the conditional GAN
		if F.use_tfrecords == True:
			# load images from tfrecords + queue thread runner for better GPU utilization
			tfrecords_filename = ['train_records/' + x for x in os.listdir('train_records/')]
			filename_queue = tf.train.string_input_producer(
								tfrecords_filename, num_epochs=100)


			self.images, self.keypoints = reader.read_and_decode(filename_queue, F.batch_size)

			if F.output_size == 64:
				self.images = tf.image.resize_images(self.images, (64, 64))
				self.keypoints = tf.image.resize_images(self.keypoints, (64, 64))

			self.images = (self.images / 127.5) - 1
			self.keypoints = (self.keypoints / 127.5) - 1

		else:    
			self.images = tf.placeholder(tf.float32,
									   [F.batch_size, F.output_size, F.output_size,
										F.c_dim],
									   name='real_images')
			self.keypoints = tf.placeholder(tf.float32, [F.batch_size, F.output_size, F.output_size, F.c_dim], name='keypts')
		
		self.mask = tf.placeholder(tf.float32, [F.batch_size, F.output_size, F.output_size, 3], name='mask')
		self.is_training = tf.placeholder(tf.bool, name='is_training')        
		self.get_z_init = tf.placeholder(tf.bool, name='get_z_init')

		self.images_ = tf.multiply(self.mask, self.images)
		
		if F.append_masked_keypoints == True:
			self.keypoints_ = tf.multiply(self.mask, self.keypoints)
			self.z_gen = tf.cond(self.get_z_init, lambda: self.generate_z(tf.concat([self.images_, self.keypoints_], 3)), lambda: tf.placeholder(tf.float32, [F.batch_size, 100], name='z_gen'))
		else:
			self.z_gen = tf.cond(self.get_z_init, lambda: self.generate_z(self.images_), lambda: tf.placeholder(tf.float32, [F.batch_size, 100], name='z_gen'))

		self.G = self.generator(self.z_gen, self.keypoints)
		self.D, self.D_logits = self.discriminator(self.images, self.keypoints, reuse=False)
		self.D_, self.D_logits_, = self.discriminator(self.G, self.keypoints, reuse=True)

		self.d_loss_real = tf.reduce_mean(
			tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits, labels=tf.ones_like(self.D)))
		self.d_loss_fake = tf.reduce_mean(
			tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.zeros_like(self.D_)))
		self.d_loss = self.d_loss_real + self.d_loss_fake
		self.g_loss_actual = tf.reduce_mean(
			tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.ones_like(self.D_)))

		self.loss = 0.0
		if F.error_conceal == True:
			self.contextual_loss = tf.reduce_sum(tf.contrib.layers.flatten(
												 tf.abs(tf.multiply(self.mask, self.G) -
												 tf.multiply(self.mask, self.images))), 1)
			self.perceptual_loss = self.g_loss_actual
			self.complete_loss = self.contextual_loss + F.lam * self.perceptual_loss
			self.grad_complete_loss = tf.gradients(self.complete_loss, self.z_gen)

		else:
			if F.vggface_loss == True:
				vgg_net_inp = tf.concat([self.G, self.images], 0)
				vgg_net_inp = (vgg_net_inp + 1) * 127.5
				print (vgg_net_inp.get_shape())

				#Reduce mean values from pixel values
				rgb_mean = tf.constant([129.18628, 104.76238,  93.59396], dtype=tf.float32)
				rgb_mean = tf.reshape(rgb_mean, [1, 1, 1, 3])

				vgg_net_inp = vgg_net_inp - rgb_mean

				vgg_net = vggface.vgg_face('vgg-face.mat', vgg_net_inp)

				self.loss = 0.01 * (tf.reduce_mean(tf.square(vgg_net['relu3_3'][:F.batch_size] - vgg_net['relu3_3'][F.batch_size:]))+ \
							tf.reduce_mean(tf.square(vgg_net['relu4_3'][:F.batch_size] - vgg_net['relu4_3'][F.batch_size:])))

			else:
				self.loss = 0.01 * tf.reduce_sum(tf.square(self.G - self.images))
			tf.summary.scalar('loss', self.loss)

		if F.disc_loss == True:
			self.disc_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.ones_like(self.D_)))
			tf.summary.scalar('disc_loss', self.disc_loss)
			self.loss += self.disc_loss

		# create summaries  for Tensorboard visualization
		tf.summary.scalar('disc_loss', self.d_loss)
		tf.summary.scalar('disc_loss_real', self.d_loss_real)
		tf.summary.scalar('disc_loss_fake', self.d_loss_fake)
		tf.summary.scalar('gen_loss', self.g_loss_actual)

		self.g_loss = tf.constant(0) 

		t_vars = tf.trainable_variables()
		self.z_vars = [var for var in t_vars if 'Z/z_' in var.name]
		self.d_vars = [var for var in t_vars if 'D/d_' in var.name]
		self.g_vars = [var for var in t_vars if 'G/g_' in var.name]

		# print ([x.name for x in self.g_vars + self.d_vars])
		self.saver_gen = tf.train.Saver(self.g_vars + self.d_vars)
		self.saver = tf.train.Saver()

	def train(self):
		# main method for training conditonal GAN
		global_step = tf.placeholder(tf.int32, [], name="global_step_iterations")

		learning_rate_D = tf.train.exponential_decay(F.learning_rate_D, global_step,
													 decay_steps=F.decay_step,
													 decay_rate=F.decay_rate, staircase=True)
		
		self.summary_op = tf.summary.merge_all()

		z_optim = tf.train.AdamOptimizer(learning_rate_D, beta1=F.beta1D)\
		  .minimize(self.loss, var_list=self.z_vars)

		init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
		self.sess.run(init_op)

		start_time = time.time()

		#self.load_G("checkpoint/celebA/model_weights_" + str(F.output_size))
		self.load_G(F.gan_chkpt)

		"""if F.load_chkpt:
			pass
			try:
				self.load(F.checkpointt_dir)
				print(" [*] Checkpoint Load Success !!!")
			except:
				print(" [!] Checkpoint Load failed !!!!")
		else:
			print(" [*] Not Loaded")
		"""
		self.ra, self.rb = -1, 1
		counter = 1
		step = 1
		idx = 1

		writer = tf.summary.FileWriter(F.log_dir, graph=tf.get_default_graph())

		# Start input enqueue threads.
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)

		try:
			while not coord.should_stop():
				start_time = time.time()
				step += 1

				masks = self.next_mask()
				train_summary, _, zloss = self.sess.run(
						[self.summary_op, z_optim,  self.loss],
						feed_dict={global_step: counter, self.mask: masks, self.is_training: True, self.get_z_init: True})
				writer.add_summary(train_summary, counter)
		
				lrateD = learning_rate_D.eval({global_step: counter})
				print(("Iteration: [%6d],  loss:%.4f, learning_rate: %.8f")
					  % (idx, zloss, lrateD))
				 
				# periodically save checkpoints for future loading
				if np.mod(counter, F.saveInterval) == 1:
					self.save(F.checkpoint_dir, counter)
					print("Checkpoint saved successfully !!!")
					save_imgs, save_opts = self.sess.run([self.images_, self.G], feed_dict={global_step: counter, self.mask: masks, self.is_training: False, self.get_z_init: True})

					save_images(save_imgs, [8, 8], "z_gens/samples_imgs" + str(counter) + ".png")
					save_images(save_opts, [8, 8], "z_gens/samples_opts" + str(counter) + ".png")

				counter += 1
				idx += 1
				
		except tf.errors.OutOfRangeError:
			print('Done training for %d epochs, %d steps.' % (F.num_epochs, step))
		finally:
			# When done, ask the threads to stop.
			coord.request_stop()

		coord.request_stop()
		coord.join(threads)

	def next_mask(self):
		masks = []
		for i in range(F.batch_size):
			masks.append(self.create_mask(temporal=True))
		return masks

	def create_mask(self, centerScale=None, temporal=False, check_size=16):
		# specifically creates random sized/designed mask for consistency experiemnts
		mask_dict = {'freehand_poly': 0, 'center': 1, 'checkboard': 2, 'random': 3, 'left': 4, 'right' : 5, 'top' : 6, 'bottom' : 7}
		if F.error_conceal == False:
			maskType = np.random.randint(0, 8)
		else:
			maskType = mask_dict[F.maskType]

		if maskType == 0:
			image = np.ones(self.image_shape)
			mask = np.ones(self.image_shape)
			freehand_list = []
			freehand_list.append(np.array([ [10,10], [15,10], [30,7], [54, 12], [50, 35], [48, 50], [25, 30]]))
			freehand_list.append(np.array([ [10,10], [10, 15], [7, 30], [12, 54], [35, 50], [50, 48], [30, 25]]))
			freehand_list.append(np.array([ [20,1], [20,20], [10,52], [25, 48], [48,40], [28,20], [20,1] ]))
			freehand_list.append(np.array([ [1,20], [20,20], [52,10], [48, 25], [40,48], [20, 28], [1, 20] ]))
			index = np.random.randint(0,4)

			black = (0, 0, 0)
			if F.output_size ==128:
				 cv2.fillPoly(image, pts = [2 * freehand_list[index]], color = (0, 0, 0))
			else:
				 cv2.fillPoly(image, pts = [freehand_list[index]], color = (0, 0, 0))
			mask = image 


		elif maskType == 1:
			centerScale = 0.25
			# if centerScale == None:
				# centerScale = np.random.uniform(0.2, 0.4, [1])[0]
			assert(centerScale <= 0.5)
			mask = np.ones(self.image_shape)
			sz = F.output_size
			if temporal == True:
			  centerScale = random.uniform(centerScale - 0.05, centerScale + 0.05)
			  
			l = int(F.output_size * centerScale)
			u = int(F.output_size * (1.0-centerScale))
			mask[l:u, l:u, :] = 0.0

		elif maskType == 2:
			if temporal == True:
				check_size_list = [8, 16, 32]
				index = np.random.randint(0, 3)
				check_size = check_size_list[index]

			num_tiles = int(self.image_shape[0] / (2 * check_size))
			w1 = np.ones((check_size, check_size, 3))
			b1 = np.zeros((check_size, check_size, 3))
			stack1 = np.hstack((w1, b1))
			stack2 = np.hstack((b1, w1))
			atom = np.vstack((stack1, stack2))
			mask = np.tile(atom, (num_tiles, num_tiles, 1))

		elif maskType == 3:
			fraction_masked = np.random.uniform(0.2, 0.8) #F.fraction_masked
			mask = np.ones(self.image_shape)
			mask[np.random.random(self.image_shape[:2]) < fraction_masked] = 0.0

		elif maskType == 4:
			mask = np.ones(self.image_shape)
			c = F.output_size // 2
			mask[:,:c,:] = 0.0

		elif maskType == 5:
			mask = np.ones(self.image_shape)
			c = F.output_size // 2
			mask[:,:-c,:] = 0.0

		elif maskType == 6:
			mask = np.ones(self.image_shape)
			c = F.output_size // 2
			mask[:c,:,:] = 0.0

		else:
			mask = np.ones(self.image_shape)
			c = F.output_size // 2
			mask[:-c,:,:] = 0.0

		return mask

	def complete(self):
		# # this is main method which does inpainting (correctness experiment)
		def make_dir(name):
			# Works on python 2.7, where exist_ok arg to makedirs isn't available.
			p = os.path.join(F.outDir, name)
			if not os.path.exists(p):
				os.makedirs(p)

		make_dir('hats_imgs')
		make_dir('completed')
		make_dir('logs')

		try:
			tf.global_variables_initializer().run()
		except:
			tf.initialize_all_variables().run()

		print (F.checkpoint_dir)
		isLoaded = self.load(F.checkpoint_dir)

		#assert(isLoaded)
		img_data_path = 'data/samples_complete/'

		files = os.listdir(img_data_path)#os.listdir('test_images/') #path of held out images for inpainitng experiment
		print("Total files to inpaint :", len(files))
		imgs = files
		# imgs = [x for x in files if 'faces' in x]
		keys = [x.replace('img', 'ky') for x in imgs]
		nImgs = len(imgs)

		batch_idxs = int(np.ceil(nImgs / F.batch_size))
		print("Total batches:::", batch_idxs)
		mask = self.create_mask()
		
		#mask=[]
		#for i in range(int(F.batch_size )):
		#    mask.append(self.create_mask(temporal=True))
			#masks.append(m)

		#with open("masks.txt", "wb") as fp:   #Pickling
		#    pickle.dump(mask, fp)

		psnr_list, psnr_list2 = [], []
		im_cnt = 1
		
		for idx in xrange(0, batch_idxs+1):
			print("Processing batch number:  ", idx)
			#if idx != 0:
			#  continue   # only doing for 7 batches for face net loss reporting
			l = idx * F.batch_size
			u = min((idx + 1) * F.batch_size, nImgs)
			batchSz = u - l
			batch_files = imgs[l:u]
			batch_images = np.array([get_image(img_data_path + batch_file, F.output_size, is_crop=self.is_crop)
								   for batch_file in batch_files]).astype(np.float32)

			batch_files = keys[l:u]
			batch_keypoints = np.array([get_image(img_data_path + batch_file, F.output_size, is_crop=self.is_crop)
									  for batch_file in batch_files]).astype(np.float32)

			m = 0
			v = 0

			nRows = np.ceil(batchSz / 8)
			nCols = min(8, batchSz)
			print (batch_images.shape, nRows, nCols)
			save_images(batch_images[:batchSz,:,:,:], [nRows,nCols],
						os.path.join(F.outDir, 'before_' + str(idx) + '.png'))
			masked_images = np.multiply(batch_images, mask)# - np.multiply(np.ones(batch_images.shape), 1.0 - mask)
			save_images(np.array(masked_images - np.multiply(np.ones(batch_images.shape), 1.0 - mask)), [nRows,nCols],
						os.path.join(F.outDir, 'mask_' + str(idx) + '.png'))

			if F.z_init:
			  zhats, G_imgs = self.sess.run([self.z_gen, self.G], feed_dict = {self.images: batch_images, \
									   self.mask: [mask] * F.batch_size, self.is_training: False, self.get_z_init: True})
			else:
			  zhats = np.random.uniform(-1, 1, size=(F.batch_size, F.z_dim))
			#np.save('zhats.npy', zhats)

			for i in xrange(F.nIter):
				fd = {
					self.mask: [mask] * F.batch_size,
					self.images: batch_images,
					self.keypoints: batch_keypoints,
					self.is_training: False,
					self.z_gen: zhats,
					self.get_z_init: False
				}
				run = [self.complete_loss, self.contextual_loss, self.perceptual_loss, self.grad_complete_loss,  self.G]
				loss, c, p, g, G_imgs = self.sess.run(run, feed_dict=fd)

				for img in range(batchSz):
					with open(os.path.join(F.outDir, 'logs/hats_{:02d}.log'.format(img)), 'ab') as f:
						f.write('{} {} '.format(i, loss[img]).encode())
						np.savetxt(f, zhats[img:img+1])

				if i % F.outInterval == 0:
					print("Iteration: {:04d} |  Contextual Loss = {:.6f}".format(i, np.mean(c)))
					print("                  |  Perceptual Loss = {:.6f}".format( np.mean(p)))

					inv_masked_hat_images = masked_images + np.multiply(G_imgs, 1.0-mask)
					completed = inv_masked_hat_images
					imgName = os.path.join(F.outDir,
										   'completed/_{:02d}_{:04d}.png'.format(idx, i))
					# scipy.misc.imsave(imgName, (G_imgs[0] + 1) * 127.5)
					save_images(completed[:batchSz,:,:,:], [nRows,nCols], imgName)

				if F.approach == 'adam':
					# Optimize single completion with Adam
					m_prev = np.copy(m)
					v_prev = np.copy(v)
					m = F.beta1 * m_prev + (1 - F.beta1) * g[0]
					v = F.beta2 * v_prev + (1 - F.beta2) * np.multiply(g[0], g[0])
					m_hat = m / (1 - F.beta1 ** (i + 1))
					v_hat = v / (1 - F.beta2 ** (i + 1))
					# print 'here'
					zhats += - np.true_divide(F.lr * m_hat, (np.sqrt(v_hat) + F.eps))
					zhats = np.clip(zhats, -1, 1)

				elif F.approach == 'hmc':
					# Sample example completions with HMC (not in paper)
					zhats_old = np.copy(zhats)
					loss_old = np.copy(loss)
					v = np.random.randn(self.batch_size, F.z_dim)
					v_old = np.copy(v)

					for steps in range(F.hmcL):
						v -= F.hmcEps/2 * F.hmcBeta * g[0]
						zhats += F.hmcEps * v
						np.copyto(zhats, np.clip(zhats, -1, 1))
						loss, g, _, _ = self.sess.run(run, feed_dict=fd)
						v -= F.hmcEps/2 * F.hmcBeta * g[0]

					for img in range(batchSz):
						logprob_old = F.hmcBeta * loss_old[img] + np.sum(v_old[img]**2)/2
						logprob = F.hmcBeta * loss[img] + np.sum(v[img]**2)/2
						accept = np.exp(logprob_old - logprob)
						if accept < 1 and np.random.uniform() > accept:
							np.copyto(zhats[img], zhats_old[img])

					F.hmcBeta *= F.hmcAnneal

				else:
					assert(False)   

			inv_masked_hat_images = masked_images + np.multiply(G_imgs, 1.0 - mask)
			blended_images = self.poisson_blend(batch_images, G_imgs, mask)     
			imgName = os.path.join(F.outDir, 'completed/{:02d}_output.png'.format(idx))
			save_images(inv_masked_hat_images[:batchSz,:,:,:], [nRows,nCols], imgName)

			for i in range(len(masked_images)):
				psnr_list2.append(self.get_psnr(batch_images[i], inv_masked_hat_images[i]))
				if F.save_output_images:
					scipy.misc.imsave('facenet_i/' + str(im_cnt) + '.png', batch_images[i])
					scipy.misc.imsave('facenet_o/' + str(im_cnt) + '.png', inv_masked_hat_images[i])
					scipy.misc.imsave('facenet_ob/' + str(im_cnt) + '.png', blended_images[i])
					im_cnt += 1
   
			imgName = os.path.join(F.outDir, 'completed/{:02d}_blended.png'.format(idx))
			save_images(blended_images[:batchSz,:,:,:], [nRows,nCols], imgName)
			
			for i in range(len(masked_images)):
				psnr_list.append(self.get_psnr(batch_images[i], blended_images[i]))

			print("After current batch | PSNR before blending::: ",  np.mean(psnr_list2))
			print("After current batch | PSNR after blending::: ",  np.mean(psnr_list))

		# print ('Final | PSNR Before Blending:: ', np.mean(psnr_list2))
		# np.save(F.outDir + '/complete_psnr_after_blend.npy', np.array(psnr_list)) # For statistical testing

		# print ('Final | PSNR After Blending:: ', np.mean(psnr_list))
		# np.save(F.outDir + '/complete_psnr_before_blend.npy', np.array(psnr_list2)) # For statistical testing

	def temporal(self):
		# main method for performing experiments related to consistency
		# idea: in a batch of 64 images, there will be 8 subjects with 8 different kinds of damages
		# 8 rows of different subjects and 8 columns for each subject

		def make_dir(name):
			# Works on python 2.7, where exist_ok arg to makedirs isn't available.
			p = os.path.join(F.outDir, name)
			if not os.path.exists(p):
				os.makedirs(p)
		make_dir('hats_imgs')
		make_dir('completed')
		make_dir('logs')

		try:
			tf.global_variables_initializer().run()
		except:
			tf.initialize_all_variables().run()

		isLoaded = self.load(F.checkpoint_dir)
		assert(isLoaded)

		img_data_path = 'data/samples_complete/'

		files = os.listdir(img_data_path)
		imgs = [x for x in files if 'im' in x]
		keys = [x.replace('img', 'ky') for x in imgs]
		nImgs = len(imgs)
		batch_idxs = int(np.ceil(nImgs / int(F.batch_size / 8)))

		masks = []
		for i in range(int(F.batch_size / 8)):
			masks.append(self.create_mask(temporal=True))

		mask = np.zeros([F.batch_size, F.output_size, F.output_size, 3])
		for i in range(F.batch_size):
			mask[i] = masks[i % 8]

		psnr_list, psnr_list2 = [], []
		for idx in xrange(0, batch_idxs):  # because in a batch we are taking 8 images instead of 64
			print("Processing batch {:03d} out of {:03d}".format(idx,  batch_idxs))
			batch_size = int(F.batch_size / 8)
			batchSz = F.batch_size
			l = idx * batch_size
			u = min((idx + 1) * batch_size, nImgs)
			batch_files = imgs[l:u]
			batch_images = np.array([get_image(img_data_path + batch_files[int(i / 8)], F.output_size, is_crop=self.is_crop)
					 for i in range(len(batch_files) * 8)]).astype(np.float32)

			batch_files = keys[l:u]
			batch_keypoints = np.array([get_image(img_data_path + batch_files[int(i / 8)], F.output_size, is_crop=self.is_crop)
					 for i in range(len(batch_files) * 8)]).astype(np.float32)

			nRows = np.ceil(batchSz / 8)
			nCols = min(8, batchSz)

			if batchSz < F.batch_size:
				print(batchSz)
				padSz = ((0, int(F.batch_size - batchSz)), (0,0), (0,0), (0,0))
				batch_images = np.pad(batch_images, padSz, 'constant')
				batch_images = batch_images.astype(np.float32)

			if F.z_init:
				zhats, G_imgs = self.sess.run([self.z_gen, self.G], feed_dict = {self.images: batch_images, \
									   self.mask: [mask] * F.batch_size, self.is_training: False, self.get_z_init: True})
			else:
				zhats = np.random.uniform(-1, 1, size=(F.batch_size, F.z_dim))

			m = 0
			v = 0

			nRows = np.ceil(batchSz / 8)
			nCols = min(8, batchSz)
			save_images(batch_images[:batchSz,:,:,:], [nRows,nCols],
						os.path.join(F.outDir, 'before_' + str(idx) + '.png'))

			masked_images = np.multiply(batch_images, mask) 
			save_images(np.array(mask - np.multiply(np.ones(batch_images.shape), 1.0 - mask)), [nRows,nCols],
						os.path.join(F.outDir, 'mask_' + str(idx) + '.png'))

			for i in xrange(F.nIter):
				fd = {
					self.z_gen: zhats,
					self.mask: mask,
					self.keypoints: batch_keypoints,
					self.images: batch_images,
					self.is_training: False,
					self.get_z_init: False
				}
				run = [self.complete_loss, self.contextual_loss, self.perceptual_loss, self.grad_complete_loss,  self.G]
				loss, c, p, g, G_imgs = self.sess.run(run, feed_dict=fd)

				for img in range(batchSz):
					with open(os.path.join(F.outDir, 'logs/hats_{:02d}.log'.format(img)), 'ab') as f:
						f.write('{} {} '.format(i, loss[img]).encode())
						np.savetxt(f, zhats[img:img+1])

				if i % F.outInterval == 0:
					print("Iterations: {:04d} |  Loss = {:.4f}".format(i, np.mean(loss[0:batchSz])))

					inv_masked_hat_images = masked_images + np.multiply(G_imgs, 1.0-mask)
					completed = inv_masked_hat_images
					imgName = os.path.join(F.outDir, 'completed/{:02d}_{:04d}.png'.format(idx, i))
					save_images(completed[:batchSz,:,:,:], [nRows,nCols], imgName)

				if F.approach == 'adam':
					# Optimize single completion with Adam
					m_prev = np.copy(m)
					v_prev = np.copy(v)
					m = F.beta1 * m_prev + (1 - F.beta1) * g[0]
					v = F.beta2 * v_prev + (1 - F.beta2) * np.multiply(g[0], g[0])
					m_hat = m / (1 - F.beta1 ** (i + 1))
					v_hat = v / (1 - F.beta2 ** (i + 1))
					zhats += - np.true_divide(F.lr * m_hat, (np.sqrt(v_hat) + F.eps))
					zhats = np.clip(zhats, -1, 1)

				elif F.approach == 'hmc':
					# Sample example completions with HMC (not in paper)
					zhats_old = np.copy(zhats)
					loss_old = np.copy(loss)
					v = np.random.randn(self.batch_size, F.z_dim)
					v_old = np.copy(v)

					for steps in range(F.hmcL):
						v -= F.hmcEps/2 * F.hmcBeta * g[0]
						zhats += F.hmcEps * v
						np.copyto(zhats, np.clip(zhats, -1, 1))
						loss, g, _, _ = self.sess.run(run, feed_dict=fd)
						v -= F.hmcEps/2 * F.hmcBeta * g[0]

					for img in range(batchSz):
						logprob_old = F.hmcBeta * loss_old[img] + np.sum(v_old[img]**2)/2
						logprob = F.hmcBeta * loss[img] + np.sum(v[img]**2)/2
						accept = np.exp(logprob_old - logprob)
						if accept < 1 and np.random.uniform() > accept:
							np.copyto(zhats[img], zhats_old[img])

					F.hmcBeta *= F.hmcAnneal

				else:
					assert(False)

			inv_masked_hat_images = masked_images + np.multiply(G_imgs, 1.0 - mask)
			blended_images = self.poisson_blend2(batch_images, G_imgs, mask)
			imgName = os.path.join(F.outDir, 'completed/{:02d}_blended.png'.format(idx))
			save_images(blended_images[:batchSz,:,:,:], [nRows,nCols], imgName)

			# calculate consistency for each possible pairwise inpainted images before/after blending
			for i in range(8):
				for j in range(8):
					for k in range(j):
						psnr_list2.append(self.get_mse(batch_images[i * 8 + j], inv_masked_hat_images[i * 8 + k]))
						psnr_list.append(self.get_mse(blended_images[i * 8 + j], blended_images[i * 8 + k]))
			

			print("Uptil now | MSE Before Blending::",  np.mean(psnr_list2))
			print("Uptil now | MSE After Blending::",  np.mean(psnr_list))

	def poisson_blend(self, imgs1, imgs2, mask):
		# call this while performing correctness experiment
		out = np.zeros(imgs1.shape)

		for i in range(0, len(imgs1)):
			img1 = (imgs1[i] + 1.) / 2.0
			img2 = (imgs2[i] + 1.) / 2.0
			out[i] = np.clip((poissonblending.blend(img1, img2, 1 - mask) - 0.5) * 2, -1.0, 1.0)

		return out.astype(np.float32)

	def poisson_blend2(self, imgs1, imgs2, mask):
		# call this while performing consistency experiment
		out = np.zeros(imgs1.shape)

		for i in range(0, len(imgs1)):
			img1 = (imgs1[i] + 1.) / 2.0
			img2 = (imgs2[i] + 1.) / 2.0
			out[i] = np.clip((poissonblending.blend(img1, img2, 1 - mask[i]) - 0.5) * 2, -1.0, 1.0)

		return out.astype(np.float32)

	def get_psnr(self, img_true, img_gen):
		return compare_psnr(img_true.astype(np.float32), img_gen.astype(np.float32))

	def get_mse(self, img_true, img_gen):
		return compare_mse(img_true.astype(np.float32), img_gen.astype(np.float32))

	def discriminator(self, image, keypoints, reuse=False):
		with tf.variable_scope('D'):
			if reuse:
				tf.get_variable_scope().reuse_variables()

			dim = 64
			image = tf.concat([image, keypoints], 3)
			if F.output_size == 128:
				  h0 = lrelu(conv2d(image, dim, name='d_h0_conv'))
				  h1 = lrelu(batch_norm(name='d_bn1')(conv2d(h0, dim * 2, name='d_h1_conv'), self.is_training))
				  h2 = lrelu(batch_norm(name='d_bn2')(conv2d(h1, dim * 4, name='d_h2_conv'), self.is_training))
				  h3 = lrelu(batch_norm(name='d_bn3')(conv2d(h2, dim * 8, name='d_h3_conv'), self.is_training))
				  h4 = lrelu(batch_norm(name='d_bn4')(conv2d(h3, dim * 16, name='d_h4_conv'), self.is_training))
				  h4 = tf.reshape(h4, [F.batch_size, -1])
				  h5 = linear(h4, 1, 'd_h5_lin')
				  return tf.nn.sigmoid(h5), h5

			else:
				  h0 = lrelu(conv2d(image, dim, name='d_h0_conv'))
				  h1 = lrelu(batch_norm(name='d_bn1')(conv2d(h0, dim * 2, name='d_h1_conv'), self.is_training))
				  h2 = lrelu(batch_norm(name='d_bn2')(conv2d(h1, dim * 4, name='d_h2_conv'), self.is_training))
				  h3 = lrelu(batch_norm(name='d_bn3')(conv2d(h2, dim * 8, name='d_h3_conv'), self.is_training))
				  h4 = tf.reshape(h3, [F.batch_size, -1])
				  h5 = linear(h4, 1, 'd_h5_lin')
				  return tf.nn.sigmoid(h5), h5

	def generator(self, z, keypoints):
		dim = 64
		k = 5
		with tf.variable_scope("G"):
			  s2, s4, s8, s16 = int(F.output_size / 2), int(F.output_size / 4), int(F.output_size / 8), int(F.output_size / 16)
			  z = tf.reshape(z, [F.batch_size, 1, 1, 100])
			  z = tf.tile(z, [1, F.output_size, F.output_size, 1])
			  z = tf.concat([z, keypoints], 3)

			  h0 = z
			
			  h1 = tf.nn.relu(batch_norm(name='g_bn1')(conv2d(h0, dim * 2, 5, 5, 1, 1, name='g_h1_conv'), self.is_training))
			  h2 = tf.nn.relu(batch_norm(name='g_bn2')(conv2d(h1, dim * 2, k, k, 2, 2, name='g_h2_conv'), self.is_training))
			  h3 = tf.nn.relu(batch_norm(name='g_bn3')(conv2d(h2, dim * 4, k, k, 2, 2, name='g_h3_conv'), self.is_training))
			  h4 = tf.nn.relu(batch_norm(name='g_bn4')(conv2d(h3, dim * 8, k, k, 2, 2, name='g_h4_conv'), self.is_training))
			  h5 = tf.nn.relu(batch_norm(name='g_bn5')(conv2d(h4, dim * 16, k, k, 2, 2, name='g_h5_conv'), self.is_training))

			  h6 = deconv2d(h5, [F.batch_size, s8, s8, dim * 8], k, k, 2, 2, name = 'g_deconv1')
			  h6 = tf.nn.relu(batch_norm(name = 'g_bn6')(h6, self.is_training))
					  
			  h7 = deconv2d(h6, [F.batch_size, s4, s4, dim * 4], k, k, 2, 2, name = 'g_deconv2')
			  h7 = tf.nn.relu(batch_norm(name = 'g_bn7')(h7, self.is_training))

			  h8 = deconv2d(h7, [F.batch_size, s2, s2, dim * 2], k, k, 2, 2, name = 'g_deconv4')
			  h8 = tf.nn.relu(batch_norm(name = 'g_bn8')(h8, self.is_training))

			  h9 = deconv2d(h8, [F.batch_size, F.output_size, F.output_size, 3], k, k, 2, 2, name ='g_hdeconv5')
			  h9 = tf.nn.tanh(h9, name = 'g_tanh')
			  return h9
			  
	def generate_z(self, image):
		with tf.variable_scope('Z'):
			dim = 64
			if F.output_size == 128:
				  h0 = lrelu(conv2d(image, dim, name='z_h0_conv'))
				  h1 = lrelu(batch_norm(name='z_bn1')(conv2d(h0, dim * 2, name='z_h1_conv'), self.is_training))
				  h2 = lrelu(batch_norm(name='z_bn2')(conv2d(h1, dim * 4, name='z_h2_conv'), self.is_training))
				  h3 = lrelu(batch_norm(name='z_bn3')(conv2d(h2, dim * 8, name='z_h3_conv'), self.is_training))
				  h4 = lrelu(batch_norm(name='z_bn4')(conv2d(h3, dim * 16, name='z_h4_conv'), self.is_training))
				  h4 = tf.reshape(h4, [F.batch_size, -1])
				  h5 = linear(h4, 100, 'z_h5_lin')
				  return tf.nn.tanh(h5)

			else:
				  h0 = lrelu(conv2d(image, dim, name='z_h0_conv'))
				  h1 = lrelu(batch_norm(name='z_bn1')(conv2d(h0, dim * 2, name='z_h1_conv'), self.is_training))
				  h2 = lrelu(batch_norm(name='z_bn2')(conv2d(h1, dim * 4, name='z_h2_conv'), self.is_training))
				  h3 = lrelu(batch_norm(name='z_bn3')(conv2d(h2, dim * 8, name='z_h3_conv'), self.is_training))
				  h4 = tf.reshape(h3, [F.batch_size, -1])
				  h5 = linear(h4, 100, 'z_h5_lin')
				  return tf.nn.tanh(h5)

	def save(self, checkpoint_dir, step=0):
		model_name = "model"
		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)

		self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=step)

	def load_G(self, checkpoint_dir):
		print(" [*] Reading checkpoints of G...")
		ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
			self.saver_gen.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
			return True
		else:
			return False

	def load(self, checkpoint_dir):
		print("Normal loading !!!!!!!!!!!!!")
		print(" [*] Reading checkpoints...")
		ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
			self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
			return True
		else:
			return False
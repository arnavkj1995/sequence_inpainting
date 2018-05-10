from __future__ import division
import os
import sys
import time
import reader
import random
from ops import *
import scipy.misc
import numpy as np
import tensorflow as tf
import poissonblending
from six.moves import xrange
from skimage.measure import compare_psnr
from skimage.measure import compare_mse

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
            tfrecords_filename = ['train_records_seq/' + x for x in os.listdir('train_records_seq/')]
            filename_queue = tf.train.string_input_producer(
                                tfrecords_filename, num_epochs=100)


            self.images = reader.read_and_decode(filename_queue, F.batch_size)

            if F.output_size == 64:
                self.images = tf.image.resize_images(self.images, (64, 64))

            self.images = (self.images / 127.5) - 1

        else:    
            self.images = tf.placeholder(tf.float32,
                                       [F.batch_size, F.output_size, F.output_size,
                                        F.c_dim],
                                       name='real_images')
            
        self.is_training = tf.placeholder(tf.bool, name='is_training')        
        self.z_gen = tf.placeholder(tf.float32, [F.batch_size, F.z_dim], name='z')

        self.G = self.generator(self.z_gen)
        self.D, self.D_logits = self.discriminator(self.images, reuse=False)
        self.D_, self.D_logits_, = self.discriminator(self.G, reuse=True)

        self.d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits, labels=tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.zeros_like(self.D_)))
        self.d_loss = self.d_loss_real + self.d_loss_fake
        self.g_loss_actual = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.ones_like(self.D_)))

        if F.error_conceal == True:
            self.mask = tf.placeholder(tf.float32, [F.batch_size] + self.image_shape, name='mask')
            self.contextual_loss = tf.reduce_sum(tf.contrib.layers.flatten(
                                                 tf.abs(tf.multiply(self.mask, self.G) -
                                                 tf.multiply(self.mask, self.images))), 1)
            self.perceptual_loss = self.g_loss_actual
            self.complete_loss = self.contextual_loss + F.lam * self.perceptual_loss
            self.grad_complete_loss = tf.gradients(self.complete_loss, self.z_gen)

        # create summaries  for Tensorboard visualization
        tf.summary.scalar('disc_loss', self.d_loss)
        tf.summary.scalar('disc_loss_real', self.d_loss_real)
        tf.summary.scalar('disc_loss_fake', self.d_loss_fake)
        tf.summary.scalar('gen_loss', self.g_loss_actual)

        self.g_loss = tf.constant(0) 

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'D/d_' in var.name]
        self.g_vars = [var for var in t_vars if 'G/g_' in var.name]

        print ([x.name for x in t_vars])
        self.saver = tf.train.Saver()

    def train(self):
        # main method for training conditonal GAN

        global_step = tf.placeholder(tf.int32, [], name="global_step_iterations")

        learning_rate_D = tf.train.exponential_decay(F.learning_rate_D, global_step,
                                                     decay_steps=F.decay_step,
                                                     decay_rate=F.decay_rate, staircase=True)
        learning_rate_G = tf.train.exponential_decay(F.learning_rate_G, global_step,
                                                     decay_steps=F.decay_step,
                                                     decay_rate=F.decay_rate, staircase=True)
        
        self.summary_op = tf.summary.merge_all()

        d_optim = tf.train.AdamOptimizer(learning_rate_D, beta1=F.beta1D)\
          .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(learning_rate_G, beta1=F.beta1G)\
          .minimize(self.g_loss_actual, var_list=self.g_vars)

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(init_op)

        start_time = time.time()
        self.load(F.checkpoint_dir)
        
        # if F.load_chkpt:
        #     try:
        #         self.load(F.checkpoint_dir)
        #         print(" [*] Checkpoint Load Success !!!")
        #     except:
        #         print(" [!] Checkpoint Load failed !!!!")
        # else:
        #     print(" [*] Not Loaded")

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

                # sample a noise vector 
                sample_z_gen = np.random.uniform(
                        self.ra, self.rb, [F.batch_size, F.z_dim]).astype(np.float32)

                # Update D network
                iters = 1
                if True: 
                    train_summary, _, dloss, errD_fake, errD_real = self.sess.run(
                            [self.summary_op, d_optim,  self.d_loss, self.d_loss_fake, self.d_loss_real],
                            feed_dict={self.z_gen: sample_z_gen, global_step: counter, self.is_training: True})
                    writer.add_summary(train_summary, counter)

                # Update G network
                iters = 1  # can play around 
                if True :
                    for iter_gen in range(iters):
                        sample_z_gen = np.random.uniform(self.ra, self.rb,
                            [F.batch_size, F.z_dim]).astype(np.float32)
                        _,  gloss, dloss = self.sess.run(
                            [g_optim,  self.g_loss_actual, self.d_loss],
                            feed_dict={self.z_gen: sample_z_gen, global_step: counter, self.is_training: True})
                       
                lrateD = learning_rate_D.eval({global_step: counter})
                lrateG = learning_rate_G.eval({global_step: counter})

                print(("Iteration: [%6d] lrateD:%.2e lrateG:%.2e d_loss_f:%.8f d_loss_r:%.8f " +
                      "g_loss_act:%.8f")
                      % (idx, lrateD, lrateG, errD_fake, errD_real, gloss))

                # peridically save generated images with corresponding checkpoints

                if np.mod(counter, F.sampleInterval) == 2:
                    sample_z_gen = np.random.uniform(self.ra, self.rb, [F.batch_size, F.z_dim]).astype(np.float32)
                    samples, d_loss, g_loss_actual = self.sess.run(
                        [self.G, self.d_loss, self.g_loss_actual],
                        feed_dict={self.z_gen: sample_z_gen, global_step: counter, self.is_training: False}
                    )
                    #save_images(samples, [8, 8], "initial_images.png")
                    save_images(samples, [8, 8],
                                 F.sample_dir + "/sample_" + str(counter) + ".png")
                    print("new samples stored!!")
                 
                # # periodically save checkpoints for future loading
                if np.mod(counter, F.saveInterval) == 0:
                     self.save(F.checkpoint_dir_vid, counter)
                     print("Checkpoint saved successfully !!!")

                counter += 1
                idx += 1
                
        except tf.errors.OutOfRangeError:
            print('Done training for %d epochs, %d steps.' % (F.num_epochs, step))
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()

        coord.request_stop()
        coord.join(threads)

    def discriminator(self, image, reuse=False):
        with tf.variable_scope('D'):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            dim = 64
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

    def generator(self, z):
        dim = 64
        k = 5
        with tf.variable_scope("G"):
              s2, s4, s8, s16 = int(F.output_size / 2), int(F.output_size / 4), int(F.output_size / 8), int(F.output_size / 16)

              h0 = linear(z, s16 * s16 * dim * 16, 'g_lin')
              h0 = tf.reshape(h0, [F.batch_size, s16, s16, dim * 16])

              h1 = deconv2d(h0, [F.batch_size, s8, s8, dim * 8], k, k, 2, 2, name = 'g_deconv1')
              h1 = tf.nn.relu(batch_norm(name = 'g_bn1')(h1, self.is_training))
                      
              h2 = deconv2d(h1, [F.batch_size, s4, s4, dim * 4], k, k, 2, 2, name = 'g_deconv2')
              h2 = tf.nn.relu(batch_norm(name = 'g_bn2')(h2, self.is_training))

              h3 = deconv2d(h2, [F.batch_size, s2, s2, dim * 2], k, k, 2, 2, name = 'g_deconv4')
              h3 = tf.nn.relu(batch_norm(name = 'g_bn3')(h3, self.is_training))

              h4 = deconv2d(h3, [F.batch_size, F.output_size, F.output_size, 3], k, k, 2, 2, name ='g_hdeconv5')
              h4 = tf.nn.tanh(h4, name = 'g_tanh')
              return h4

    def save(self, checkpoint_dir, step=0):
        model_name = "model"
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

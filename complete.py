import numpy as np
import scipy.misc
import os
import sys
sys.path.append('../')

from generate_z import DCGAN
from utils import pp

import tensorflow as tf

flags = tf.app.flags
# flags.DEFINE_integer("momentum_decay_steps", 100,
#                      "change after 100 iterations of inner loop of G")
# flags.DEFINE_float("momentum_decay_rate", 1.17, "factor of change in momentum")
flags.DEFINE_integer("epoch_pretrain", 500, "Epoch to train [25]")
flags.DEFINE_integer("epoch_policy", 2000000, "Epochs for policy gradient")
flags.DEFINE_float("learning_rate_D", 0.0001,
				   "Learning rate of for adam [0.0002]")
flags.DEFINE_float("learning_rate_G", 0.0005,
				   "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1D", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_float("beta1G", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("decay_step", 5000, "Decay step of learning rate in epochs")
flags.DEFINE_float("decay_rate", 0.8, "Decay rate of learning rate")
flags.DEFINE_float("eps", 1e-6, "Epsilon")
flags.DEFINE_float("var", 1e-5, "Variance")
flags.DEFINE_float("gpu_frac", 0.5, "Gpu fraction")
flags.DEFINE_integer("no_of_samples", 50,
					 "no of samples for each noise vector Z during policy gradient")
flags.DEFINE_boolean("teacher_forcing", False,
					 "True if teacher forcing is enabled")
flags.DEFINE_boolean("label_to_disc", True,
					 "True if labels are passed to the discriminator")
flags.DEFINE_boolean("conditional", True,
					 "True if want to train conditional GAN")
flags.DEFINE_integer("pre_train_iters", 2000,
					  "Number of iterations to pre-train D")
flags.DEFINE_integer("num_keypoints", 68,
					  "Number of keypoints extracted in the face")
# flags.DEFINE_integer('outDir', 'completions', "Directory to save completed images.")

dataset = "celebA"
comment = "model_weights_64_vgg"


"""   --meaning for the acronyms for folder names ----
 chkpt
BN:Batch Norm to G
 margin: margin restriction applied to alternate training
 slow decay: decay learning rate after 50 epochs  with decay rate 0.8
"""
# meaningful comment to make dirs for chkpt and storing samples for
# different exp

flags.DEFINE_float(
	"margin", 0.3, "Threshold to judge stopping of D and G nets training")
flags.DEFINE_boolean("margin_restriction", True,
					 "whether to use margin restriction to stop D or G nets")
flags.DEFINE_boolean("policy_train", True,
					 "Whether to use PolicyGan training procedure")

flags.DEFINE_string("dataset", dataset,
					"The name of dataset [celebA, mnist, lsun]")
if dataset == 'celebA':
  flags.DEFINE_string("data_dir", "data/",
					"Directory name containing the dataset [data]")
else:
  flags.DEFINE_string("data_dir", "data/" + dataset,
					"Directory name containing the dataset [data]")
flags.DEFINE_string("checkpoint_dir", "checkpoint/" + dataset + "/" + comment,
					"Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples/" + dataset + "/" + comment,
					"Directory name to save the image samples [samples]")
flags.DEFINE_string("log_dir", "logs/" + dataset + "/" + comment,
					"Directory name to save the logs [logs]")
flags.DEFINE_boolean("load_chkpt", False, "True for loading saved checkpoint")
flags.DEFINE_boolean("inc_score", False, "True for computing inception score")
flags.DEFINE_boolean("gauss_noise", False, "True for adding noise to disc input")
flags.DEFINE_boolean("flip_label", False, "True for flipping the labels")
flags.DEFINE_boolean("error_conceal", True, "True for running error concealment part")
flags.DEFINE_boolean("use_tfrecords", False, "True for running error concealment part")
flags.DEFINE_boolean("disc_loss", True, "True for using D loss")

flags.DEFINE_string('--approach', 'adam', 'Approach for back tracking in z-space')
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("z_dim", 100, "Dimension of latent vector.")

flags.DEFINE_float('lr',0.001, 'lr for z')
flags.DEFINE_float('beta1',0.9, 'beta1')
flags.DEFINE_float('beta2',0.999, 'beta2')
# flags.DEFINE_float('eps',1e-8, 'eps')
flags.DEFINE_float('hmcBeta', 0.2, "hmcBeta")
flags.DEFINE_float('hmcEps', 0.001, "hmcEps")
flags.DEFINE_integer('hmcL', 100, "hmcL")
flags.DEFINE_integer('hmcAnneal', 1, "hmcAnneal")
flags.DEFINE_integer('nIter', 100, "nIter")
flags.DEFINE_float('lam', 0.01, "lam")
flags.DEFINE_string('outDir', 'completions_results_video_conf', "Directory to save completed images.")
flags.DEFINE_integer('outInterval', 10, 'outInterval')
flags.DEFINE_string('maskType', 'checkboard', 'maskType')
flags.DEFINE_float('centerScale', 0.15, 'centerScale')
flags.DEFINE_string('imgs', ' ', 'Images list')

if dataset == "mnist":
	flags.DEFINE_integer("c_dim", 1, "Number of channels in input image")
	flags.DEFINE_boolean("is_grayscale", True, "True for grayscale image")
	flags.DEFINE_integer("output_size", 28, "True for grayscale image")
	flags.DEFINE_integer("num_classes", 10, "Number of class labels")
elif dataset == "lsun" or dataset == "lfw" or dataset == "celebA":
	flags.DEFINE_integer("c_dim", 3, "Number of channels in input image")
	flags.DEFINE_boolean("is_grayscale", False, "True for grayscale image")
	flags.DEFINE_integer("output_size", 64, "True for grayscale image")
else:
	flags.DEFINE_integer("c_dim", 3, "Number of channels in input image")
	flags.DEFINE_boolean("is_grayscale", False, "True for grayscale image")
	flags.DEFINE_integer("output_size", 32, "True for grayscale image")
	if dataset == "cifar":
		flags.DEFINE_integer("num_classes", 10, "Number of class labels")


FLAGS = flags.FLAGS

def main(_):
	pp.pprint(flags.FLAGS.__flags)

	if not os.path.exists(FLAGS.checkpoint_dir):
		os.makedirs(FLAGS.checkpoint_dir)
	if not os.path.exists(FLAGS.sample_dir):
		os.makedirs(FLAGS.sample_dir)

	gpu_options = tf.GPUOptions(
		per_process_gpu_memory_fraction=FLAGS.gpu_frac)

	with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
		dcgan = DCGAN(sess)
		dcgan.complete()

if __name__ == '__main__':
	tf.app.run()

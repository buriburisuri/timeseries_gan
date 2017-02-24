import sugartensor as tf
import numpy as np
from data import TimeSeriesData
from model import *


__author__ = 'namju.kim@kakaobrain.com'


# set log level to debug
tf.sg_verbosity(10)

#
# hyper parameters
#

batch_size = 32   # batch size


#
# inputs
#

# input tensor ( with QueueRunner )
data = TimeSeriesData(batch_size=batch_size)
x = data.X

# labels for discriminator
y_real = tf.ones(batch_size)
y_fake = tf.zeros(batch_size)

# categorical latent variable
z_cat = tf.multinomial(tf.ones((batch_size, cat_dim), dtype=tf.sg_floatx) / cat_dim, 1).sg_squeeze().sg_int()
# continuous latent variable
z_con = tf.random_uniform((batch_size, con_dim))
# random latent variable dimension
z_rand = tf.random_uniform((batch_size, rand_dim))
# latent variable
z = tf.concat([z_cat.sg_one_hot(depth=cat_dim), z_con, z_rand], 1)


#
# Computational graph
#

# generator
gen = generator(z)

# discriminator
disc_real, _, _ = discriminator(x)
disc_fake, cat_fake, con_fake = discriminator(gen)


#
# loss
#

# discriminator loss
loss_d_r = disc_real.sg_bce(target=y_real, name='disc_real')
loss_d_f = disc_fake.sg_bce(target=y_fake, name='disc_fake')
loss_d = (loss_d_r + loss_d_f) / 2


# generator loss
loss_g = disc_fake.sg_bce(target=y_real, name='gen')

# categorical factor loss
loss_c = cat_fake.sg_ce(target=z_cat, name='cat')

# continuous factor loss
loss_con = con_fake.sg_mse(target=z_con, name='con').sg_mean(axis=1)


#
# train ops
#

# discriminator train ops
train_disc = tf.sg_optim(loss_d + loss_c + loss_con, lr=0.0001, category='discriminator')
# generator train ops
train_gen = tf.sg_optim(loss_g + loss_c + loss_con, lr=0.001, category='generator')


#
# training
#

# def alternate training func
@tf.sg_train_func
def alt_train(sess, opt):
    l_disc = sess.run([loss_d] + train_disc)[0]  # training discriminator
    l_gen = sess.run([loss_g] + train_gen)[0]  # training generator
    return np.mean(l_disc) + np.mean(l_gen)

# do training
alt_train(log_interval=10, max_ep=100, ep_size=data.num_batch, early_stop=False)


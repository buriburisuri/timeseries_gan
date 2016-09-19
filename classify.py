# -*- coding: utf-8 -*-
import sugartensor as tf
import numpy as np
import matplotlib.pyplot as plt


__author__ = 'njkim@jamonglab.com'

# set log level to debug
tf.sg_verbosity(10)

#
# hyper parameters
#

batch_size = 32   # batch size
num_category = 10  # category variable number
num_cont = 2   # continuous variable number
num_dim = 30   # total latent dimension ( category + continuous + noise )

#
# Data loading
#

# load data
x = np.genfromtxt('asset/data/sample.csv', delimiter=',', dtype=np.float32)
x = x[1:, 1:]

window = 384  # window size
max = 3000  # max value

# delete zero pad data
n = ((np.where(np.any(x, axis=1))[0][-1] + 1) // window) * window

# normalize data between 0 and 1
x = x[:n] / max

# make to matrix
X = np.asarray([x[i:i+window] for i in range(n-window)])
X = np.expand_dims(X, axis=2)

num_batch = X.shape[0] // batch_size

#
# inputs place holder
#

ph = tf.sg_input(shape=(window, 1, 2))

#
# create generator
#

# dummy place holder
z = tf.sg_input(num_dim)

# generator network
with tf.sg_context(name='generator', size=(4, 1), stride=(2, 1), act='relu', bn=True):
    gen = (z.sg_dense(dim=1024)
           .sg_dense(dim=48*1*128)
           .sg_reshape(shape=(-1, 48, 1, 128))
           .sg_upconv(dim=64)
           .sg_upconv(dim=32)
           .sg_upconv(dim=2, act='sigmoid', bn=False))

#
# create discriminator & recognizer
#

with tf.sg_context(name='discriminator', size=(4, 1), stride=(2, 1), act='leaky_relu'):
    # shared part
    shared = (ph.sg_conv(dim=32)
              .sg_conv(dim=64)
              .sg_conv(dim=128)
              .sg_flatten()
              .sg_dense(dim=1024))
    # shared recognizer part
    recog_shared = shared.sg_dense(dim=128)
    # discriminator end
    disc = shared.sg_dense(dim=1, act='linear').sg_squeeze()
    # categorical recognizer end
    recog_cat = recog_shared.sg_dense(dim=num_category, act='softmax')
    # continuous recognizer end
    recog_cont = recog_shared.sg_dense(dim=num_cont, act='sigmoid')

#
# run discrimination
#
with tf.Session() as sess:
    # init session
    tf.sg_init(sess)
    # restore parameters
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint('asset/train/ckpt'))
    # run recoginizer
    cats, conts = [], []
    for i in range(0, len(X), batch_size):
        a, b = sess.run([recog_cat, recog_cont], {ph: X[i:i+batch_size]})
        cats.append(a)
        conts.append(b)
        tf.sg_debug('%d/%d processed.' % (i/batch_size + 1, num_batch))
    # to numpy array
    cats = np.argmax(np.concatenate(cats, axis=0), axis=1)   # we take max value.
    conts = np.concatenate(conts, axis=0)

#
# Plotting result
#
##
colors = [(0, 0, 0.5), (0, 0, 1), (0, 0.5, 0), (0, 0.5, 0.5), (0, 0.5, 1),
          (0, 1, 0), (0, 1, 0.5), (0, 1, 1), (0.5, 0, 0), (0.5, 0, 0.5)]

_, ax = plt.subplots(2, 1, sharex=True, sharey=True)

# plot original time series
ax[0].plot(x)
ax[0].set_title('original time series')
ax[0].set_xlabel('time')
ax[0].set_ylabel('normalized value')
ax[0].grid()

# coloring category division
i_prev = 0
for i in np.where(np.diff(cats))[0]:
    ax[0].axvspan(i_prev + window/2, i + window/2, facecolor=colors[cats[i]], alpha=0.2)
    i_prev = i + 1

# plot continous factors
ax[1].plot(np.arange(len(conts))+window/2, conts)
ax[1].set_title('decomposed continuous factors')
ax[1].set_xlabel('time')
ax[1].set_ylabel('normalized value')
ax[1].grid()

# save plot
plt.savefig('asset/train/classify.png', dpi=600)

# show plot
plt.show()


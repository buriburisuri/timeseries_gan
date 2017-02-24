import sugartensor as tf
import numpy as np
import matplotlib.pyplot as plt
from model import *


__author__ = 'namju.kim@kakaobrain.com'


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

# discriminator
_, recog_cat, recog_cont = discriminator(ph)

#
# run discrimination
#
with tf.Session() as sess:

    # init session
    tf.sg_init(sess)

    # restore parameters
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint('asset/train/'))

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


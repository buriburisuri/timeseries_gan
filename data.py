# -*- coding: utf-8 -*-
import sugartensor as tf
import numpy as np


__author__ = 'njkim@jamonglab.com'


class TimeSeriesData(object):

    def __init__(self, batch_size=128):

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
        np.random.shuffle(X)
        X = np.expand_dims(X, axis=2)

        # save to member variable
        self.batch_size = batch_size
        self.X = tf.sg_data._data_to_tensor([X], batch_size, name='train')
        self.num_batch = X.shape[0] // batch_size

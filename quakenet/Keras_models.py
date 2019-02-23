__author__ = 'Lei Huang'
"""
ConvNetQuake model rewritten using Keras
02/22/2019

"""
import os
import time

import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline

from tensorflow.keras.layers import Conv1D, MaxPool1D, Flatten, Dense, Dropout, Input


class KerasConvNetQuake:

  def __init__(self, inputs, config, checkpoint_dir, is_training=False,
               reuse=False):
    self.is_training = is_training
    self.config = config
    self.is_training = is_training
    self.model = None


  def set_model(self):
      inputs = Input(shape=(1001, 3))
      c = 32  # number of channels per conv layer
      ksize = 3  # size of the convolution kernel
      depth = 8
      x = inputs
      for i in range(depth):
          x = Conv1D(c, ksize, activation='relu', padding='same')(x)
          x = MaxPool1D(pool_size=2)(x)
      x = Flatten()(x)
      x = Dense(128, activation='relu')(x)
      #x = Dropout(0.5)(x)
      outputs = Dense(self.config.n_clusters, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)



      self.model = tf.keras.Model(inputs, outputs)
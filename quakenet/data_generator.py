__author__ = 'Lei Huang'

import os
import re
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import csv
import json
from obspy.core.utcdatetime import UTCDateTime


class DataGenerator(object):

    """Creates a dataset to stream data for training.

    Attributes:
    samples: Tensor(float). batch of input samples [batch_size, n_channels, n_points]
    labels: Tensor(int32). Corresponding batch 0 or 1 labels, [batch_size,]

    """

    def __init__(self, dataset_path, config, is_training=True):

        min_after_dequeue = 1000
        capacity = 1000 + 3 * config.batch_size
        self.is_training = is_training
        self.config = config
        self._path = dataset_path if isinstance(input, list) else [dataset_path]
        self.win_size = config.win_size
        self.n_traces = config.n_traces


    def getFeatures(self):
        filename_queue = self._filename_queue()
        dataset = tf.data.TFRecordDataset(filename_queue)
        dataset = dataset.map(self._parse_features)
        return dataset

    def _parse_features(self, serialized_example):
        features = tf.parse_single_example(
            serialized_example,
            features={
                'window_size': tf.FixedLenFeature([], tf.int64),
                'n_traces': tf.FixedLenFeature([], tf.int64),
                'data': tf.FixedLenFeature([], tf.string),
                'cluster_id': tf.FixedLenFeature([], tf.int64),
                'start_time': tf.FixedLenFeature([],tf.int64),
                'end_time': tf.FixedLenFeature([], tf.int64)})

        return features

    def read(self):
        filename_queue = self._filename_queue()
        print(filename_queue)
        dataset = tf.data.TFRecordDataset(filename_queue)
        dataset = dataset.apply(tf.data.experimental.map_and_batch(
            self._parse_function, self.config.batch_size,
            num_parallel_batches=4,  # cpu cores
            drop_remainder=True if self.is_training else False))
        if self.is_training:
            dataset = dataset.shuffle(100000)  # depends on sample size
        dataset = dataset.repeat()
        dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)

        return dataset

    def _filename_queue(self):
        fnames = []

        for path in self._path:
            for root, dirs, files in os.walk(path):
                for f in files:
                    if f.endswith(".tfrecords"):
                        fnames.append(os.path.join(root, f))

        return fnames

    def _parse_function(self, serialized_example):
        features = tf.parse_single_example(
            serialized_example,
            features={
                'window_size': tf.FixedLenFeature([], tf.int64),
                'n_traces': tf.FixedLenFeature([], tf.int64),
                'data': tf.FixedLenFeature([], tf.string),
                'cluster_id': tf.FixedLenFeature([], tf.int64),
                'start_time': tf.FixedLenFeature([],tf.int64),
                'end_time': tf.FixedLenFeature([], tf.int64)})

        # Convert and reshape
        data = tf.decode_raw(features['data'], tf.float32)
        data.set_shape([self.n_traces * self.win_size])
        data = tf.reshape(data, [self.n_traces, self.win_size])
        data = tf.transpose(data, [1, 0])

        # Pack
        features['data'] = data
        label = tf.cast(features["cluster_id"]+1, tf.uint8)
        #label = tf.one_hot(tf.cast(features['cluster_id'], tf.uint8), self._config.n_clusters)
        return data, label

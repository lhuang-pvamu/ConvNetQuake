#!/usr/bin/env python
# -------------------------------------------------------------------
# File Name : predict_from_tfrecords.py
# Creation Date : 09-12-2016
# Last Modified : Mon Jan  9 16:04:01 2017
# Author: Thibaut Perol <tperol@g.harvard.edu>
# Lei Huang: change to Keras with tf.data generator
# -------------------------------------------------------------------
""" Prediction from a tfrecords. Create a catalog of found events
with their cluster id, cluster proba, start event time of the window, end
event time of the window
"""

import os
import setproctitle
import time
import argparse
import shutil

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
from obspy.core.utcdatetime import UTCDateTime


import quakenet.Keras_models as models
import quakenet.data_generator as dg
import quakenet.config as config
from tensorflow.keras.models import model_from_json

def main(args):
    setproctitle.setproctitle('quakenet_eval')

    if args.n_clusters == None:
        raise ValueError('Define the number of clusters with --n_clusters')

    ckpt = tf.train.get_checkpoint_state(args.checkpoint_dir)

    cfg = config.Config()
    #cfg.batch_size = 64
    cfg.n_clusters = args.n_clusters
    cfg.add = 1
    cfg.n_clusters += 1
    cfg.n_epochs = 1

    # Remove previous output directory
    if os.path.exists(args.output):
        shutil.rmtree(args.output)
    os.makedirs(args.output)
    if args.plot:
        os.makedirs(os.path.join(args.output,"viz"))

    # data pipeline
    data_generator = dg.DataGenerator(args.dataset, cfg, is_training=False)
    dataset = data_generator.read()
    #samples = {
    #    'data': data_pipeline.samples,
    #    'cluster_id': data_pipeline.labels,
    #    'start_time': data_pipeline.start_time,
    #    'end_time': data_pipeline.end_time}

    # set up model and validation metrics
    #model = models.get(args.model, samples, cfg,
    #                    args.checkpoint_dir,
    #                    is_training=False)

    # load json and create model
    json_file = open("models/model.json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("models/model.h5")
    print("Loaded model from disk")
    loaded_model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

    if args.max_windows is None:
        max_windows = 2**31
    else:
        max_windows = args.max_windows

    # Dictonary to store info on detected events
    events_dic ={"start_time": [],
                 "end_time": [],
                 "utc_timestamp": [],
                 "cluster_id": [],
                 "clusters_prob": []}

    # Create catalog name in which the events are stored
    output_catalog = os.path.join(args.output,'catalog_detection.csv')
    print('Catalog created to store events', output_catalog)
    n_events = 0
    idx = 0
    time_start = time.time()

    clusters_prob = loaded_model.predict(dataset,steps=max_windows)
    print(clusters_prob.shape)

    cluster_id = np.argmax(clusters_prob, axis=1)
    print(cluster_id.shape)

    features = []
    with tf.Session() as sess:
        data_generator = dg.DataGenerator(args.dataset, cfg, is_training=False)
        dataset = data_generator.getFeatures()
        dataset = dataset.batch(1)
        iterator = dataset.make_one_shot_iterator()
        n = iterator.get_next()
        for idx in range(cluster_id.shape[0]):
            values = sess.run(n)
            if (cluster_id[idx] > 0):
                features.append(dict({"start_time": values['start_time'],
                                     "end_time": values['end_time'],
                                      "cluster_id":cluster_id[idx]}))
                events_dic["start_time"].append(UTCDateTime(values['start_time']))
                events_dic["end_time"].append(UTCDateTime(values['end_time']))
                events_dic["utc_timestamp"].append((values['start_time'] +
                                                    values['end_time']) / 2.0)
                events_dic["cluster_id"].append(cluster_id[idx])
                events_dic["clusters_prob"].append(list(clusters_prob[idx]))

    for i in range(len(features)):
        print(i, features[i]['start_time'], features[i]['end_time'], features[i]['cluster_id'])

    # label for noise = -1, label for cluster \in {0:n_clusters}


    m, s = divmod(time.time() - time_start, 60)
    print("Prediction took {} min {} seconds".format(m,s))


    # Dump dictionary into csv file
    df = pd.DataFrame.from_dict(events_dic)
    df.to_csv(output_catalog)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",type=str,default=None,
                        help="path to tfrecords to analyze")
    parser.add_argument("--checkpoint_dir",type=str,default=None,
                        help="path to directory of chekpoints")
    parser.add_argument("--step",type=int,default=None,
                        help="step to load, if None the final step is loaded")
    parser.add_argument("--n_clusters",type=int,default=None,
                        help= 'n of clusters')
    parser.add_argument("--model",type=str,default="ConvNetQuake",
                        help="model to load")
    parser.add_argument("--max_windows",type=int,default=None,
                        help="number of windows to analyze")
    parser.add_argument("--output",type=str,default="output/predict",
                        help="dir of predicted events")
    parser.add_argument("--plot", action="store_true",
                     help="pass flag to plot detected events in output")
    args = parser.parse_args()

    main(args)

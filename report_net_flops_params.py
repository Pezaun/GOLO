#!/usr/bin/env python
from keras.models import Sequential
from keras.layers import Reshape, Activation, Convolution2D, Conv2D, Input, ZeroPadding2D, MaxPooling2D, BatchNormalization, Flatten, Dense
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adadelta, SGD, Adam, Adagrad
from keras.callbacks import TensorBoard, CSVLogger
from utils import parse_settings
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from keras import backend as K
import pickle
import math
from imgaug import augmenters as iaa
import feature_extractor
import loader
import cv2
import utils
import random
import time

# Workaround for not using all GPU memory
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

random.seed(1)
settings = {}


if __name__ == "__main__":
    settings = parse_settings(sys.argv[1])
    settings["ANCHORS"] = np.reshape(settings["ANCHORS"], [settings["DETECTORS"],2])
    run_meta = tf.RunMetadata()
    with tf.Session(graph=tf.Graph()) as sess:
        K.set_session(sess)
        K.set_learning_phase(0)
        fe = feature_extractor.FeatureExtractor(settings)

        nets = {"yolo_v1": (fe.yolo_convolutional_net, 0),
                "shot_yolo_A": (fe.shot_yolo_convolutional_net_A, 0),
                "shot_yolo_B": (fe.shot_yolo_convolutional_net_B, 0),
                "shot_yolo_C": (fe.shot_yolo_convolutional_net_C, 0),
                "yolo_tiny": (fe.tiny_yolo_convolutional_net, 0),
                "inceptionv3": (fe.inceptionv3_convolutional_net, 2),
                "mobilenetv2": (fe.mobilenetv2_convolutional_net, 0),
                "xception": (fe.xception_convolutional_net, 0)}

        net            = nets[settings["NET_ARCH"]][0](1)
        out_dim_factor = nets[settings["NET_ARCH"]][1]


        net = nets[settings["NET_ARCH"]][0](1)
        out_dim_factor = nets[settings["NET_ARCH"]][1]

        net.summary()
        net.load_weights(settings["PRED_MODEL"])

        opts = tf.profiler.ProfileOptionBuilder.float_operation()    
        flops = tf.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)
        print type(flops)
        print "{:,}".format(flops.total_float_ops)

        # opts = tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()    
        # params = tf.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)

        # print("{:,} --- {:,}".format(flops.total_float_ops, params.total_parameters))

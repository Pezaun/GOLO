#!/usr/bin/env python
from keras.models import Sequential
from keras.layers import Reshape, Activation, Convolution2D, Conv2D, Input, ZeroPadding2D, MaxPooling2D, BatchNormalization, Flatten, Dense
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adadelta, SGD, Adam, Adagrad
from keras.callbacks import TensorBoard, CSVLogger
from keras.models import Model
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
    bn_epsilon = 0.0
    with tf.Session(graph=tf.Graph()) as sess:
        K.set_session(sess)
        K.set_learning_phase(0)
        

        inputs = Input(shape=(32, 32, 3), name="input_0")
       
        net = Conv2D(32, 6, padding="valid", use_bias=False, name="0_conv")(inputs)
        net = Conv2D(64, 3, padding="valid", use_bias=True, name="1_conv")(net)
        # net = BatchNormalization(epsilon=bn_epsilon)(net)
        # net = LeakyReLU(alpha=0.1)(net)
        # net = MaxPooling2D(pool_size=(2,2), strides=(2,2), name="1_max")(net)
        net = Model(inputs, net)
        net.summary()


        # opts = tf.profiler.ProfileOptionBuilder.float_operation()    

        builder = tf.profiler.ProfileOptionBuilder
        opts = builder(builder.float_operation()).order_by('node name').build()


        flops = tf.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)
        print type(flops)
        print "{:,}".format(flops.total_float_ops)

        # opts = tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()    
        # params = tf.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)

        # print("{:,} --- {:,}".format(flops.total_float_ops, params.total_parameters))

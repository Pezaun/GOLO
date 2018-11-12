#!/usr/bin/env python
from keras.models import Sequential, Model
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


if __name__ == "__main__":
    settings = parse_settings(sys.argv[1])
    box_colors = [(int(random.random()*255),int(random.random()*255),int(random.random()*255)) for i in range(settings["CLASSES"])]
    settings["ANCHORS"] = np.reshape(settings["ANCHORS"], [settings["DETECTORS"],2])
    print "Predict..."
    fe = feature_extractor.FeatureExtractor(settings)

    nets = {"yolo_v1": (fe.yolo_convolutional_net, 0),
            "yolo_v1_att": (fe.yolo_convolutional_net_att, 0),
            "yolo_v1_att_2": (fe.yolo_convolutional_net_att_2, 0),
            "yolo_v1_att_sum": (fe.yolo_convolutional_net_att_sum, 0),
            "shot_yolo_A": (fe.shot_yolo_convolutional_net_A, 0),
            "shot_yolo_B": (fe.shot_yolo_convolutional_net_B, 0),
            "shot_yolo_C": (fe.shot_yolo_convolutional_net_C, 0),
            "yolo_tiny": (fe.tiny_yolo_convolutional_net, 0),
            "inceptionv3": (fe.inceptionv3_convolutional_net, 2),
            "mobilenetv2": (fe.mobilenetv2_convolutional_net, 0),
            "xception": (fe.xception_convolutional_net, 0),
            "golonet": (fe.golonet, 0),
            "golonetB": (fe.golonetB, 0)}

    # net            = nets[settings["NET_ARCH"]][0](1)
    # out_dim_factor = nets[settings["NET_ARCH"]][1]


    net_0 = nets["yolo_v1"][0](1)
    net_0.summary()
    net_0.load_weights(settings["PRED_MODEL"])
    out_dim_factor = nets[settings["NET_ARCH"]][1]


    net   = nets["yolo_v1_att_sum"][0](1)
    out_dim_factor = nets["yolo_v1_att_sum"][1]

    fe.surgery(net_0, net)

    # net = fe.extractor
    net.summary()
    net.save_weights("weights.h5")

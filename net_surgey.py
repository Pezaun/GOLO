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

def print_net(net):
    for layer in net.layers:
        print layer.name
        for w in layer.weights:
            print w.shape


def surgery(src, dest, layers):
    for name in layers:
        print "Exchange layer {}...".format(name)
        src_layer = src.get_layer(name)
        dest_layer = dest.get_layer(name)
        dest_layer.set_weights(src_layer.get_weights())
        

def print_means(net, layers):
    for name in layers:
        layer = net.get_layer(name)
        for weights in layer.get_weights():
            print "{}: {}".format(name, weights.mean())


if __name__ == "__main__":
    settings = parse_settings(sys.argv[1])
    print "Training on X_Dataset!"
    fe = feature_extractor.FeatureExtractor(settings)

    nets = {"yolo_v1": (fe.yolo_convolutional_net, 0),
            "shot_yolo_A": (fe.shot_yolo_convolutional_net_A, 0),    
            "shot_yolo_B": (fe.shot_yolo_convolutional_net_B, 0),    
            "shot_yolo_C": (fe.shot_yolo_convolutional_net_C, 0),    
            "yolo_tiny": (fe.tiny_yolo_convolutional_net, 0),
            "inceptionv3": (fe.inceptionv3_convolutional_net, 2),
            "mobilenetv1": (fe.mobilenetv1_convolutional_net, 0),
            "mobilenetv2": (fe.mobilenetv2_convolutional_net, 0),
            "xception": (fe.xception_convolutional_net, 0),
            "golonet": (fe.golonet, 0),
            "golonetB": (fe.golonetB, 0)}

    
    layers_to_exchange_shot_yolo_A = ["0_conv","2_conv","4_conv","5_conv","6_conv","8_conv","9_conv","10_conv","12_conv","13_conv","14_conv","15_conv","16_conv","18_conv","19_conv","20_conv","21_conv","22_conv","24_conv","last_conv"]
    layers_to_exchange_shot_yolo_B = ["0_conv","2_conv","4_conv","5_conv","6_conv","8_conv","9_conv","10_conv","12_conv","18_conv","19_conv","20_conv","21_conv","22_conv","24_conv","last_conv"]
    layers_to_exchange_shot_yolo_C = ["0_conv","2_conv","4_conv","5_conv","6_conv","8_conv","12_conv","18_conv","19_conv","22_conv","24_conv","last_conv"]

    layers_to_exchange = layers_to_exchange_shot_yolo_C

    net_name = "yolo_v1"
    netA = nets[net_name][0](1)
    netA.load_weights("/data/golo_models/output_xdataset_darknet19/golo_.286-0.45456647.hdf5")
    netA.summary()

    print net_name
    print_net(netA)
    print "\n\n\n"


    # net_name = "mobilenetv1"
    # netB = nets[net_name][0](1)
    # netB.summary()

    # print net_name
    # print_net(netB)


    net_name = "shot_yolo_C"
    netDest = nets[net_name][0](1)
    netDest.summary()

    print net_name
    print_net(netDest)

    print_means(netDest, layers_to_exchange)
    surgery(netA, netDest, layers_to_exchange)
    print_means(netDest, layers_to_exchange)
    print "End!"
    netDest.save("{}.h5".format(net_name))


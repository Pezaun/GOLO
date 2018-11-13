#!/usr/bin/env python
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import SeparableConv2D
from keras.layers.core import Flatten
from keras.layers.core import Reshape
from keras.layers.merge import Concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adadelta, SGD, Adam
from keras import backend as K
from keras.layers import Input, Convolution2D, MaxPooling2D, Activation, concatenate, Dropout, GlobalAveragePooling2D, warnings, Lambda, ReLU, Add, Multiply
from keras.engine.input_layer import InputLayer
from keras.utils import get_file
from keras.utils import layer_utils
from keras.models import load_model
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenetv2 import MobileNetV2
import keras
import tensorflow as tf
import numpy as np
import sys
from keras import backend as K
from utils import ReorgLayer


def reorg(input_tensor, stride=2, darknet=True):
    shapes = tf.shape(input_tensor)
    channel_first = tf.transpose(input_tensor,(0,3,1,2))
    reshape_tensor = tf.reshape(channel_first, (-1,shapes[3] // (stride ** 2), shapes[1], stride, shapes[2], stride))
    permute_tensor = tf.transpose(reshape_tensor,(0,3,5,1,2,4))
    target_tensor = tf.reshape(permute_tensor, (-1, shapes[3]*stride**2,shapes[1] // stride, shapes[2] // stride))
    channel_last = tf.transpose(target_tensor,(0,2,3,1))
    result = tf.reshape(channel_last, (-1,shapes[1]//stride, shapes[2]//stride, tf.cast(input_tensor.shape[3]*4, tf.int32)))
    return result

# def reorg(input_tensor, stride=2, darknet=True):
#     shapes = K.shape(input_tensor)
#     channel_first = K.permute_dimensions(input_tensor,(0,3,1,2))
#     reshape_tensor = K.reshape(channel_first, (-1,shapes[3] // (stride ** 2), shapes[1], stride, shapes[2], stride))
#     permute_tensor = K.permute_dimensions(reshape_tensor,(0,3,5,1,2,4))
#     target_tensor = K.reshape(permute_tensor, (-1, shapes[3]*stride**2,shapes[1] // stride, shapes[2] // stride))
#     channel_last = K.permute_dimensions(target_tensor,(0,2,3,1))
#     result = K.reshape(channel_last, (-1,shapes[1]//stride, shapes[2]//stride, input_tensor.shape[3]*4))
#     return result

def bottleneck_SES(x, expand=64, squeeze=16, name="bt_block"):
    m = Conv2D(squeeze, (1,1), padding="same", use_bias=False, kernel_initializer='glorot_uniform', name="{}_{}".format(name, "Conv2D_S1"))(x)
    # m = BatchNormalization(epsilon=0.0)(m)
    # m = ReLU(max_value=6)(m)
    m = LeakyReLU(alpha=0.1)(m)
    m = Conv2D(squeeze, (3,3), padding="same", use_bias=False, kernel_initializer='glorot_uniform', name="{}_{}".format(name, "Conv2D_S3"))(m)
    # m = BatchNormalization(epsilon=0.0)(m)
    # m = ReLU(max_value=6)(m)
    m = LeakyReLU(alpha=0.1)(m)
    m = Conv2D(expand, (1,1), padding="same", use_bias=False, kernel_initializer='glorot_uniform', name="{}_{}".format(name, "Conv2D_E"))(m)
    m = BatchNormalization(epsilon=0.0)(m)
    # m = ReLU(max_value=6)(m)
    m = LeakyReLU(alpha=0.1)(m)
    return Add(name="{}_{}".format(name, "Add"))([m, x])
    # return m

def space_to_depth_x2(x):
    return tf.space_to_depth(x, block_size=2)
    
class FeatureExtractor:
    def __init__(self, settings):
        self.extractor = None
        self.settings = settings
        self.true_boxes = Input(shape=(1, 1, 1, self.settings["MAX_BOXES"] , 4))
        self.anchors_map  = Input(shape=(None, None, self.settings["DETECTORS"], 1))
        # self.inputs = []
        self.outputs = []
        self.net = None


    def tiny_yolo_convolutional_net(self, version):
        bn_epsilon = 0.0
        inputs = Input(shape=(None, None, 3))
        # inputs = Input(shape=(416, 416, 3))

        net = Conv2D(16, 3, padding="same", use_bias=False, name="0_conv")(inputs)
        net = BatchNormalization(epsilon=bn_epsilon)(net)
        net = LeakyReLU(alpha=0.1)(net)
        net = MaxPooling2D(pool_size=(2,2), strides=(2,2), name="1_max")(net)

        net = Conv2D(32, 3, padding="same", use_bias=False, name="2_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon)(net)
        net = LeakyReLU(alpha=0.1)(net)
        net = MaxPooling2D(pool_size=(2,2), strides=(2,2), name="3_max")(net)

        net = Conv2D(64, 3, padding="same", use_bias=False, name="4_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon)(net)
        net = LeakyReLU(alpha=0.1)(net)
        net = MaxPooling2D(pool_size=(2,2), strides=(2,2), name="5_max")(net)

        net = Conv2D(128, 3, padding="same", use_bias=False, name="6_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon)(net)
        net = LeakyReLU(alpha=0.1)(net)
        net = MaxPooling2D(pool_size=(2,2), strides=(2,2), name="7_max")(net)
        
        net = Conv2D(256, 3, padding="same", use_bias=False, name="8_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon)(net)
        net = LeakyReLU(alpha=0.1)(net)
        net = MaxPooling2D(pool_size=(2,2), strides=(2,2), name="9_max")(net)

        net = Conv2D(512, 3, padding="same", use_bias=False, name="10_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon)(net)
        net = LeakyReLU(alpha=0.1)(net)
        net = MaxPooling2D(pool_size=(2,2), strides=(1,1), padding="same", name="11_max")(net)

        net = Conv2D(1024, 3, padding="same", use_bias=False, name="12_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon)(net)
        net = LeakyReLU(alpha=0.1)(net)

        net = Conv2D(1024, 3, padding="same", use_bias=False, name="13_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon)(net)
        net = LeakyReLU(alpha=0.1)(net)        

        net = Lambda(lambda args: args[0])([net, self.true_boxes])
        net = Lambda(lambda args: args[0])([net, self.anchors_map])
        if self.settings["FINETUNE"]:
            # VOC output to load pretrain model.
            net = Conv2D((125), 1, padding="same", use_bias=True)(net)
            self.extractor = Model([inputs, self.true_boxes, self.anchors_map], net)

            print "Loading model {} to tune...".format(self.settings["BASE_MODEL_WEIGHTS"]),
            self.extractor.load_weights(self.settings["BASE_MODEL_WEIGHTS"])
            print "Done!"
            print "Exchanging last layer...",
            net = Conv2D((self.settings["CLASSES"] + 5) * self.settings["DETECTORS"], 1, padding="same", use_bias=True, kernel_initializer='lecun_normal', name="last_conv")(self.extractor.layers[-2].output)
            self.extractor = Model([inputs, self.true_boxes, self.anchors_map], net)

            layer = self.extractor.layers[-1]
            weights = layer.get_weights()

            new_kernel = np.random.normal(size=weights[0].shape)/(13*13)
            new_bias   = np.random.normal(size=weights[1].shape)/(13*13)

            layer.set_weights([new_kernel, new_bias])
            print "Done!"
        else:
            net = Conv2D((self.settings["CLASSES"] + 5) * self.settings["DETECTORS"], 1, padding="same", use_bias=True, name="last_conv")(net)
            self.extractor = Model([inputs, self.true_boxes, self.anchors_map], net)
        self.outputs += [net]
        return self.extractor



    def golonet(self, size):
        inputs = Input(shape=(None, None, 3), name="input_0")
        net = Conv2D(32, 3, strides=1, padding="same", use_bias=False, name="0_conv")(inputs)
        net = BatchNormalization(epsilon=0.0)(net)
        # net = ReLU(max_value=6)(net)
        net = LeakyReLU(alpha=0.1)(net)

        depht = 32    
        net = bottleneck_SES(net, expand=depht, name="bt_block_0")
        
        # depht = 64
        # net = Conv2D(depht, 3, strides=2, padding="same", use_bias=False, name="0_squeeze_conv")(net)        
        # net = ReLU(max_value=6)(net)
        net = LeakyReLU(alpha=0.1)(net)
        # net = MaxPooling2D(pool_size=(2,2), strides=(2,2), name="1_max")(net)
        # net = bottleneck_SES(net, expand=depht, name="bt_block_1")

        net = MaxPooling2D(pool_size=(2,2), strides=(2,2), name="0_max")(net)
        net = Conv2D(64, 3, strides=1, padding="same", use_bias=False, name="2_conv")(net)
        net = BatchNormalization(epsilon=0.0)(net)
        # net = ReLU(max_value=6)(net)
        net = LeakyReLU(alpha=0.1)(net)
            
        depht = 64
        # net = Conv2D(depht, 3, strides=2, padding="same", use_bias=False, name="1_squeeze_conv")(net)
        # net = ReLU(max_value=6)(net)
        net = MaxPooling2D(pool_size=(2,2), strides=(2,2), name="2_max")(net)
        net = bottleneck_SES(net, expand=depht, squeeze=16, name="bt_block_2")

        net = Conv2D(128, 3, strides=1, padding="same", use_bias=False, name="4_conv")(net)
        net = BatchNormalization(epsilon=0.0)(net)
        # net = ReLU(max_value=6)(net)
        net = LeakyReLU(alpha=0.1)(net)

        depht = 128
        # net = Conv2D(depht, 3, strides=2, padding="same", use_bias=False, name="2_squeeze_conv")(net)
        # net = ReLU(max_value=6)(net)
        net = MaxPooling2D(pool_size=(2,2), strides=(2,2), name="3_max")(net)
        net = bottleneck_SES(net, expand=depht, squeeze=32, name="bt_block_3")

        net = Conv2D(256, 3, strides=1, padding="same", use_bias=False, name="8_conv")(net)
        # net = BatchNormalization(epsilon=0.0)(net)
        # net = ReLU(max_value=6)(net)
        net = LeakyReLU(alpha=0.1)(net)

        depht = 256
        # net = Conv2D(depht, 3, strides=2, padding="same", use_bias=False, name="3_squeeze_conv")(net)
        # net = ReLU(max_value=6)(net)
        net = MaxPooling2D(pool_size=(2,2), strides=(2,2), name="4_max")(net)
        net = bottleneck_SES(net, expand=depht, squeeze=64, name="bt_block_4")

        net = Conv2D(512, 3, strides=1, padding="same", use_bias=False, name="12_conv")(net)
        # net = BatchNormalization(epsilon=0.0)(net)
        # net = ReLU(max_value=6)(net)
        net = LeakyReLU(alpha=0.1)(net)

        depht = 512
        # net = Conv2D(depht, 3, strides=2, padding="same", use_bias=False, name="4_squeeze_conv")(net)
        # net = ReLU(max_value=6)(net)
        net = MaxPooling2D(pool_size=(2,2), strides=(2,2), name="5_max")(net)
        net = bottleneck_SES(net, expand=depht, squeeze=128, name="bt_block_5")

        net = Conv2D(1024, 3, padding="same", use_bias=False, name="18_conv")(net)
        # net = Conv2D(512, 1, padding="same", use_bias=False, name="19_conv")(net)
        net = BatchNormalization(epsilon=0.0)(net)
        # net = ReLU(max_value=6)(net)
        net = LeakyReLU(alpha=0.1)(net)

        net = Conv2D((self.settings["CLASSES"] + 5) * self.settings["DETECTORS"], 1, padding="same", use_bias=True, kernel_initializer='glorot_uniform', name="last_conv")(net)

        self.extractor = Model([inputs, self.true_boxes, self.anchors_map], net)
        self.outputs += [net]
        return self.extractor


    def golonetB(self, size):
        inputs = Input(shape=(None, None, 3), name="input_0")
        net = Conv2D(32, 3, strides=1, padding="same", use_bias=False, name="0_conv")(inputs)
        net = BatchNormalization(epsilon=0.0)(net)
        # net = ReLU(max_value=6)(net)
        net = LeakyReLU(alpha=0.1)(net)

        depht = 32    
        net = bottleneck_SES(net, expand=depht, name="bt_block_0")
        
        # depht = 64
        # net = Conv2D(depht, 3, strides=2, padding="same", use_bias=False, name="0_squeeze_conv")(net)        
        # net = ReLU(max_value=6)(net)
        net = MaxPooling2D(pool_size=(2,2), strides=(2,2), name="1_max")(net)
        net = bottleneck_SES(net, expand=depht, name="bt_block_1")

        net = Conv2D(64, 3, strides=1, padding="same", use_bias=False, name="2_conv")(net)
        net = BatchNormalization(epsilon=0.0)(net)
        # net = ReLU(max_value=6)(net)
        net = LeakyReLU(alpha=0.1)(net)
            
        depht = 64
        # net = Conv2D(depht, 3, strides=2, padding="same", use_bias=False, name="1_squeeze_conv")(net)
        # net = ReLU(max_value=6)(net)
        net = MaxPooling2D(pool_size=(2,2), strides=(2,2), name="2_max")(net)
        net = bottleneck_SES(net, expand=depht, squeeze=16, name="bt_block_2")

        net = Conv2D(128, 3, strides=1, padding="same", use_bias=False, name="4_conv")(net)
        net = BatchNormalization(epsilon=0.0)(net)
        # net = ReLU(max_value=6)(net)
        net = LeakyReLU(alpha=0.1)(net)

        depht = 128
        # net = Conv2D(depht, 3, strides=2, padding="same", use_bias=False, name="2_squeeze_conv")(net)
        # net = ReLU(max_value=6)(net)
        net = MaxPooling2D(pool_size=(2,2), strides=(2,2), name="3_max")(net)
        net = bottleneck_SES(net, expand=depht, squeeze=32, name="bt_block_3")

        net = Conv2D(256, 3, strides=1, padding="same", use_bias=False, name="8_conv")(net)
        net = BatchNormalization(epsilon=0.0)(net)
        # net = ReLU(max_value=6)(net)
        net = LeakyReLU(alpha=0.1)(net)

        depht = 256
        # net = Conv2D(depht, 3, strides=2, padding="same", use_bias=False, name="3_squeeze_conv")(net)
        # net = ReLU(max_value=6)(net)
        net = MaxPooling2D(pool_size=(2,2), strides=(2,2), name="4_max")(net)
        net = bottleneck_SES(net, expand=depht, squeeze=64, name="bt_block_4")

        net = Conv2D(512, 3, strides=1, padding="same", use_bias=False, name="12_conv")(net)
        net = BatchNormalization(epsilon=0.0)(net)
        # net = ReLU(max_value=6)(net)
        net = LeakyReLU(alpha=0.1)(net)

        depht = 512
        # net = Conv2D(depht, 3, strides=2, padding="same", use_bias=False, name="4_squeeze_conv")(net)
        # net = ReLU(max_value=6)(net)
        net = MaxPooling2D(pool_size=(2,2), strides=(2,2), name="5_max")(net)
        net = bottleneck_SES(net, expand=depht, squeeze=128, name="bt_block_5")

        net = Conv2D((self.settings["CLASSES"] + 5) * self.settings["DETECTORS"], 1, padding="same", use_bias=True, kernel_initializer='glorot_uniform', name="last_conv")(net)

        self.extractor = Model([inputs, self.true_boxes, self.anchors_map], net)
        self.outputs += [net]
        return self.extractor


    def mobilenetv2_convolutional_net(self,version):
        bn_epsilon = 0.0        
        model = MobileNetV2(include_top=False)
        input_data = model.layers[0].input
        output_data = model.layers[-1].output

        net = Conv2D((self.settings["CLASSES"] + 5) * self.settings["DETECTORS"], 1, padding="same", use_bias=True, kernel_initializer='glorot_uniform', name="last_conv")(output_data)
        self.extractor = Model([input_data, self.true_boxes, self.anchors_map], net)
        self.outputs += [net]
        return self.extractor

    def mobilenetv1_convolutional_net(self,version):
        bn_epsilon = 0.0        
        model = MobileNet(include_top=False)
        input_data = model.layers[0].input
        output_data = model.layers[-1].output

        net = Conv2D((self.settings["CLASSES"] + 5) * self.settings["DETECTORS"], 1, padding="same", use_bias=True, kernel_initializer='glorot_uniform', name="last_conv")(output_data)
        self.extractor = Model([input_data, self.true_boxes, self.anchors_map], net)
        self.outputs += [net]
        return self.extractor


    def inceptionv3_convolutional_net(self,version):
        bn_epsilon = 0.0
        model = InceptionV3(include_top=False)
        input_data = model.layers[0].input
        output_data = model.layers[-1].output

        net = Conv2D((self.settings["CLASSES"] + 5) * self.settings["DETECTORS"], 1, padding="same", use_bias=True, kernel_initializer='glorot_uniform', name="last_conv")(output_data)
        self.extractor = Model([input_data, self.true_boxes, self.anchors_map], net)
        self.outputs += [net]
        return self.extractor
    

    def xception_convolutional_net(self,version):
        bn_epsilon = 0.0
        model = Xception(include_top=False)
        input_data = model.layers[0].input
        output_data = model.layers[-1].output

        net = Conv2D((self.settings["CLASSES"] + 5) * self.settings["DETECTORS"], 1, padding="same", use_bias=True, kernel_initializer='glorot_uniform', name="last_conv")(output_data)
        self.extractor = Model([input_data, self.true_boxes, self.anchors_map], net)
        self.outputs += [net]
        return self.extractor
        

    def yolo_convolutional_net(self,version):
        bn_epsilon = 0.0
        inputs = Input(shape=(None, None, 3))
        # inputs = Input(batch_shape=(1,416,416,3))
       
        net = Conv2D(32, 3, padding="same", use_bias=False, name="0_conv")(inputs)
        net = BatchNormalization(epsilon=bn_epsilon, name="0_bn")(net)
        net = LeakyReLU(alpha=0.1)(net)
        net = MaxPooling2D(pool_size=(2,2), strides=(2,2), name="1_max")(net)

        net = Conv2D(64, 3, padding="same", use_bias=False, name="2_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon, name="2_bn")(net)
        net = LeakyReLU(alpha=0.1)(net)
        net = MaxPooling2D(pool_size=(2,2), strides=(2,2), name="3_max")(net)

        net = Conv2D(128, 3, padding="same", use_bias=False, name="4_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon, name="4_bn")(net)
        net = LeakyReLU(alpha=0.1)(net)

        net = Conv2D(64, 1, padding="same", use_bias=False, name="5_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon, name="5_bn")(net)
        net = LeakyReLU(alpha=0.1)(net)
        
        net = Conv2D(128, 3, padding="same", use_bias=False, name="6_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon, name="6_bn")(net)
        net = LeakyReLU(alpha=0.1)(net)
        net = MaxPooling2D(pool_size=(2,2), strides=(2,2), name="7_max")(net)

        net = Conv2D(256, 3, padding="same", use_bias=False, name="8_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon, name="8_bn")(net)
        net = LeakyReLU(alpha=0.1)(net)

        net = Conv2D(128, 1, padding="same", use_bias=False, name="9_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon, name="9_bn")(net)
        net = LeakyReLU(alpha=0.1)(net)

        net = Conv2D(256, 3, padding="same", use_bias=False, name="10_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon, name="10_bn")(net)
        net = LeakyReLU(alpha=0.1)(net)
        net = MaxPooling2D(pool_size=(2,2), strides=(2,2), name="11_max")(net)

        net = Conv2D(512, 3, padding="same", use_bias=False, name="12_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon, name="12_bn")(net)
        net = LeakyReLU(alpha=0.1)(net)

        net = Conv2D(256, 1, padding="same", use_bias=False, name="13_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon, name="13_bn")(net)
        net = LeakyReLU(alpha=0.1)(net)

        net = Conv2D(512, 3, padding="same", use_bias=False, name="14_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon, name="14_bn")(net)
        net = LeakyReLU(alpha=0.1)(net)

        net = Conv2D(256, 1, padding="same", use_bias=False, name="15_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon, name="15_bn")(net)
        net = LeakyReLU(alpha=0.1)(net)

        net = Conv2D(512, 3, padding="same", use_bias=False, name="16_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon, name="16_bn")(net)
        c16 = LeakyReLU(alpha=0.1)(net)
        net = MaxPooling2D(pool_size=(2,2), strides=(2,2), name="17_max")(c16)

        net = Conv2D(1024, 3, padding="same", use_bias=False, name="18_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon, name="18_bn")(net)
        net = LeakyReLU(alpha=0.1)(net)

        net = Conv2D(512, 1, padding="same", use_bias=False, name="19_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon, name="19_bn")(net)
        net = LeakyReLU(alpha=0.1)(net)

        net = Conv2D(1024, 3, padding="same", use_bias=False, name="20_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon, name="20_bn")(net)
        net = LeakyReLU(alpha=0.1)(net)

        net = Conv2D(512, 1, padding="same", use_bias=False, name="21_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon, name="21_bn")(net)
        net = LeakyReLU(alpha=0.1)(net)

        net = Conv2D(1024, 3, padding="same", use_bias=False, name="22_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon, name="22_bn")(net)
        net = LeakyReLU(alpha=0.1)(net)

        net = Conv2D(1024, 3, padding="same", use_bias=False, name="23_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon, name="23_bn")(net)
        net = LeakyReLU(alpha=0.1)(net)

        net = Conv2D(1024, 3, padding="same", use_bias=False, name="24_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon, name="24_bn")(net)
        net = LeakyReLU(alpha=0.1)(net)

        conv_id = 25
        # if version == 1:
        if True:
            c16 = Conv2D(64, 1, padding="same", use_bias=False, name="{}_conv".format(conv_id))(c16)
            conv_id += 1
            c16 = BatchNormalization(epsilon=bn_epsilon, name="{}_bn".format(conv_id))(c16)
            c16 = LeakyReLU(alpha=0.1)(c16)            

        # c16 = Lambda(space_to_depth_x2)(c16)
        # c16 = Lambda(reorg)(c16)   
        c16 = ReorgLayer()(c16)     
        net = Concatenate()([c16, net])

        net = Conv2D(1024, 3, padding="same", use_bias=False, name="{}_conv".format(conv_id))(net)
        conv_id += 1
        net = BatchNormalization(epsilon=bn_epsilon, name="{}_bn".format(conv_id))(net)
        net = LeakyReLU(alpha=0.1)(net)        


        net = Lambda(lambda args: args[0])([net, self.true_boxes])
        net = Lambda(lambda args: args[0])([net, self.anchors_map])
        if self.settings["FINETUNE"]:
            # VOC output to load pretrain model.
            net = Conv2D((self.settings["BASE_MODEL_CLASSES"] + 5) * self.settings["DETECTORS"], 1, padding="same", use_bias=True)(net)
            self.extractor = Model([inputs, self.true_boxes, self.anchors_map], net)

            print "Loading model {} to tune...".format(self.settings["BASE_MODEL_WEIGHTS"]),
            if version == 1:
                self.extractor.load_weights(self.settings["BASE_MODEL_WEIGHTS"])
            print "Done!"
            print "Exchanging last layer...",
            net = Conv2D((self.settings["CLASSES"] + 5) * self.settings["DETECTORS"], 1, padding="same", use_bias=True, kernel_initializer='glorot_uniform', name="last_conv")(self.extractor.layers[-2].output)
            self.extractor = Model([inputs, self.true_boxes, self.anchors_map], net)

            # layer = self.extractor.layers[-1]
            # weights = layer.get_weights()

            # new_kernel = np.random.normal(size=weights[0].shape)/(13*13)
            # new_bias   = np.random.normal(size=weights[1].shape)/(13*13)

            # layer.set_weights([new_kernel, new_bias])
            print "Done!"
        else:
            net = Conv2D((self.settings["BASE_MODEL_CLASSES"] + 5) * self.settings["DETECTORS"], 1, padding="same", use_bias=True, name="last_conv")(net)
            self.extractor = Model([inputs, self.true_boxes, self.anchors_map], net)
        self.outputs += [net]
        return self.extractor



    def yolo_convolutional_net_att(self,version):
        bn_epsilon = 0.0
        inputs = Input(shape=(None, None, 3))
       
        net = Conv2D(32, 3, padding="same", use_bias=False, name="0_conv")(inputs)
        net = BatchNormalization(epsilon=bn_epsilon)(net)
        net = LeakyReLU(alpha=0.1)(net)
        net = MaxPooling2D(pool_size=(2,2), strides=(2,2), name="1_max")(net)

        net = Conv2D(64, 3, padding="same", use_bias=False, name="2_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon)(net)
        net = LeakyReLU(alpha=0.1)(net)
        net = MaxPooling2D(pool_size=(2,2), strides=(2,2), name="3_max")(net)

        net = Conv2D(128, 3, padding="same", use_bias=False, name="4_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon)(net)
        net = LeakyReLU(alpha=0.1)(net)

        net = Conv2D(64, 1, padding="same", use_bias=False, name="5_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon)(net)
        net = LeakyReLU(alpha=0.1)(net)
        
        net = Conv2D(128, 3, padding="same", use_bias=False, name="6_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon)(net)
        net = LeakyReLU(alpha=0.1)(net)
        net = MaxPooling2D(pool_size=(2,2), strides=(2,2), name="7_max")(net)

        net = Conv2D(256, 3, padding="same", use_bias=False, name="8_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon)(net)
        net = LeakyReLU(alpha=0.1)(net)

        net = Conv2D(128, 1, padding="same", use_bias=False, name="9_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon)(net)
        net = LeakyReLU(alpha=0.1)(net)

        net = Conv2D(256, 3, padding="same", use_bias=False, name="10_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon)(net)
        net = LeakyReLU(alpha=0.1)(net)
        net = MaxPooling2D(pool_size=(2,2), strides=(2,2), name="11_max")(net)

        net = Conv2D(512, 3, padding="same", use_bias=False, name="12_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon)(net)
        net = LeakyReLU(alpha=0.1)(net)

        net = Conv2D(256, 1, padding="same", use_bias=False, name="13_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon)(net)
        net = LeakyReLU(alpha=0.1)(net)

        net = Conv2D(512, 3, padding="same", use_bias=False, name="14_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon)(net)
        net = LeakyReLU(alpha=0.1)(net)

        net = Conv2D(256, 1, padding="same", use_bias=False, name="15_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon)(net)
        net = LeakyReLU(alpha=0.1)(net)

        net = Conv2D(512, 3, padding="same", use_bias=False, name="16_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon)(net)
        c16 = LeakyReLU(alpha=0.1)(net)
        net = MaxPooling2D(pool_size=(2,2), strides=(2,2), name="17_max")(c16)

        net = Conv2D(1024, 3, padding="same", use_bias=False, name="18_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon)(net)
        net = LeakyReLU(alpha=0.1)(net)

        net = Conv2D(512, 1, padding="same", use_bias=False, name="19_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon)(net)
        net = LeakyReLU(alpha=0.1)(net)

        net = Conv2D(1024, 3, padding="same", use_bias=False, name="20_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon)(net)
        net = LeakyReLU(alpha=0.1)(net)

        net = Conv2D(512, 1, padding="same", use_bias=False, name="21_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon)(net)
        net = LeakyReLU(alpha=0.1)(net)

        net = Conv2D(1024, 3, padding="same", use_bias=False, name="22_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon)(net)
        net = LeakyReLU(alpha=0.1)(net)

        net = Conv2D(1024, 3, padding="same", use_bias=False, name="23_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon)(net)
        net = LeakyReLU(alpha=0.1)(net)

        net = Conv2D(1024, 3, padding="same", use_bias=False, name="24_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon)(net)
        net = LeakyReLU(alpha=0.1)(net)

        conv_id = 25
        if version == 1:
            c16 = Conv2D(64, 1, padding="same", use_bias=False, name="{}_conv".format(conv_id))(c16)
            conv_id += 1
            c16 = BatchNormalization(epsilon=bn_epsilon)(c16)
            c16 = LeakyReLU(alpha=0.1)(c16)            

        # c16 = Lambda(space_to_depth_x2)(c16)
        c16 = Lambda(reorg)(c16)        
        net = Concatenate()([c16, net])

        net = Conv2D(1024, 3, padding="same", use_bias=False, name="{}_conv".format(conv_id))(net)
        conv_id += 1
        net = BatchNormalization(epsilon=bn_epsilon)(net)
        net = LeakyReLU(alpha=0.1)(net)        


        net = Lambda(lambda args: args[0])([net, self.true_boxes])
        net = Lambda(lambda args: args[0])([net, self.anchors_map])
        if self.settings["FINETUNE"]:
            net = Conv2D((self.settings["BASE_MODEL_CLASSES"] + 5) * self.settings["DETECTORS"], 1, padding="same", use_bias=True)(net)
            self.extractor = Model([inputs, self.true_boxes, self.anchors_map], net)

            print "Loading model {} to tune...".format(self.settings["BASE_MODEL_WEIGHTS"]),
            self.extractor.load_weights(self.settings["BASE_MODEL_WEIGHTS"])
            print "Done!"
            print "Exchanging last layer...",
            last_layer = self.extractor.layers[-2].output

            att_layer = Conv2D((self.settings["CLASSES"] + 5) * self.settings["DETECTORS"], 1, padding="same", use_bias=True, kernel_initializer='glorot_uniform', name="att_conv", activation="softmax")(last_layer)

            net = Conv2D((self.settings["CLASSES"] + 5) * self.settings["DETECTORS"], 1, padding="same", use_bias=True, kernel_initializer='glorot_uniform', name="last_conv")(last_layer)
            net = Multiply()([net, att_layer])
            self.extractor = Model([inputs, self.true_boxes, self.anchors_map], net)
            print "Done!"
        else:
            net = Conv2D((self.settings["BASE_MODEL_CLASSES"] + 5) * self.settings["DETECTORS"], 1, padding="same", use_bias=True, name="last_conv")(net)
            self.extractor = Model([inputs, self.true_boxes, self.anchors_map], net)
        self.outputs += [net]
        return self.extractor


    def yolo_convolutional_net_att_2(self,version):
        bn_epsilon = 0.0
        inputs = Input(shape=(None, None, 3))
        # inputs = Input(batch_shape=(1,416,416,3))

        # ATT
        net_att = Conv2D(32, 3, padding="same", use_bias=False, name="0_conv_att")(inputs)
        net_att = BatchNormalization(epsilon=bn_epsilon, name="0_bn_att")(net_att)
        net_att = LeakyReLU(alpha=0.1)(net_att)
        net_att = MaxPooling2D(pool_size=(2,2), strides=(2,2), name="1_max_att")(net_att)

        net_att = Conv2D(64, 3, padding="same", use_bias=False, name="1_conv_att")(net_att)
        net_att = BatchNormalization(epsilon=bn_epsilon, name="1_bn_att")(net_att)
        net_att = LeakyReLU(alpha=0.1)(net_att)
        net_att = MaxPooling2D(pool_size=(2,2), strides=(2,2), name="2_max_att")(net_att)

        net_att = Conv2D(128, 3, padding="same", use_bias=False, name="2_conv_att")(net_att)
        net_att = BatchNormalization(epsilon=bn_epsilon, name="2_bn_att")(net_att)
        net_att = LeakyReLU(alpha=0.1)(net_att)
        net_att = MaxPooling2D(pool_size=(2,2), strides=(2,2), name="3_max_att")(net_att)

        net_att = Conv2D(256, 3, padding="same", use_bias=False, name="3_conv_att")(net_att)
        net_att = BatchNormalization(epsilon=bn_epsilon, name="3_bn_att")(net_att)
        net_att = LeakyReLU(alpha=0.1)(net_att)
        net_att = MaxPooling2D(pool_size=(2,2), strides=(2,2), name="4_max_att")(net_att)

        net_att = Conv2D(1280, 3, padding="same", use_bias=True, name="4_conv_att")(net_att)
        net_att = BatchNormalization(epsilon=bn_epsilon, name="4_bn_att")(net_att)
        net_att = MaxPooling2D(pool_size=(2,2), strides=(2,2), name="5_max_att")(net_att)
        net_att = Activation("softmax")(net_att)

        # END ATT
       
        net = Conv2D(32, 3, padding="same", use_bias=False, name="0_conv")(inputs)
        net = BatchNormalization(epsilon=bn_epsilon, name="0_bn")(net)
        net = LeakyReLU(alpha=0.1)(net)
        net = MaxPooling2D(pool_size=(2,2), strides=(2,2), name="1_max")(net)

        net = Conv2D(64, 3, padding="same", use_bias=False, name="2_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon, name="2_bn")(net)
        net = LeakyReLU(alpha=0.1)(net)

        net = MaxPooling2D(pool_size=(2,2), strides=(2,2), name="3_max")(net)

        net = Conv2D(128, 3, padding="same", use_bias=False, name="4_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon, name="4_bn")(net)
        net = LeakyReLU(alpha=0.1)(net)

        net = Conv2D(64, 1, padding="same", use_bias=False, name="5_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon, name="5_bn")(net)
        net = LeakyReLU(alpha=0.1)(net)
        
        net = Conv2D(128, 3, padding="same", use_bias=False, name="6_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon, name="6_bn")(net)
        net = LeakyReLU(alpha=0.1)(net)        

        net = MaxPooling2D(pool_size=(2,2), strides=(2,2), name="7_max")(net)

        net = Conv2D(256, 3, padding="same", use_bias=False, name="8_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon, name="8_bn")(net)
        net = LeakyReLU(alpha=0.1)(net)

        net = Conv2D(128, 1, padding="same", use_bias=False, name="9_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon, name="9_bn")(net)
        net = LeakyReLU(alpha=0.1)(net)

        net = Conv2D(256, 3, padding="same", use_bias=False, name="10_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon, name="10_bn")(net)
        net = LeakyReLU(alpha=0.1)(net)

        net = MaxPooling2D(pool_size=(2,2), strides=(2,2), name="11_max")(net)

        net = Conv2D(512, 3, padding="same", use_bias=False, name="12_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon, name="12_bn")(net)
        net = LeakyReLU(alpha=0.1)(net)

        net = Conv2D(256, 1, padding="same", use_bias=False, name="13_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon, name="13_bn")(net)
        net = LeakyReLU(alpha=0.1)(net)


        net = Conv2D(512, 3, padding="same", use_bias=False, name="14_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon, name="14_bn")(net)
        net = LeakyReLU(alpha=0.1)(net)

        net = Conv2D(256, 1, padding="same", use_bias=False, name="15_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon, name="15_bn")(net)
        net = LeakyReLU(alpha=0.1)(net)


        net = Conv2D(512, 3, padding="same", use_bias=False, name="16_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon, name="16_bn")(net)
        c16 = LeakyReLU(alpha=0.1)(net)
        net = MaxPooling2D(pool_size=(2,2), strides=(2,2), name="17_max")(c16)

        net = Conv2D(1024, 3, padding="same", use_bias=False, name="18_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon, name="18_bn")(net)
        net = LeakyReLU(alpha=0.1)(net)

        net = Conv2D(512, 1, padding="same", use_bias=False, name="19_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon, name="19_bn")(net)
        net = LeakyReLU(alpha=0.1)(net)

        net = Conv2D(1024, 3, padding="same", use_bias=False, name="20_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon, name="20_bn")(net)
        net = LeakyReLU(alpha=0.1)(net)

        net = Conv2D(512, 1, padding="same", use_bias=False, name="21_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon, name="21_bn")(net)
        net = LeakyReLU(alpha=0.1)(net)

        net = Conv2D(1024, 3, padding="same", use_bias=False, name="22_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon, name="22_bn")(net)
        net = LeakyReLU(alpha=0.1)(net)

        net = Conv2D(1024, 3, padding="same", use_bias=False, name="23_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon, name="23_bn")(net)
        net = LeakyReLU(alpha=0.1)(net)

        net = Conv2D(1024, 3, padding="same", use_bias=False, name="24_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon, name="24_bn")(net)
        net = LeakyReLU(alpha=0.1)(net)

        conv_id = 25
        if version == 1:
            c16 = Conv2D(64, 1, padding="same", use_bias=False, name="{}_conv".format(conv_id))(c16)
            conv_id += 1
            c16 = BatchNormalization(epsilon=bn_epsilon, name="{}_bn".format(conv_id))(c16)
            c16 = LeakyReLU(alpha=0.1)(c16)            

        c16 = Lambda(space_to_depth_x2)(c16)
        # c16 = Lambda(reorg)(c16)        
        net = Concatenate()([c16, net])
        net = Multiply()([net, net_att])

        net = Conv2D(1024, 3, padding="same", use_bias=False, name="{}_conv".format(conv_id))(net)
        conv_id += 1
        net = BatchNormalization(epsilon=bn_epsilon, name="{}_bn".format(conv_id))(net)
        net = LeakyReLU(alpha=0.1)(net)        


        net = Lambda(lambda args: args[0])([net, self.true_boxes])
        net = Lambda(lambda args: args[0])([net, self.anchors_map])        


        if self.settings["FINETUNE"]:
            net = Conv2D((self.settings["BASE_MODEL_CLASSES"] + 5) * self.settings["DETECTORS"], 1, padding="same", use_bias=True)(net)
            self.extractor = Model([inputs, self.true_boxes, self.anchors_map], net)

            print "Loading model {} to tune...".format(self.settings["BASE_MODEL_WEIGHTS"]),
            self.extractor.load_weights(self.settings["BASE_MODEL_WEIGHTS"])
            print "Done!"
            print "Exchanging last layer...",
            net = Conv2D((self.settings["CLASSES"] + 5) * self.settings["DETECTORS"], 1, padding="same", use_bias=True, kernel_initializer='glorot_uniform', name="last_conv")(self.extractor.layers[-2].output)
            # net = Multiply()([net, net_att])
            self.extractor = Model([inputs, self.true_boxes, self.anchors_map], net)
            print "Done!"
        else:
            net = Conv2D((self.settings["CLASSES"] + 5) * self.settings["DETECTORS"], 1, padding="same", use_bias=True, name="last_conv")(net)
            # net = Multiply()([net, net_att])
            self.extractor = Model([inputs, self.true_boxes, self.anchors_map], net)
        self.outputs += [net]
        return self.extractor


    def yolo_convolutional_net_att_sum(self,version):
        bn_epsilon = 0.0
        inputs = Input(shape=(None, None, 3))
        # inputs = Input(batch_shape=(1,416,416,3))

        # ATT
        net_att = Conv2D(32, 3, padding="same", use_bias=False, name="0_conv_att")(inputs)
        net_att = BatchNormalization(epsilon=bn_epsilon, name="0_bn_att")(net_att)
        net_att = LeakyReLU(alpha=0.1)(net_att)
        net_att = MaxPooling2D(pool_size=(2,2), strides=(2,2), name="1_max_att")(net_att)

        net_att = Conv2D(64, 3, padding="same", use_bias=False, name="1_conv_att")(net_att)
        net_att = BatchNormalization(epsilon=bn_epsilon, name="1_bn_att")(net_att)
        net_att = LeakyReLU(alpha=0.1)(net_att)
        net_att = MaxPooling2D(pool_size=(2,2), strides=(2,2), name="2_max_att")(net_att)

        net_att = Conv2D(128, 3, padding="same", use_bias=False, name="2_conv_att")(net_att)
        net_att = BatchNormalization(epsilon=bn_epsilon, name="2_bn_att")(net_att)
        net_att = LeakyReLU(alpha=0.1)(net_att)
        net_att = MaxPooling2D(pool_size=(2,2), strides=(2,2), name="3_max_att")(net_att)

        net_att = Conv2D(256, 3, padding="same", use_bias=False, name="3_conv_att")(net_att)
        net_att = BatchNormalization(epsilon=bn_epsilon, name="3_bn_att")(net_att)
        net_att = LeakyReLU(alpha=0.1)(net_att)
        net_att = MaxPooling2D(pool_size=(2,2), strides=(2,2), name="4_max_att")(net_att)

        net_att = Conv2D(1280, 3, padding="same", use_bias=True, name="4_conv_att")(net_att)
        net_att = BatchNormalization(epsilon=bn_epsilon, name="4_bn_att")(net_att)
        net_att = MaxPooling2D(pool_size=(2,2), strides=(2,2), name="5_max_att")(net_att)
        net_att = Activation("softmax")(net_att)

        # END ATT
       
        net = Conv2D(32, 3, padding="same", use_bias=False, name="0_conv")(inputs)
        net = BatchNormalization(epsilon=bn_epsilon, name="0_bn")(net)
        net = LeakyReLU(alpha=0.1)(net)
        net = MaxPooling2D(pool_size=(2,2), strides=(2,2), name="1_max")(net)

        net = Conv2D(64, 3, padding="same", use_bias=False, name="2_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon, name="2_bn")(net)
        net = LeakyReLU(alpha=0.1)(net)

        net = MaxPooling2D(pool_size=(2,2), strides=(2,2), name="3_max")(net)

        net = Conv2D(128, 3, padding="same", use_bias=False, name="4_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon, name="4_bn")(net)
        net = LeakyReLU(alpha=0.1)(net)

        net = Conv2D(64, 1, padding="same", use_bias=False, name="5_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon, name="5_bn")(net)
        net = LeakyReLU(alpha=0.1)(net)
        
        net = Conv2D(128, 3, padding="same", use_bias=False, name="6_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon, name="6_bn")(net)
        net = LeakyReLU(alpha=0.1)(net)        

        net = MaxPooling2D(pool_size=(2,2), strides=(2,2), name="7_max")(net)

        net = Conv2D(256, 3, padding="same", use_bias=False, name="8_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon, name="8_bn")(net)
        net = LeakyReLU(alpha=0.1)(net)

        net = Conv2D(128, 1, padding="same", use_bias=False, name="9_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon, name="9_bn")(net)
        net = LeakyReLU(alpha=0.1)(net)

        net = Conv2D(256, 3, padding="same", use_bias=False, name="10_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon, name="10_bn")(net)
        net = LeakyReLU(alpha=0.1)(net)

        net = MaxPooling2D(pool_size=(2,2), strides=(2,2), name="11_max")(net)

        net = Conv2D(512, 3, padding="same", use_bias=False, name="12_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon, name="12_bn")(net)
        net = LeakyReLU(alpha=0.1)(net)

        net = Conv2D(256, 1, padding="same", use_bias=False, name="13_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon, name="13_bn")(net)
        net = LeakyReLU(alpha=0.1)(net)


        net = Conv2D(512, 3, padding="same", use_bias=False, name="14_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon, name="14_bn")(net)
        net = LeakyReLU(alpha=0.1)(net)

        net = Conv2D(256, 1, padding="same", use_bias=False, name="15_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon, name="15_bn")(net)
        net = LeakyReLU(alpha=0.1)(net)


        net = Conv2D(512, 3, padding="same", use_bias=False, name="16_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon, name="16_bn")(net)
        c16 = LeakyReLU(alpha=0.1)(net)
        net = MaxPooling2D(pool_size=(2,2), strides=(2,2), name="17_max")(c16)

        net = Conv2D(1024, 3, padding="same", use_bias=False, name="18_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon, name="18_bn")(net)
        net = LeakyReLU(alpha=0.1)(net)

        net = Conv2D(512, 1, padding="same", use_bias=False, name="19_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon, name="19_bn")(net)
        net = LeakyReLU(alpha=0.1)(net)

        net = Conv2D(1024, 3, padding="same", use_bias=False, name="20_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon, name="20_bn")(net)
        net = LeakyReLU(alpha=0.1)(net)

        net = Conv2D(512, 1, padding="same", use_bias=False, name="21_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon, name="21_bn")(net)
        net = LeakyReLU(alpha=0.1)(net)

        net = Conv2D(1024, 3, padding="same", use_bias=False, name="22_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon, name="22_bn")(net)
        net = LeakyReLU(alpha=0.1)(net)

        net = Conv2D(1024, 3, padding="same", use_bias=False, name="23_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon, name="23_bn")(net)
        net = LeakyReLU(alpha=0.1)(net)

        net = Conv2D(1024, 3, padding="same", use_bias=False, name="24_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon, name="24_bn")(net)
        net = LeakyReLU(alpha=0.1)(net)

        conv_id = 25
        if version == 1:
            c16 = Conv2D(64, 1, padding="same", use_bias=False, name="{}_conv".format(conv_id))(c16)
            conv_id += 1
            c16 = BatchNormalization(epsilon=bn_epsilon, name="{}_bn".format(conv_id))(c16)
            c16 = LeakyReLU(alpha=0.1)(c16)            

        c16 = Lambda(space_to_depth_x2)(c16)
        # c16 = Lambda(reorg)(c16)        
        net = Concatenate()([c16, net])
        net = Add()([net, net_att])

        net = Conv2D(1024, 3, padding="same", use_bias=False, name="{}_conv".format(conv_id))(net)
        conv_id += 1
        net = BatchNormalization(epsilon=bn_epsilon, name="{}_bn".format(conv_id))(net)
        net = LeakyReLU(alpha=0.1)(net)        


        net = Lambda(lambda args: args[0])([net, self.true_boxes])
        net = Lambda(lambda args: args[0])([net, self.anchors_map])        


        if self.settings["FINETUNE"]:
            net = Conv2D((self.settings["BASE_MODEL_CLASSES"] + 5) * self.settings["DETECTORS"], 1, padding="same", use_bias=True)(net)
            self.extractor = Model([inputs, self.true_boxes, self.anchors_map], net)

            print "Loading model {} to tune...".format(self.settings["BASE_MODEL_WEIGHTS"]),
            self.extractor.load_weights(self.settings["BASE_MODEL_WEIGHTS"])
            print "Done!"
            print "Exchanging last layer...",
            net = Conv2D((self.settings["CLASSES"] + 5) * self.settings["DETECTORS"], 1, padding="same", use_bias=True, kernel_initializer='glorot_uniform', name="last_conv")(self.extractor.layers[-2].output)
            # net = Multiply()([net, net_att])
            self.extractor = Model([inputs, self.true_boxes, self.anchors_map], net)
            print "Done!"
        else:
            net = Conv2D((self.settings["CLASSES"] + 5) * self.settings["DETECTORS"], 1, padding="same", use_bias=True, name="last_conv")(net)
            # net = Multiply()([net, net_att])
            self.extractor = Model([inputs, self.true_boxes, self.anchors_map], net)
        self.outputs += [net]
        return self.extractor


    def shot_yolo_convolutional_net_A(self,version):
        bn_epsilon = 0.0
        inputs = Input(shape=(None, None, 3))
       
        net = Conv2D(32, 3, padding="same", use_bias=False, name="0_conv")(inputs)
        net = BatchNormalization(epsilon=bn_epsilon)(net)
        net = LeakyReLU(alpha=0.1)(net)
        net = MaxPooling2D(pool_size=(2,2), strides=(2,2), name="1_max")(net)

        net = Conv2D(64, 3, padding="same", use_bias=False, name="2_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon)(net)
        net = LeakyReLU(alpha=0.1)(net)
        net = MaxPooling2D(pool_size=(2,2), strides=(2,2), name="3_max")(net)

        net = Conv2D(128, 3, padding="same", use_bias=False, name="4_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon)(net)
        net = LeakyReLU(alpha=0.1)(net)

        net = Conv2D(64, 1, padding="same", use_bias=False, name="5_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon)(net)
        net = LeakyReLU(alpha=0.1)(net)
        
        net = Conv2D(128, 3, padding="same", use_bias=False, name="6_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon)(net)
        net = LeakyReLU(alpha=0.1)(net)
        net = MaxPooling2D(pool_size=(2,2), strides=(2,2), name="7_max")(net)

        net = Conv2D(256, 3, padding="same", use_bias=False, name="8_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon)(net)
        net = LeakyReLU(alpha=0.1)(net)

        net = Conv2D(128, 1, padding="same", use_bias=False, name="9_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon)(net)
        net = LeakyReLU(alpha=0.1)(net)

        net = Conv2D(256, 3, padding="same", use_bias=False, name="10_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon)(net)
        net = LeakyReLU(alpha=0.1)(net)
        net = MaxPooling2D(pool_size=(2,2), strides=(2,2), name="11_max")(net)

        net = Conv2D(512, 3, padding="same", use_bias=False, name="12_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon)(net)
        net = LeakyReLU(alpha=0.1)(net)

        net = Conv2D(256, 1, padding="same", use_bias=False, name="13_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon)(net)
        net = LeakyReLU(alpha=0.1)(net)

        net = Conv2D(512, 3, padding="same", use_bias=False, name="14_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon)(net)
        net = LeakyReLU(alpha=0.1)(net)

        net = Conv2D(256, 1, padding="same", use_bias=False, name="15_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon)(net)
        net = LeakyReLU(alpha=0.1)(net)

        net = Conv2D(512, 3, padding="same", use_bias=False, name="16_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon)(net)
        c16 = LeakyReLU(alpha=0.1)(net)
        net = MaxPooling2D(pool_size=(2,2), strides=(2,2), name="17_max")(c16)

        net = Conv2D(1024, 3, padding="same", use_bias=False, name="18_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon)(net)
        net = LeakyReLU(alpha=0.1)(net)

        net = Conv2D(512, 1, padding="same", use_bias=False, name="19_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon)(net)
        net = LeakyReLU(alpha=0.1)(net)

        net = Conv2D(1024, 3, padding="same", use_bias=False, name="20_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon)(net)
        net = LeakyReLU(alpha=0.1)(net)

        net = Conv2D(512, 1, padding="same", use_bias=False, name="21_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon)(net)
        net = LeakyReLU(alpha=0.1)(net)

        net = Conv2D(1024, 3, padding="same", use_bias=False, name="22_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon)(net)
        net = LeakyReLU(alpha=0.1)(net)


        net = Conv2D(1024, 3, padding="same", use_bias=False, name="24_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon)(net)
        net = LeakyReLU(alpha=0.1)(net)


        net = Conv2D(1024, 3, padding="same", use_bias=False, kernel_initializer='glorot_uniform', name="new_26_conv")(net)

        net = BatchNormalization(epsilon=bn_epsilon)(net)
        net = LeakyReLU(alpha=0.1)(net)        


        net = Lambda(lambda args: args[0])([net, self.true_boxes])
        net = Lambda(lambda args: args[0])([net, self.anchors_map])
        net = Conv2D(45, 1, padding="same", use_bias=True, name="last_conv")(net)
        self.extractor = Model([inputs, self.true_boxes, self.anchors_map], net)
        self.outputs += [net]
        return self.extractor

    def shot_yolo_convolutional_net_B(self,version):
        bn_epsilon = 0.0
        inputs = Input(shape=(None, None, 3))
       
        net = Conv2D(32, 3, padding="same", use_bias=False, name="0_conv")(inputs)
        net = BatchNormalization(epsilon=bn_epsilon)(net)
        net = LeakyReLU(alpha=0.1)(net)
        net = MaxPooling2D(pool_size=(2,2), strides=(2,2), name="1_max")(net)

        net = Conv2D(64, 3, padding="same", use_bias=False, name="2_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon)(net)
        net = LeakyReLU(alpha=0.1)(net)
        net = MaxPooling2D(pool_size=(2,2), strides=(2,2), name="3_max")(net)

        net = Conv2D(128, 3, padding="same", use_bias=False, name="4_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon)(net)
        net = LeakyReLU(alpha=0.1)(net)

        net = Conv2D(64, 1, padding="same", use_bias=False, name="5_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon)(net)
        net = LeakyReLU(alpha=0.1)(net)
        
        net = Conv2D(128, 3, padding="same", use_bias=False, name="6_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon)(net)
        net = LeakyReLU(alpha=0.1)(net)
        net = MaxPooling2D(pool_size=(2,2), strides=(2,2), name="7_max")(net)

        net = Conv2D(256, 3, padding="same", use_bias=False, name="8_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon)(net)
        net = LeakyReLU(alpha=0.1)(net)

        net = Conv2D(128, 1, padding="same", use_bias=False, name="9_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon)(net)
        net = LeakyReLU(alpha=0.1)(net)

        net = Conv2D(256, 3, padding="same", use_bias=False, name="10_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon)(net)
        net = LeakyReLU(alpha=0.1)(net)
        net = MaxPooling2D(pool_size=(2,2), strides=(2,2), name="11_max")(net)

        net = Conv2D(512, 3, padding="same", use_bias=False, name="12_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon)(net)
        net = LeakyReLU(alpha=0.1)(net)

        net = MaxPooling2D(pool_size=(2,2), strides=(2,2), name="17_max")(net)

        net = Conv2D(1024, 3, padding="same", use_bias=False, name="18_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon)(net)
        net = LeakyReLU(alpha=0.1)(net)

        net = Conv2D(512, 1, padding="same", use_bias=False, name="19_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon)(net)
        net = LeakyReLU(alpha=0.1)(net)

        net = Conv2D(1024, 3, padding="same", use_bias=False, name="20_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon)(net)
        net = LeakyReLU(alpha=0.1)(net)

        net = Conv2D(512, 1, padding="same", use_bias=False, name="21_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon)(net)
        net = LeakyReLU(alpha=0.1)(net)

        net = Conv2D(1024, 3, padding="same", use_bias=False, name="22_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon)(net)
        net = LeakyReLU(alpha=0.1)(net)


        net = Conv2D(1024, 3, padding="same", use_bias=False, name="24_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon)(net)
        net = LeakyReLU(alpha=0.1)(net)


        net = Conv2D(1024, 3, padding="same", use_bias=False, kernel_initializer='glorot_uniform', name="new_26_conv")(net)

        net = BatchNormalization(epsilon=bn_epsilon)(net)
        net = LeakyReLU(alpha=0.1)(net)        


        net = Lambda(lambda args: args[0])([net, self.true_boxes])
        net = Lambda(lambda args: args[0])([net, self.anchors_map])
        net = Conv2D(45, 1, padding="same", use_bias=True, name="last_conv")(net)
        self.extractor = Model([inputs, self.true_boxes, self.anchors_map], net)
        self.outputs += [net]
        return self.extractor

    def shot_yolo_convolutional_net_C(self,version):
        bn_epsilon = 0.0
        inputs = Input(shape=(None, None, 3))
       
        net = Conv2D(32, 3, padding="same", use_bias=False, name="0_conv")(inputs)
        net = BatchNormalization(epsilon=bn_epsilon)(net)
        net = LeakyReLU(alpha=0.1)(net)
        net = MaxPooling2D(pool_size=(2,2), strides=(2,2), name="1_max")(net)

        net = Conv2D(64, 3, padding="same", use_bias=False, name="2_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon)(net)
        net = LeakyReLU(alpha=0.1)(net)
        net = MaxPooling2D(pool_size=(2,2), strides=(2,2), name="3_max")(net)

        net = Conv2D(128, 3, padding="same", use_bias=False, name="4_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon)(net)
        net = LeakyReLU(alpha=0.1)(net)

        net = Conv2D(64, 1, padding="same", use_bias=False, name="5_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon)(net)
        net = LeakyReLU(alpha=0.1)(net)
        
        net = Conv2D(128, 3, padding="same", use_bias=False, name="6_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon)(net)
        net = LeakyReLU(alpha=0.1)(net)
        net = MaxPooling2D(pool_size=(2,2), strides=(2,2), name="7_max")(net)

        net = Conv2D(256, 3, padding="same", use_bias=False, name="8_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon)(net)
        net = LeakyReLU(alpha=0.1)(net)

        net = MaxPooling2D(pool_size=(2,2), strides=(2,2), name="11_max")(net)

        net = Conv2D(512, 3, padding="same", use_bias=False, name="12_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon)(net)
        net = LeakyReLU(alpha=0.1)(net)

        net = MaxPooling2D(pool_size=(2,2), strides=(2,2), name="17_max")(net)

        net = Conv2D(1024, 3, padding="same", use_bias=False, name="18_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon)(net)
        net = LeakyReLU(alpha=0.1)(net)

        net = Conv2D(512, 1, padding="same", use_bias=False, name="19_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon)(net)
        net = LeakyReLU(alpha=0.1)(net)

        net = Conv2D(1024, 3, padding="same", use_bias=False, name="22_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon)(net)
        net = LeakyReLU(alpha=0.1)(net)


        net = Conv2D(1024, 3, padding="same", use_bias=False, name="24_conv")(net)
        net = BatchNormalization(epsilon=bn_epsilon)(net)
        net = LeakyReLU(alpha=0.1)(net)


        net = Conv2D(1024, 3, padding="same", use_bias=False, kernel_initializer='glorot_uniform', name="new_26_conv")(net)

        net = BatchNormalization(epsilon=bn_epsilon)(net)
        net = LeakyReLU(alpha=0.1)(net)        


        net = Lambda(lambda args: args[0])([net, self.true_boxes])
        net = Lambda(lambda args: args[0])([net, self.anchors_map])
        net = Conv2D(45, 1, padding="same", use_bias=True, name="last_conv")(net)
        self.extractor = Model([inputs, self.true_boxes, self.anchors_map], net)
        self.outputs += [net]
        return self.extractor

    def freeze_layers_from_list(self, layers, mode=False):
        for layer_name in layers:
            layer = self.extractor.get_layer(layer_name)
            layer.trainable = mode

        print "Freezing Done!"

    def freeze_layers(self, freeze_until="last_conv", mode="freeze"):
        last_freezed_layer = ""
        if mode == "freeze":
            print "Freezing layers...",
        else:
            print "Unfreezing layers...",
        layers = self.extractor.layers

        for layer in layers:
            if type(layer).__name__ == "InputLayer":
                continue
            
            layer_type = type(layer).__name__
            if freeze_until == "last_conv" and layer.name == "last_conv":
                break
            if mode == "freeze":
                layer.trainable = False
            else:
                layer.trainable = True

            last_freezed_layer = layer.name
            if freeze_until == layer.name:
                break
        print "Done! Last affected layer: {}".format(last_freezed_layer)


    def golo_loss(self, y_true, y_pred):        
        output_shape = tf.shape(y_true)
        output_shape = tf.stop_gradient(output_shape)
        default_shape = output_shape[0:-1]
        default_shape = tf.stop_gradient(default_shape)
        
        pred_shape = tf.concat([default_shape, tf.Variable((5,5+self.settings["CLASSES"]))], 0)
        delta_shape = tf.concat([default_shape, tf.Variable((5,2))], 0)
        pred_shape = tf.stop_gradient(pred_shape)

        true_shape = tf.concat([default_shape, tf.Variable((5,5+self.settings["CLASSES"]))], 0)
        true_shape = tf.stop_gradient(true_shape)

        GRID_DIM_INT = output_shape[1]
        GRID_DIM_INT = tf.stop_gradient(GRID_DIM_INT)
        seen = tf.Variable(0.)        
        # y_true = tf.Print(y_true, [GRID_DIM_INT], "GRID_DIM_INT: ", summarize=30000)

        # Reshape net output dims according to detectors
        y_true = tf.reshape(y_true, true_shape)
        y_pred = tf.reshape(y_pred, pred_shape)    

        cell_x = tf.to_float(tf.reshape(tf.tile(tf.range(GRID_DIM_INT), [GRID_DIM_INT]), (1, GRID_DIM_INT, GRID_DIM_INT, 1, 1)))
        cell_y = tf.transpose(cell_x, (0,2,1,3,4))

        cell_grid = tf.tile(tf.concat([cell_x,cell_y], -1), [self.settings["BATCH_SIZE"], 1, 1, 5, 1])
        cell_grid = tf.stop_gradient(cell_grid)

        seen = tf.Variable(0.)
                        
        y_pred_xy = y_pred[..., :2]
        pred_xy = tf.sigmoid(y_pred[..., :2])
        pred_wh = y_pred[..., 2:4]
        pred_box_xy = (pred_xy + cell_grid) / tf.to_float(GRID_DIM_INT)
        pred_box_wh = tf.truediv((tf.exp(y_pred[..., 2:4]) * np.reshape(self.settings["ANCHORS"], [1,1,1,self.settings["DETECTORS"],2])),tf.to_float(GRID_DIM_INT))
        
        pred_box_conf = tf.sigmoid(y_pred[..., 4])
        pred_box_class = tf.nn.softmax(y_pred[..., 5:])

        true_box_xy = y_true[..., 0:2]
        true_xy = y_true[..., 0:2]
        true_box_wh = y_true[..., 2:4]
        true_box_class = y_true[...,5:]

        anchor_wh = np.reshape(self.settings["ANCHORS"], [1,1,self.settings["DETECTORS"],2]) / tf.to_float(GRID_DIM_INT)

        # WARMUP SETUP
        delta_xy = tf.zeros(delta_shape, dtype=tf.float32)
        delta_wh = tf.zeros(delta_shape, dtype=tf.float32)

        warmup_xy = (0.5 + cell_grid) / tf.to_float(GRID_DIM_INT)
        warmup_wh = tf.to_float(tf.ones(delta_shape) * np.reshape(self.settings["ANCHORS"], [1,1,1,self.settings["DETECTORS"],2]) / tf.to_float(GRID_DIM_INT))

        delta_xy = ((warmup_xy - pred_xy) * 0.01)
        delta_wh = ((tf.stop_gradient(tf.log((warmup_wh + self.settings["EPSILON"])/(anchor_wh + self.settings["EPSILON"]))) - y_pred[...,2:4]) * 0.01)

        seen = tf.assign_add(seen, 1.)
        delta_xy, delta_wh = tf.cond(tf.less(seen, self.settings["WARMUP"]), 
                              lambda: [delta_xy, 
                                       delta_wh],
                              lambda: [tf.zeros(delta_shape, dtype=tf.float32), 
                                       tf.zeros(delta_shape, dtype=tf.float32)])

        y_true = tf.cond(tf.equal(seen, self.settings["WARMUP"]), 
                              lambda: [tf.Print(y_true, [seen], "WARMUP END! ", summarize=3000)],
                              lambda: [y_true])
        
        is_object = y_true[..., 4]
        no_object = tf.square(is_object - 1.0)

        #-----------------------------------
        #IOU PREDs vs ALL GT
        truth_xy = self.true_boxes[..., 0:2]
        truth_wh = self.true_boxes[..., 2:4]

        truth_xy = tf.stop_gradient(truth_xy)
        truth_wh = tf.stop_gradient(truth_wh)

        truth_wh_half = truth_wh / 2.
        truth_mins    = truth_xy - truth_wh_half
        truth_maxes   = truth_xy + truth_wh_half
        
        pred_box_xy_vs_gt = tf.expand_dims(pred_box_xy, 4)
        pred_box_wh_vs_gt = tf.expand_dims(pred_box_wh, 4)
        
        pred_wh_half = pred_box_wh_vs_gt / 2.
        pred_mins    = pred_box_xy_vs_gt - pred_wh_half
        pred_maxes   = pred_box_xy_vs_gt + pred_wh_half    
        
        # is_object = tf.Print(is_object, [tf.shape(pred_mins),tf.shape(truth_mins)], "----: ", summarize=30000)

        intersect_mins  = tf.maximum(pred_mins,  truth_mins)
        intersect_maxes = tf.minimum(pred_maxes, truth_maxes)
        intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
        
        truth_areas = truth_wh[..., 0] * truth_wh[..., 1]
        pred_areas = pred_box_wh_vs_gt[..., 0] * pred_box_wh_vs_gt[..., 1]

        union_areas = pred_areas + truth_areas - intersect_areas# + self.settings["EPSILON"]
        iou_pred_gt_scores  = tf.truediv(intersect_areas, union_areas)
        iou_pred_gt_scores = tf.stop_gradient(iou_pred_gt_scores)

        best_ious = tf.reduce_max(iou_pred_gt_scores, axis=4)
        ious_below_threshold = tf.greater(self.settings["THRESHOLD"],best_ious)
        best_ious_map = tf.to_float(ious_below_threshold)
        best_ious_map = tf.stop_gradient(best_ious_map)

        # is_object = tf.Print(is_object, [tf.shape(pred_mins)],     "shape(pred_mins): ",     summarize=30000)
        # is_object = tf.Print(is_object, [tf.shape(best_ious_map)], "shape(best_ious_map): ", summarize=30000)

        # is_object = tf.Print(is_object, [best_ious[...,6,4,:,:]],      "best_ious[6,4]: ",      summarize=30000)        
        # is_object = tf.Print(is_object, [best_ious[...,6,6,:,:]],      "best_ious[6,6]: ",      summarize=30000)        
        # is_object = tf.Print(is_object, [best_ious[...,12,6,:,:]],     "best_ious[12,6]: ",     summarize=30000)
        # is_object = tf.Print(is_object, [best_ious_map[...,12,6,:,:]], "best_ious_map[12,6]: ", summarize=30000)

        delta_conf = self.settings["SCALE_NOOBJ"] * (0.0 - pred_box_conf) * best_ious_map
        #--------------------------------

        true_anchors_map = tf.stop_gradient(self.anchors_map)

        true_scale = (2.0 - true_box_wh[...,0] * true_box_wh[...,1]) * self.settings["SCALE_COOR"]
        true_scale = tf.expand_dims(true_scale, -1)
        true_scale = tf.stop_gradient(true_scale)

        true_grid_xy = (true_xy * tf.to_float(GRID_DIM_INT) - cell_grid) * tf.expand_dims(is_object, -1)
        true_grid_xy = tf.stop_gradient(true_grid_xy)


        delta_xy += ((true_grid_xy - pred_xy) * true_scale) * true_anchors_map
        delta_wh += ((tf.stop_gradient(tf.log((true_box_wh + self.settings["EPSILON"])/(anchor_wh + self.settings["EPSILON"]))) - y_pred[...,2:4]) * true_scale) * true_anchors_map

        delta_class = self.settings["SCALE_PROB"] * (true_box_class - pred_box_class) * true_anchors_map


        #IOU PREDs vs TRUEs
        true_wh_half = true_box_wh / 2.
        true_mins    = true_box_xy - true_wh_half
        true_maxes   = true_box_xy + true_wh_half
        
        pred_wh_half = pred_box_wh / 2.
        pred_mins    = pred_box_xy - pred_wh_half
        pred_maxes   = pred_box_xy + pred_wh_half       
        
        intersect_mins  = tf.maximum(pred_mins,  true_mins)
        intersect_maxes = tf.minimum(pred_maxes, true_maxes)
        intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
        
        true_areas = true_box_wh[..., 0] * true_box_wh[..., 1]
        pred_areas = pred_box_wh[..., 0] * pred_box_wh[..., 1]

        union_areas = pred_areas + true_areas - intersect_areas# + self.settings["EPSILON"]
        iou_cell_vs_true  = tf.div(intersect_areas, union_areas)
        iou_cell_vs_true = tf.stop_gradient(iou_cell_vs_true)


        max_iou_cell_vs_true = tf.expand_dims(tf.reduce_max(iou_cell_vs_true, 3),-1)
        best_cell_iou = tf.to_float(tf.equal(iou_cell_vs_true, max_iou_cell_vs_true)) * tf.to_float(tf.greater(iou_cell_vs_true, 0.0))

        delta_conf += (iou_cell_vs_true - pred_box_conf) * self.settings["SCALE_OBJ"] * tf.squeeze(true_anchors_map,-1)        

        delta = tf.concat([delta_xy, delta_wh, tf.expand_dims(delta_conf,-1), delta_class], -1)
        delta = tf.reduce_mean(delta, axis=0)
        loss = tf.reduce_sum(tf.square(delta), name="loss")

        return loss

    def surgery(self, model_A, model_B):
        for layer in model_A.layers:
            name = layer.name
            print "Exchanging {} weights...".format(name)
            try:
                model_B.get_layer(name).set_weights(model_A.get_layer(name).get_weights())
            except ValueError:
                print "{} layer not found!".format(name)
        return model_B

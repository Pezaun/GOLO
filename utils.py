#!/usr/bin/env python
from keras.callbacks import Callback
from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
import warnings
import cv2
import sys
import json
import tensorflow as tf


#=============#
# Keras Utils #
#=============#
class StagedReduceLROnPlateau(Callback):
    def __init__(self, monitor='val_loss', factor=0.1, patience=10,
                 verbose=0, mode='auto', epsilon=1e-4, cooldown=0, min_lr=0, stages=None, initial_lr=None, reduce_on_plateau=True):
        super(StagedReduceLROnPlateau, self).__init__()

        self.monitor = monitor
        if factor >= 1.0:
            raise ValueError('StagedReduceLROnPlateau '
                             'does not support a factor >= 1.0.')
        self.factor = factor
        self.min_lr = min_lr
        self.epsilon = epsilon
        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0  # Cooldown counter.
        self.wait = 0
        self.best = 0
        self.mode = mode
        self.monitor_op = None
        self._reset()
        self.stages = stages
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        self.reduce_on_plateau = reduce_on_plateau
        print "Observing LR Stages!"

    def _reset(self):
        """Resets wait counter and cooldown counter.
        """
        if self.mode not in ['auto', 'min', 'max']:
            warnings.warn('Learning Rate Plateau Reducing mode %s is unknown, '
                          'fallback to auto mode.' % (self.mode),
                          RuntimeWarning)
            self.mode = 'auto'
        if (self.mode == 'min' or
           (self.mode == 'auto' and 'acc' not in self.monitor)):
            self.monitor_op = lambda a, b: np.less(a, b - self.epsilon)
            self.best = np.Inf
        else:
            self.monitor_op = lambda a, b: np.greater(a, b + self.epsilon)
            self.best = -np.Inf
        self.cooldown_counter = 0
        self.wait = 0

    def on_train_begin(self, logs=None):
        self._reset()
        print "Setting initial LR to {}".format(self.initial_lr)
        K.set_value(self.model.optimizer.lr, self.initial_lr)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn(
                'Reduce LR on plateau conditioned on metric `%s` '
                'which is not available. Available metrics are: %s' %
                (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning
            )

        else:

            last_stage = max(self.stages.keys())
            if last_stage >= epoch + 1:
                print "\nApply stage LR rules..."
                try:
                    new_lr = self.stages[epoch+1]
                    self.current_lr = new_lr
                    K.set_value(self.model.optimizer.lr, new_lr)
                    print "Stage {} reached. LR={}".format(epoch+1,new_lr)
                except KeyError:
                    print "Keep LR={}".format(self.current_lr)
                return

            if not self.reduce_on_plateau:
                print "\nReduce LR on plateau OFF. Keep last LR..."
                return

            print "\nReduce LR on plateau..."
            if self.in_cooldown():
                self.cooldown_counter -= 1
                self.wait = 0

            if self.monitor_op(current, self.best):
                self.best = current
                self.wait = 0
            elif not self.in_cooldown():
                if self.wait >= self.patience:
                    old_lr = float(K.get_value(self.model.optimizer.lr))
                    if old_lr > self.min_lr:
                        new_lr = old_lr * self.factor
                        new_lr = max(new_lr, self.min_lr)
                        K.set_value(self.model.optimizer.lr, new_lr)
                        if self.verbose > 0:
                            print('\nEpoch %05d: StagedReduceLROnPlateau reducing learning '
                                  'rate to %s.' % (epoch + 1, new_lr))
                        self.cooldown_counter = self.cooldown
                        self.wait = 0
                self.wait += 1

    def in_cooldown(self):
        return self.cooldown_counter > 0



class ReorgLayer(Layer):

    def __init__(self, **kwargs):
        # self.output_dim = output_dim
        super(ReorgLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ReorgLayer, self).build(input_shape)

    def call(self, input_tensor):
        stride = 2
        darknet = True
        shapes = tf.shape(input_tensor)
        channel_first = tf.transpose(input_tensor,(0,3,1,2))
        reshape_tensor = tf.reshape(channel_first, (-1,shapes[3] // (stride ** 2), shapes[1], stride, shapes[2], stride))
        permute_tensor = tf.transpose(reshape_tensor,(0,3,5,1,2,4))
        target_tensor = tf.reshape(permute_tensor, (-1, shapes[3]*stride**2,shapes[1] // stride, shapes[2] // stride))
        channel_last = tf.transpose(target_tensor,(0,2,3,1))
        result = tf.reshape(channel_last, (-1,shapes[1]//stride, shapes[2]//stride, tf.cast(input_tensor.shape[3]*4, tf.int32)))
        return result

    def compute_output_shape(self, input_shape):
        if input_shape[1] == None:
            return (input_shape[0], input_shape[1], input_shape[2], input_shape[3]*4)
        return (input_shape[0], input_shape[1]/2, input_shape[2]/2, input_shape[3]*4)

    


#=============#
# Image Utils #
#=============#
def mergeset_images(src_image, x, y, dest_image):
    if x < 0:
        src_image = src_image[:,abs(x):]
        x = 0

    if y < 0:
        src_image = src_image[abs(y):,:]
        y = 0

    nh,nw = dest_image.shape[:2]
    oh,ow = src_image.shape[:2]

    aw = nw - x
    ah = nh - y

    if ow > aw:
        src_image = src_image[:,:aw]
        ow = aw

    if oh > ah:
        src_image = src_image[:ah,:]
        oh = ah

    dest_image[y:y+oh,x:x+ow] = src_image
    return dest_image

def scale_image(im_data, w, h):
    im_h, im_w, z = im_data.shape
    wp = w/float(im_w)
    hp = h/float(im_h)
    if wp < hp:
        nw = w
        nh = (im_h * nw) /im_w
    else:
        nh = h
        nw = (im_w * nh) /im_h

    return (nw,nh)

def letter(im_data, w, h, color=127):
    n_shape = scale_image(im_data, w,h)
    im_dest = (np.ones((h,w,3))*color).astype(np.uint8)
    im_data = cv2.resize(im_data, (n_shape[0], n_shape[1]), interpolation=cv2.INTER_NEAREST)
    x = (im_dest.shape[1] - im_data.shape[1]) / 2
    y = (im_dest.shape[0] - im_data.shape[0]) / 2
    im_dest = mergeset_images(im_data, x, y, im_dest)
    return im_dest

def unletter_boxes(boxes, out_img, input_w, input_h):
    dim_change = scale_image(out_img, input_w, input_h)
    w_change = input_w - dim_change[0]
    h_change = input_h - dim_change[1]


    print dim_change
    for box in boxes:
        x = (box.x * input_w) - w_change / 2.0
        y = (box.y * input_h) - h_change / 2.0
        w = box.w * input_w
        h = box.h * input_h

        box.x = x/(input_w - w_change)
        box.w = w/(input_w - w_change)
        box.y = y/(input_h - h_change)
        box.h = h/(input_h - h_change)


#================#
# Settings Utils #
#================#
def parse_settings(path):
    try:
        data = json.load(open(path, "r"))
    except Exception, e:
        print "Invalid config file with error: " + str(e)
        sys.exit(0)

    data["settings"]["ANCHORS"]     = [float(ANCHORS.strip()) for ANCHORS in data["settings"]["ANCHORS"].split(',')]
    data["settings"]["CLASSES"]     = len(data["settings"]["CLASSES_LABEL"])
    data["settings"]["LR_STAGES"]   = {int(t.split(":")[0]):float(t.split(":")[1]) for t in data["settings"]["LR_STAGES"].split(",")}
    data["settings"]["TRAIN_MULT"]  = 1 if data["settings"]["TRAIN_MULT"] < 1 else data["settings"]["TRAIN_MULT"]

    return data["settings"]
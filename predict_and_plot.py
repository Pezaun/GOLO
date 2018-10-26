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

augmentate_off = iaa.Sequential(
            [
                #iaa.Sharpen(alpha=0.25),
                iaa.ContrastNormalization(1.1),
                iaa.Multiply(0.9)
            ],
            random_order=False
        )

settings = {}

class BoundBox:
    def __init__(self, x, y, w, h, c = None, classes = None):
        self.x     = x
        self.y     = y
        self.w     = w
        self.h     = h
        
        self.c     = c
        self.classes = classes

        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)
        
        return self.label
    
    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]
            
        return self.score
    def __str__(self):
        return "X:{} Y:{} W:{} H:{} C:{}".format(self.x, self.y, self.w, self.h, self.get_label())

def nms(boxes):
    for c in range(settings["CLASSES"]):
        sorted_indices = list(reversed(np.argsort([box.scores[c] for box in boxes])))

        for i in xrange(len(sorted_indices)):
            index_i = sorted_indices[i]
            
            if boxes[index_i].scores[c] == 0:                 
                continue
            else:
                for j in xrange(i+1, len(sorted_indices)):
                    index_j = sorted_indices[j]
                    if bbox_iou(boxes[index_i], boxes[index_j]) >= settings["NMS_THRESHOLD"]:
                        boxes[index_j].scores[c] = 0

    result = [box for box in boxes if box.get_score() > settings["PRED_CONF_THRES"]]
    return result

def bbox_iou(box1, box2):
    x1_min  = box1.x - box1.w/2
    x1_max  = box1.x + box1.w/2
    y1_min  = box1.y - box1.h/2
    y1_max  = box1.y + box1.h/2
    
    x2_min  = box2.x - box2.w/2
    x2_max  = box2.x + box2.w/2
    y2_min  = box2.y - box2.h/2
    y2_max  = box2.y + box2.h/2
    
    intersect_w = interval_overlap([x1_min, x1_max], [x2_min, x2_max])
    intersect_h = interval_overlap([y1_min, y1_max], [y2_min, y2_max])    
    intersect = intersect_w * intersect_h    
    union = box1.w * box1.h + box2.w * box2.h - intersect
    
    return float(intersect) / union
    
def interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2,x4) - x1
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2,x4) - x3  

def sigmoid(x):
    return 1. / (1.  + np.exp(-x))

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference

# def softmax(x, axis=-1, t=-100.):
#         x = x - np.max(x)
        
#         if np.min(x) < t:
#             x = x/np.min(x)*t
            
#         e_x = np.exp(x)
        
#         return e_x / e_x.sum(axis, keepdims=True)


if __name__ == "__main__":
    settings = parse_settings(sys.argv[1])
    box_colors = [(int(random.random()*255),int(random.random()*255),int(random.random()*255)) for i in range(settings["CLASSES"])]
    settings["ANCHORS"] = np.reshape(settings["ANCHORS"], [settings["DETECTORS"],2])
    print "Predict..."
    fe = feature_extractor.FeatureExtractor(settings)

    nets = {"yolo_v1": (fe.yolo_convolutional_net, 0),
            "shot_yolo_A": (fe.shot_yolo_convolutional_net_A, 0),
            "shot_yolo_B": (fe.shot_yolo_convolutional_net_B, 0),
            "shot_yolo_C": (fe.shot_yolo_convolutional_net_C, 0),
            "yolo_tiny": (fe.tiny_yolo_convolutional_net, 0),
            "inceptionv3": (fe.inceptionv3_convolutional_net, 2),
            "mobilenetv2": (fe.mobilenetv2_convolutional_net, 0),
            "xception": (fe.xception_convolutional_net, 0),
            "golonet": (fe.golonet, 0),
            "golonetB": (fe.golonetB, 0)}

    net            = nets[settings["NET_ARCH"]][0](1)
    out_dim_factor = nets[settings["NET_ARCH"]][1]


    net = nets[settings["NET_ARCH"]][0](1)
    out_dim_factor = nets[settings["NET_ARCH"]][1]

    # net = fe.extractor
    net.summary()

    # m = Model([net.inputs[0]], net.outputs)

    # m.summary()
    # m.save("darknet19.h5")

    # sys.exit(0)
    net.load_weights(settings["PRED_MODEL"])

    if len(sys.argv) > 2:
        input_data = sys.argv[2]
    else:
        input_data = ""


    with open(settings["VALID_IMAGES_INDEX"]) as f:
        instances_list = f.read().splitlines()
    
    # if not os.path.exists(os.path.join(settings["PRED_OUTPUT_PATH"], "preds")):
    #     os.makedirs(os.path.join(settings["PRED_OUTPUT_PATH"], "preds"))

    print input_data
    image_name = None
    ts = 0.0
    im_count = 0
    for image_name in instances_list:
        im_count += 1
        print image_name
        im_data = cv2.imread(os.path.join(settings["IMAGES_PATH"],image_name))
        image_name = image_name.split("/")[-1]
        
        print "Input Shape: ", im_data.shape

        im_h, im_w = im_data.shape[:2]
        im_out = im_data.copy()

        if settings["LETTER_INPUT_IMAGE"]:
            im_data = utils.letter(im_data, settings["PRED_INPUT_SIZE"], settings["PRED_INPUT_SIZE"])
        else:
            im_data = cv2.resize(im_data, (settings["PRED_INPUT_SIZE"], settings["PRED_INPUT_SIZE"]), interpolation=cv2.INTER_NEAREST)    
                
        im_data = augmentate_off.augment_image(im_data)
        im_data = im_data[:,:,::-1]

        im_data = im_data.astype(np.float32).reshape((1,settings["PRED_INPUT_SIZE"], settings["PRED_INPUT_SIZE"],3))
        
        if settings["NORM"] == "B":
            im_data /= 255.0
            im_data -= 0.5
            im_data *= 2.0
        else:
            im_data /= 255.0
        
        fake_boxes = np.zeros((1, 1, 1, 1, settings["MAX_BOXES"], 4))
        fake_anchors = np.zeros((1, 13 - out_dim_factor, 13 - out_dim_factor, 5, 1))  
        
        t0 = time.time()
        preds = net.predict([im_data, fake_boxes, fake_anchors], batch_size=1, verbose=0).reshape((1,settings["PRED_INPUT_SIZE"]/32 - out_dim_factor,settings["PRED_INPUT_SIZE"]/32 - out_dim_factor,settings["DETECTORS"],5+settings["CLASSES"]))
        tf = time.time() - t0
        ts += tf
        preds = preds.reshape((settings["PRED_INPUT_SIZE"]/32 - out_dim_factor, settings["PRED_INPUT_SIZE"]/32 - out_dim_factor,settings["DETECTORS"],settings["CLASSES"]+5))


        dim, detectors, n_classes = (settings["PRED_INPUT_SIZE"]/32 - out_dim_factor, settings["DETECTORS"], settings["CLASSES"])
        boxes = []
        for row in range(dim):
            for col in range(dim):
                for n in range(detectors):
                    x,y = (sigmoid(preds[row,col,n,:2])+[col,row])/dim
                    w,h = (np.exp(preds[row,col,n,2:4]) * settings["ANCHORS"][n])/dim
                    scale = sigmoid(preds[row,col,n,4])
                    classes_scores = softmax(preds[row,col,n,5:])*scale

                    if np.sum(classes_scores * (classes_scores>settings["PRED_CONF_THRES"])) > 0:
                    # if scale > settings["PRED_CONF_THRES"]:
                        box = BoundBox(x, y, w, h, scale, classes_scores)
                        boxes += [box]

        for c in range(settings["CLASSES"]):
            sorted_indices = list(reversed(np.argsort([box.classes[c] for box in boxes])))
            for i in xrange(len(sorted_indices)):
                index_i = sorted_indices[i]
                
                if boxes[index_i].classes[c] == 0: 
                    continue
                else:
                    for j in xrange(i+1, len(sorted_indices)):
                        index_j = sorted_indices[j]
                        
                        if bbox_iou(boxes[index_i], boxes[index_j]) >= settings["NMS_THRESHOLD"]:
                            boxes[index_j].classes[c] = 0
                            
        # remove the boxes which are less likely than a obj_threshold
        boxes = [box for box in boxes if box.get_score() > settings["PRED_CONF_THRES"]]

        
        if settings["LETTER_INPUT_IMAGE"]:
            print "Unlettering boxes..."
            utils.unletter_boxes(boxes, im_out, settings["PRED_INPUT_SIZE"], settings["PRED_INPUT_SIZE"])

        print "Plotting..."
        im_h, im_w = im_out.shape[:2]

        if settings["PRED_OUTPUT_PATH_PREDS"]:
            file_name = image_name.split(".")[0]+".gif.txt" if image_name.startswith("tumblr") else image_name.split(".")[0]+".txt"
            boxes_file = open(os.path.join(settings["PRED_OUTPUT_PATH_PREDS"], file_name), "w")

        for box in boxes:
            print box.x, box.w                
            
            xl_f = (box.x - box.w / 2.0) * im_w
            yt_f = (box.y - box.h / 2.0) * im_h
            xr_f = (box.x + box.w / 2.0) * im_w
            yb_f = (box.y + box.h / 2.0) * im_h

            xl_f = 1.0 if xl_f < 1.0 else xl_f
            yt_f = 1.0 if yt_f < 1.0 else yt_f
            xr_f = im_w if xr_f > im_w else xr_f
            yb_f = im_h if yb_f > im_h else yb_f

            xl = int(xl_f)
            yt = int(yt_f)
            xr = int(xr_f)
            yb = int(yb_f)

            box_text = "{} {} {} {} {} {}\n".format(settings["CLASSES_LABEL"][box.get_label()].replace(" ","_"), box.get_score(), xl, yt, xr, yb)
            if settings["PRED_OUTPUT_PATH_PREDS"]:
                boxes_file.write(box_text)
            
            
            # font                   = cv2.FONT_HERSHEY_COMPLEX_SMALL
            bottomLeftCornerOfText = (xl,yt-8)
            # fontScale              = 0.8
            # fontColor              = box_colors[box.get_label()]
            # lineType               = 2

            # cv2.putText(im_out, str(box.get_score())[:4] + " " + settings["CLASSES_LABEL"][box.get_label()], 
            #     bottomLeftCornerOfText, 
            #     font, 
            #     fontScale,
            #     fontColor,
            #     lineType)

            cv2.putText(im_out, str(box.get_score())[:4] + " " + settings["CLASSES_LABEL"][box.get_label()], bottomLeftCornerOfText, cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 6)
            cv2.putText(im_out, str(box.get_score())[:4] + " " + settings["CLASSES_LABEL"][box.get_label()], bottomLeftCornerOfText, cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)

            cv2.rectangle(im_out, (xl, yt), (xr, yb), box_colors[box.get_label()], 3)
            cv2.rectangle(im_out, ((xl+xr)/2-1, (yt+yb)/2-1), ((xl+xr)/2+1, (yt+yb)/2+1), (0,0,255), 3)
            print settings["CLASSES_LABEL"][box.get_label()], box.get_score()

        if settings["PRED_OUTPUT_PATH_PREDS"]:
            boxes_file.close()
        cv2.imwrite(os.path.join(settings["PRED_OUTPUT_PATH"], image_name), im_out)

    print "Total Time: {}".format(ts)
    print "Image Time: {}".format(ts/float(im_count))

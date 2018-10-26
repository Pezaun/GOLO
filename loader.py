#!/usr/bin/env python
import cv2
import numpy as np
import random
import utils as im_utils
import os
import xml.etree.ElementTree as ET
import imgaug as ia
import pdb
import math
import sys
from collections import defaultdict
from imgaug import augmenters as iaa

class AugmentateImage:
    def __init__(self, settings, show_image=False, jitter=0, input_size=(416,416), instances_path="", labels_path="", augmentate=True):
        self.settings = settings
        self.show_image = show_image
        self.jitter = jitter
        self.input_size = input_size
        self.instances_path = instances_path
        self.labels_path = labels_path
        self.classes = self.settings["CLASSES_LABEL"]
        self.augmentate_flag = augmentate
        self.augmentate = iaa.Sequential(
            [
                iaa.Add((-20, 20)),
                iaa.ContrastNormalization((0.8, 1.6)),
                iaa.AddToHueAndSaturation((-21, 21)),
            ],
            random_order=False
        )

        self.augmentate_off = iaa.Sequential(
            [
                #iaa.Sharpen(alpha=(0.4, 0.4)),
                #iaa.ContrastNormalization((0.8, 0.8), per_channel=1.0),
                # iaa.Add((-50, 50)),
                #iaa.Multiply((0.8, 0.8))
            ],
            random_order=False
        )

    def set_input_size(self, input_size):
        self.input_size = input_size

    def augment(self, im_data, boxes):        
        im_h, im_w = im_data.shape[:2]
        dw = self.jitter * im_w
        dh = self.jitter * im_h
        
        
        new_ar = (im_w + rand_uniform(-dw, dw)) / (im_h + rand_uniform(-dh, dh))
        scale = rand_uniform(.25, 2.)
        flip = np.random.binomial(1, .5)

        if not self.augmentate_flag:
            max_dim = max(im_data.shape[:2])
            scale = self.input_size[0]/float(max_dim)
            new_ar = 1
            flip = 0

        if new_ar < 1:
            nh = scale * self.input_size[0]
            nw = nh * new_ar
        else:
            nw = scale * self.input_size[1]
            nh = nw / new_ar

        dx = int(rand_uniform(0, self.input_size[1] - nw))
        dy = int(rand_uniform(0, self.input_size[0] - nh))

        if not self.augmentate_flag:
            nw = scale * im_w
            nh = scale * im_h
            dx = int((self.input_size[0] - nw) / 2)
            dy = int((self.input_size[0] - nh) / 2)

        out_frame = (np.ones((self.input_size[1], self.input_size[0],3)) * 127).astype(np.uint8)

        if flip > 0.5:
            im_data = cv2.flip(im_data, 1)
            for box in boxes:
                tmp = box[1]
                box[1] = int(box[5] - box[2])
                box[2] = int(box[5] - tmp)

        boxes = self.adjust_boxes(boxes, dx, dy, nw, nh, self.input_size[1], self.input_size[0])
        im_data_new = cv2.resize(im_data, (int(nw), int(nh)), interpolation=cv2.INTER_NEAREST)
        im_data_new = im_utils.mergeset_images(im_data_new, dx, dy, out_frame)
        if self.augmentate_flag:            
            im_data_new = self.augmentate.augment_image(im_data_new)
        else:
            im_data_new = self.augmentate_off.augment_image(im_data_new)

        

        return im_data_new, boxes

    def load_labels(self, instance_name):
        labels = []
        instance_path = os.path.join(self.labels_path, instance_name.replace(".jpg",".xml").replace(".png",".xml"))
        xml_tree = ET.parse(instance_path)
        xml_root = xml_tree.getroot()
        im_size = xml_root.find("size")
        im_w = float(im_size.find("width").text)
        im_h = float(im_size.find("height").text)

        for objs in xml_root.iter("object"):
            cls = objs.find("name").text
            difficult = int(objs.find("difficult").text)
            try:
                occluded = int(objs.find("occluded").text)
            except:
                occluded = 0
            
            if cls not in self.classes:# or difficult > 0 or occluded != 0:
                continue

            if len(labels) == self.settings["MAX_BOXES"]:
                continue

            im_bbox = objs.find("bndbox")
            box_xmin = int(math.floor(float(im_bbox.find("xmin").text)))
            box_xmax = int(math.floor(float(im_bbox.find("xmax").text)))
            box_ymin = int(math.floor(float(im_bbox.find("ymin").text)))
            box_ymax = int(math.floor(float(im_bbox.find("ymax").text)))

            labels += [[self.classes.index(cls), box_xmin, box_xmax, box_ymin, box_ymax, im_w, im_h]]
        return labels

    def adjust_boxes(self, boxes, x, y, w, h, in_w, in_h):
        adjusted_boxes = []
        for box in boxes:
            dw = w / float(box[5])
            dh = h / float(box[6])
            box[1] = int(box[1] * dw) + x
            box[2] = int(box[2] * dw) + x
            box[3] = int(box[3] * dh) + y
            box[4] = int(box[4] * dh) + y

            if box[1] < 0.0:
                box[1] = 0

            if box[2] < 0.0:
                box[2] = 0

            if box[3] < 0.0:
                box[3] = 0

            if box[4] < 0.0:
                box[4] = 0

            if box[1] > in_w:
                box[1] = in_w

            if box[3] > in_h:
                box[3] = in_h
                        

            if box[2] > in_w:
                box[2] = in_w

            if box[4] > in_h:
                box[4] = in_h

            area = (box[2] - box[1]) * (box[4] - box[3])

            if area > 0.0:
                adjusted_boxes += [box]
            
        return adjusted_boxes

    def convert_boxes_format(self, boxes):
        converted_boxes = []
        for box in boxes:
            converted_box = [0,0,0,0,0]
            converted_box[0] = box[0]
            converted_box[1] = (box[1] + box[2]) / 2.0
            converted_box[2] = (box[4] + box[3]) / 2.0
            converted_box[3] = (box[2] - box[1])
            converted_box[4] = (box[4] - box[3])
            converted_boxes += [converted_box]
        return converted_boxes

    def normalize_boxes_data(self, boxes, im_data):
        nw = im_data.shape[1]
        nh = im_data.shape[0]
        for box in boxes:
            box[1] /= float(nw)
            box[2] /= float(nw)
            box[3] /= float(nh)
            box[4] /= float(nh)

    def draw_boxes(self, image, boxes):
        for box in boxes:
            cv2.rectangle(image, (box[1],box[3]), (box[2],box[4]), (0,255,108), 2)            

    def transform(self, im, s):
        out = np.zeros(im.shape).astype(im.dtype)
        step = s[1] * 3
        h,w,c = s
        for i in range(h):
            for k in range(c):
                for j in range(w):
                    out[k*w*h + i*w + j] = im[i*step + j*c + k]
        return out    
        
    def load_instance(self, instance_name):
        boxes = self.load_labels(instance_name)
        # print instance_name
        inst_data = cv2.imread(os.path.join(self.instances_path, instance_name))
        initial_size = inst_data.shape
        inst_data, boxes = self.augment(inst_data, boxes)
        if self.show_image:
            self.draw_boxes(inst_data, boxes)
        # print boxes
        self.normalize_boxes_data(boxes, inst_data)
        converted_boxes = self.convert_boxes_format(boxes)
        if self.show_image:
            cv2.imshow('image', inst_data)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return initial_size, inst_data, converted_boxes



class GSLoader:
    def __init__(self, train_data, instances_path, labels_path, settings, DIM=(416,416,3), show_image=False, augmentate=True, max_boxes=30, out_dim_factor=0):
        self.settings = settings
        self.classes = self.settings["CLASSES_LABEL"]
        self.dims = range(self.settings["INPUT_RANGE"][0],self.settings["INPUT_RANGE"][1]+1,32)

        # self.dim = DIM
        dim_index = len(self.dims)/2
        self.dim = (self.dims[dim_index], self.dims[dim_index],3)
        
        self.out_grid_dim = self.dim[0] / 32 - out_dim_factor
        self.labels_path = labels_path
        self.out_dim_factor = out_dim_factor
        
        # self.dims = range(320,609,32)
        self.ai = AugmentateImage(self.settings, show_image=show_image, jitter=self.settings["JITTER"], instances_path=instances_path,labels_path=labels_path, augmentate=augmentate)
        self.show_image = show_image
        self.max_boxes = max_boxes
        self.anchors = [BoundBox(0, 0, self.settings["ANCHORS"][2*i], self.settings["ANCHORS"][2*i+1]) for i in range(len(self.settings["ANCHORS"])/2)]
        with open(train_data, "r") as f:
            self.train_list = f.read().splitlines()

    def data_generator(self, batch, shuffle=False, variable_dim=False):
        batch_index = 0
        batchs_count = 0
        instances_index = 0
        dim = self.dim[0]
        self.ai.set_input_size((dim, dim))    

        X = np.zeros((batch, dim, dim, self.dim[2])).astype(np.float32)
        T = np.zeros((batch, 1, 1, 1, self.max_boxes, 4))
        A = np.zeros((batch, self.out_grid_dim, self.out_grid_dim, self.settings["DETECTORS"], 1))
        Y = np.zeros((batch, self.out_grid_dim, self.out_grid_dim, self.settings["DETECTORS"], 5 + self.settings["CLASSES"])).astype(np.float32)
        Y_count = np.zeros((batch, self.out_grid_dim, self.out_grid_dim)).astype(np.int)
        images_sizes = []
        if shuffle:
            random.shuffle(self.train_list)

        while True:
            instance_path = self.train_list[instances_index]
            grid_dim = self.out_grid_dim

            instance_name = instance_path.split("/")[-1]         
            # print instance_name   
            initial_size, im_data, boxes = self.ai.load_instance(instance_name)

            im_w = float(initial_size[1])
            im_h = float(initial_size[0])

            preview = im_data.copy()
            
            im_data = im_data.astype(np.float32)
            has_boxes = False
            # print "Received boxes:", len(boxes)
            for i, box in enumerate(boxes):
                class_index = int(box[0])

                x = box[1]
                y = box[2]
                w = box[3]
                h = box[4]                

                x_ = box[1]# * float(grid_dim)
                y_ = box[2]# * float(grid_dim)
                w_ = box[3]# * float(grid_dim)
                h_ = box[4]# * float(grid_dim)
                
                center_x = int(math.floor(x * grid_dim))
                center_y = int(math.floor(y * grid_dim))
                grid_w = int(math.floor(w * grid_dim))
                grid_h = int(math.floor(h * grid_dim))

                box_pos = [x_, y_, w_, h_,1.0]


                if i == self.max_boxes:
                    continue

                tbox = BoundBox(x, y, w * grid_dim, h * grid_dim, c = class_index, classes = -1)
                tbox.x = 0
                tbox.y = 0

                best_iou = 0.0
                best_iou_index = 0
                for j in range(len(self.anchors)):
                    iou = self.bbox_iou(tbox, self.anchors[j])
                    if iou > best_iou:
                        best_iou = iou
                        best_iou_index = j

                try:
                    T[batch_index][0][0][0][i] = box_pos[:4]                    
                except IndexError:
                    print "E:", instance_path, i, len(boxes)
                    sys.exit(0)
                
                # print "BEST IOU:", best_iou_index, best_iou
                
                box_class = np.zeros(self.settings["CLASSES"]).astype(np.float32)
                box_class[class_index] = 1.0

                has_boxes = True
                detector_index = best_iou_index

                Y[batch_index, center_y, center_x, best_iou_index,:5] = box_pos
                Y[batch_index, center_y, center_x, best_iou_index,5:] = box_class
                A[batch_index, center_y, center_x, best_iou_index,0] = 1.0

                prev_center_x = x * dim
                prev_center_y = y * dim
                prev_box_w = w * dim
                prev_box_h = h * dim
                prev_xmin = int(prev_center_x - (prev_box_w / 2.0))
                prev_xmax = int(prev_center_x + (prev_box_w / 2.0))
                prev_ymin = int(prev_center_y - (prev_box_h / 2.0))
                prev_ymax = int(prev_center_y + (prev_box_h / 2.0))
                cv2.rectangle(preview, (prev_xmin,prev_ymin), (prev_xmax,prev_ymax), (0,255,255), 2)

            instances_index += 1
            instances_index = instances_index % len(self.train_list)

            if not has_boxes:
                # print "No boxes detected!"
                continue

            if self.show_image:
                cv2.imshow('image',preview)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            im_data = im_data[...,::-1]

            if self.settings["NORM"] == "B":
                im_data /= 255.0
                im_data -= 0.5
                im_data *= 2.0
            else:
                im_data /= 255.0

            X[batch_index] = im_data
            images_sizes += [initial_size]

            batch_index += 1
            batch_index = batch_index % batch

            if instances_index == 0:
                if shuffle:
                    random.shuffle(self.train_list)

            if batch_index == 0:
                # print "batch_index:", batch_index
                Y = Y.reshape((batch, self.out_grid_dim, self.out_grid_dim, (5 + self.settings["CLASSES"]) * self.settings["DETECTORS"]))
                yield [X,T,A], Y
                batchs_count += 1
                batchs_count = batchs_count % 10

                # print "Batchs Count:", batchs_count
                if variable_dim and batchs_count == 0:
                    random.shuffle(self.dims)
                    dim = self.dims[0]
                    self.out_grid_dim = dim / 32 - self.out_dim_factor
                    self.ai.set_input_size((dim, dim))

                X = np.zeros((batch, dim, dim, self.dim[2])).astype(np.float32)
                T = np.zeros((batch, 1, 1, 1, self.max_boxes, 4))
                A = np.zeros((batch, self.out_grid_dim, self.out_grid_dim, self.settings["DETECTORS"], 1))
                Y = np.zeros((batch, self.out_grid_dim, self.out_grid_dim, self.settings["DETECTORS"], 5 + self.settings["CLASSES"])).astype(np.float32)
                Y_count = np.zeros((batch, self.out_grid_dim, self.out_grid_dim)).astype(np.int)
                images_sizes = []


    def bbox_iou(self, box1, box2):
        x1_min  = box1.x - box1.w/2
        x1_max  = box1.x + box1.w/2
        y1_min  = box1.y - box1.h/2
        y1_max  = box1.y + box1.h/2
        
        x2_min  = box2.x - box2.w/2
        x2_max  = box2.x + box2.w/2
        y2_min  = box2.y - box2.h/2
        y2_max  = box2.y + box2.h/2
        
        intersect_w = self.interval_overlap([x1_min, x1_max], [x2_min, x2_max])
        intersect_h = self.interval_overlap([y1_min, y1_max], [y2_min, y2_max])
        
        intersect = intersect_w * intersect_h
        
        union = box1.w * box1.h + box2.w * box2.h - intersect
        
        return float(intersect) / union

    def interval_overlap(self, interval_a, interval_b):
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

def rand_uniform(min_v, max_v):
    F = 2147483647
    if max_v < min_v:
        swap = min_v
        min_v = max_v
        max_v = swap

    return ((   random.random() * F/F) * (max_v - min_v)) + min_v

#!/usr/bin/env python
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, CSVLogger, ReduceLROnPlateau
from keras.optimizers import Adadelta, SGD, Adam, Adagrad
from utils import StagedReduceLROnPlateau, parse_settings
import tensorflow as tf
import numpy as np
import os
import sys
import math
import feature_extractor
import loader
import json


from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(1)

# Workaround for not using all GPU memory
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print "usage: ./train.py config_file_name.json"
        sys.exit(0)
    if not os.path.exists(sys.argv[1]):
        print "Config file does not exists!"
        sys.exit(0)

    settings = parse_settings(sys.argv[1])
    print "Training on X_Dataset!"
    fe = feature_extractor.FeatureExtractor(settings)

    
    nets = {"yolo_v1": (fe.yolo_convolutional_net, 0),
            "yolo_v1_att": (fe.yolo_convolutional_net_att, 0),
            "yolo_v1_att_2": (fe.yolo_convolutional_net_att_2, 0),
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
    net.summary()

    if settings["WEIGHTS"] != None:
        print "Loading Weights: {}...".format(settings["WEIGHTS"]),
        net.load_weights(settings["WEIGHTS"])
        print "Done!"

    opt = {
            "SGD":SGD(lr=settings["INITIAL_LR_FREEZE"], momentum=0.9, decay=0.0005),
            "ADAGRAD":Adagrad(lr=1e-5),
            "ADAM":Adam(lr=settings["INITIAL_LR"])
    }


    layers_to_freeze = ["0_conv","2_conv","4_conv","5_conv","6_conv","8_conv","12_conv","18_conv","19_conv","22_conv","24_conv","last_conv"]
    fe.freeze_layers_from_list(layers_to_freeze, mode=False)
    # fe.freeze_layers(freeze_until=settings["FREEZE_NET_UNTIL"])
    net.compile(optimizer=opt[settings["OPTIMIZER"]], loss=fe.golo_loss)


    with open(settings["TRAIN_IMAGES_INDEX"], "r") as f:
        train_instances = len(f.read().splitlines())

    with open(settings["VALID_IMAGES_INDEX"], "r") as f:
        valid_instances = len(f.read().splitlines())

    steps = 1 if math.ceil(train_instances / float(settings["BATCH_SIZE"])) == 1.0 else int(math.ceil(train_instances / float(settings["BATCH_SIZE"])))
    valid_steps = int(math.ceil(valid_instances / float(settings["BATCH_SIZE"])))

    l = loader.GSLoader(settings["TRAIN_IMAGES_INDEX"], settings["IMAGES_PATH"], settings["LABELS_PATH"], settings, show_image=settings["SHOW_IMAGE"], augmentate=settings["AUGMENTATE"], max_boxes=settings["MAX_BOXES"], out_dim_factor=out_dim_factor)
    gen_ll = l.data_generator(settings["BATCH_SIZE"], variable_dim=settings["MULTISCALE"], shuffle=settings["SHUFFLE"])
    
    print "Tunning last layer for {} epochs...".format(settings["LAST_LAYERS_TUNE_EPOCHS"])
    net.fit_generator(gen_ll, steps_per_epoch=steps, epochs=settings["LAST_LAYERS_TUNE_EPOCHS"],verbose=1)
    print "Last layer tunning complete!"

    fe.freeze_layers_from_list(layers_to_freeze, mode=True)
    # fe.freeze_layers(mode="unfreeze", freeze_until=settings["FREEZE_NET_UNTIL"])
    net.compile(optimizer=opt[settings["OPTIMIZER"]], loss=fe.golo_loss)

    l = loader.GSLoader(settings["TRAIN_IMAGES_INDEX"], settings["IMAGES_PATH"], settings["LABELS_PATH"], settings, show_image=settings["SHOW_IMAGE"], augmentate=settings["AUGMENTATE"], max_boxes=settings["MAX_BOXES"], out_dim_factor=out_dim_factor)
    gen = l.data_generator(settings["BATCH_SIZE"], variable_dim=settings["MULTISCALE"], shuffle=settings["SHUFFLE"])

    l_val = loader.GSLoader(settings["VALID_IMAGES_INDEX"], settings["IMAGES_PATH"], settings["LABELS_PATH"], settings, show_image=False, augmentate=False, max_boxes=settings["MAX_BOXES"], out_dim_factor=out_dim_factor)
    gen_val = l_val.data_generator(settings["BATCH_SIZE"], variable_dim=False, shuffle=False)


    if settings["DEBUG"]:
        settings["MODELS_PATH"] += "_DEBUG"

    if not os.path.exists(settings["MODELS_PATH"]):
        os.makedirs(settings["MODELS_PATH"])
        os.makedirs(settings["MODELS_PATH"] + "/log")
   
    checkpoints = []
    if settings["SAVE_DATA"]:
        checkpoints += [ModelCheckpoint(settings["MODELS_PATH"] + "/golo_.{epoch:02d}-{"+settings["MONITORE_VALUE"]+":.8f}.hdf5", 
            monitor=settings["MONITORE_VALUE"], 
            verbose=1, 
            save_best_only=False, 
            mode='min', 
            period=1)]
        checkpoints += [CSVLogger(settings["MODELS_PATH"] + "/trainlog.txt")]
   
    checkpoints += [ReduceLROnPlateau(monitor=settings["MONITORE_VALUE"], factor=0.95, patience=10, verbose=1, mode='min', min_lr=1e-4)]


    net.fit_generator(
        gen,
        validation_data=gen_val,
        validation_steps=valid_steps,
        steps_per_epoch= steps * settings["TRAIN_MULT"],
        epochs=settings["MAX_EPOCHS"], 
        verbose=1,
        callbacks=checkpoints
    )

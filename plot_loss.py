#!/usr/bin/env python
import numpy as np
import os
import sys
from utils import parse_settings
import matplotlib.pyplot as plt


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print "usage: ./train.py config_file_name.json"
        sys.exit(0)
    if not os.path.exists(sys.argv[1]):
        print "Config file does not exists!"
        sys.exit(0)

    settings = parse_settings(sys.argv[1])
    with open(os.path.join(settings["MODELS_PATH"],"trainlog.txt"), "r") as f:
        lines = f.read().splitlines()
    del lines[0]

    loss = []
    val_loss = []
    for line in lines:
        values = line.split(",")
        loss += [float(values[1])]
        val_loss += [float(values[2])]

    best_model = np.asarray(val_loss).argmin()
    print "Best Model:", best_model + 1, "Val_Loss:", val_loss[best_model]
    
    plt.plot(loss, "r", label="Loss")
    plt.plot(val_loss, "b", label="Val_Loss")
    plt.legend()
    plt.show()

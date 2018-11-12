#!/bin/bash

for filename in $(ls -t /data/golo_models/output_xdataset_darknet19_att_sum/*.hdf5)
do
    ./test_models_lot.py "/home/gabriel/python_code/golo/configs/config_xdataset_darknet19_att_sum.json" $filename
    name_len=${#filename}
    # echo $name_len
    echo $filename
    model_id=$(echo ${filename:58:$name_len-74})
    echo $model_id
    mkdir /home/gabriel/tmp/golo_results_valid/$model_id
    mv /tmp/output_boxes/* /home/gabriel/tmp/golo_results_valid/$model_id
done

#!/bin/bash

python main_dnd.py \
 --use_cache \
 --cfg ./configs/WW2020.yml \
 --exp_name WW2020_swinv2s \
 --im_scale 256 \
 --bs 16 \
 --acc_bsz 2 \
 --codalab_pred test \
 --train_set train
#!/bin/bash

python scripts/slu_baseline.py \
    --name $1_lr_$2_bs_$3_step_$4_gamma_$5_ep_$6 \
    --pretrained_model $1 \
    --lr $2 \
    --batch_size $3 \
    --step_size $4 \
    --gamma $5 \
    --max_epoch $6 \
    --device $7 \
    
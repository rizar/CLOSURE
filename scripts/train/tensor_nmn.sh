#!/bin/bash

python $NMN/scripts/train_model.py \
    --model_type EE \
    --num_iterations 500000 \
    --num_val_samples 100000 \
    --load_features 0 \
    --loader_num_workers 1 \
    --record_loss_every 100 \
    --learning_rate 1e-4 $@

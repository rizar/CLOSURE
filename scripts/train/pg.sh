#!/bin/bash

[ -z $NPROC ] && NPROC=1

python\
   $NMN/scripts/train_model.py \
    --model_type PG \
    --num_iterations 200000 \
    --num_val_samples 100000 \
    --load_features 0 \
    --loader_num_workers 1 \
    --record_loss_every 10 \
    --allow_resume \
    --checkpoint_every 400\
    --validate_every 400\
    --learning_rate 7e-4 $@

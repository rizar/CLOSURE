#!/bin/bash

[ -z $NPROC ] && NPROC=1
if [ $NPROC == 1 ]
then
    PYTHON=python
else
    PYTHON="python -m torch.distributed.launch --nproc_per_node=$NPROC"
fi

$PYTHON\
    $NMN/scripts/train_model.py \
    --model_type EE \
    --num_iterations 500000 \
    --num_val_samples 100000 \
    --load_features 0 \
    --loader_num_workers 1 \
    --record_loss_every 100 \
    --learning_rate 1e-4 \
    --classifier_downsample=none \
    --classifier_fc_dims= \
    --classifier_proj_dim=0 \
    --discriminator_downsample=none \
    --discriminator_fc_dims= \
    --discriminator_proj_dim=0 \
    --nmn_use_film=1 \
    --nmn_module_pool=max \
    --module_num_layers=2 \
    --nmn_use_gammas=tanh \
    --classifier_fc_dims=1024 \
    --batch_size 128 \
    $@

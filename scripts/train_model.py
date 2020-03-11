#!/usr/bin/env python3
# This code is released under the MIT License in association with the following paper:
#
# CLOSURE: Assessing Systematic Generalization of CLEVR Models (https://arxiv.org/abs/1912.05783).
#
# Full copyright and license information (including third party attribution) in the NOTICE file (https://github.com/rizar/CLOSURE/NOTICE).

import argparse
import json
import os
import pdb
import random
import shutil
import sys
import subprocess
import time
import logging
import itertools
import lru
import pickle

import h5py
import numpy as np
from termcolor import colored
import torch
torch.backends.cudnn.enabled = True
from torch import autograd
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel, DataParallel

import vr
import vr.utils
import vr.preprocess
from vr.data import (ClevrDataset,
                     ClevrDataLoader)
from vr.models import *
from vr.ns_vqa.parser import Seq2seqParser
from vr.ns_vqa.clevr_executor import ClevrExecutor

parser = argparse.ArgumentParser()
logger = logging.getLogger(__name__)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def is_multigpu():
    nproc_str = os.environ.get('NPROC', '1')
    if not nproc_str:
        return False
    return int(nproc_str) > 1

def atomic_torch_save(object_, path):
    tmp_path = path + '.tmp'
    torch.save(object_, tmp_path)
    shutil.move(tmp_path, path)

def parse_int_list(input_):
    if not input_:
        return []
    return list(map(int, input_.split(',')))

def parse_float_list(input_):
    if not input_:
        return []
    return list(map(float, input_.split(',')))

def one_or_list(parser):
    def parse_one_or_list(input_):
        output = parser(input_)
        if len(output) == 1:
            return output[0]
        else:
            return output
    return parse_one_or_list

def get_parameter_norm(model):
    total_param_norm = 0
    for p in model.parameters():
        total_param_norm += (p ** 2).sum()
    return total_param_norm ** (1. / 2)

def get_parameter_grad_norm(model):
    total_param_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            total_param_norm += (p.grad ** 2).sum()
    return total_param_norm ** (1. / 2)

parser.add_argument("--seed", default=None)

# for DDP launcher
parser.add_argument("--rank", type=int, default=0)
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument("--world_size", type=int, default=1)

# Input data
parser.add_argument('--data_dir', required=True)
parser.add_argument('--val_part', default=[], action='append')

parser.add_argument('--feature_dim', default=[1024,14,14], type=parse_int_list)
parser.add_argument('--vocab_json', default='vocab.json')

parser.add_argument('--load_features', type=int, default=1)
parser.add_argument('--loader_num_workers', type=int, default=0)
parser.add_argument('--use_local_copies', default=0, type=int)
parser.add_argument('--cleanup_local_copies', default=1, type=int)

parser.add_argument('--family_split_file', default=None)
parser.add_argument('--num_train_samples', default=None, type=int)
parser.add_argument('--num_val_samples', default=None, type=int)
parser.add_argument('--shuffle_train_data', default=1, type=int)
parser.add_argument('--oversample', type=int)
parser.add_argument('--oversample_shift', type=int)

parser.add_argument('--percent_of_data_for_training', default=1., type=float)
parser.add_argument('--simple_encoder', default=0, type=int)

# What type of model to use and which parts to train
parser.add_argument('--model_type', default='PG',
  choices=['RTfilm', 'Tfilm', 'FiLM',
           'PG', 'EE', 'PG+EE', 'Control-EE',
           'LSTM', 'CNN+LSTM', 'CNN+LSTM+SA',
           'Hetero', 'MAC',
           'SimpleNMN', 'RelNet', 'SHNMN',
           'ConvLSTM'])
parser.add_argument('--train_program_generator', default=1, type=int)
parser.add_argument('--train_execution_engine', default=1, type=int)
parser.add_argument('--baseline_train_only_rnn', default=0, type=int)

# Start from an existing checkpoint
parser.add_argument('--program_generator_start_from', default=None)
parser.add_argument('--execution_engine_start_from', default=None)
parser.add_argument('--baseline_start_from', default=None)

# RNN options (for PG)
parser.add_argument('--rnn_wordvec_dim', default=300, type=int)
parser.add_argument('--rnn_hidden_dim', default=256, type=int)
parser.add_argument('--rnn_num_layers', default=2, type=int)
parser.add_argument('--rnn_dropout', default=0, type=float)
parser.add_argument('--rnn_attention', action='store_true')
parser.add_argument('--rnn_nonautoreg', action='store_true')
parser.add_argument('--ns_vqa', action='store_true')

# Symbolic EE
parser.add_argument('--symbolic_ee', action='store_true')

# Module net / FiLMedNet options
parser.add_argument('--module_stem_num_layers', default=2, type=int)
parser.add_argument('--module_stem_subsample_layers', default=[], type=parse_int_list)
parser.add_argument('--module_stem_batchnorm', default=0, type=int)
parser.add_argument('--module_dim', default=128, type=int)
parser.add_argument('--stem_dim', default=64, type=int)
parser.add_argument('--module_residual', default=1, type=int)
parser.add_argument('--module_batchnorm', default=0, type=int)
parser.add_argument('--module_intermediate_batchnorm', default=0, type=int)
parser.add_argument('--use_color', default=0, type=int)
parser.add_argument('--nmn_type', default='chain1', choices = ['chain1', 'chain2', 'chain3', 'tree'])

# FiLM only options
parser.add_argument('--set_execution_engine_eval', default=0, type=int)
parser.add_argument('--program_generator_parameter_efficient', default=1, type=int)
parser.add_argument('--rnn_output_batchnorm', default=0, type=int)
parser.add_argument('--bidirectional', default=0, type=int)
parser.add_argument('--encoder_type', default='gru', type=str,
  choices=['linear', 'gru', 'lstm'])
parser.add_argument('--decoder_type', default='linear', type=str,
  choices=['linear', 'gru', 'lstm'])
parser.add_argument('--gamma_option', default='linear',
  choices=['linear', 'sigmoid', 'tanh', 'exp'])
parser.add_argument('--gamma_baseline', default=1, type=float)
parser.add_argument('--num_modules', default=4, type=int)
parser.add_argument('--module_stem_kernel_size', default=[3], type=parse_int_list)
parser.add_argument('--module_stem_stride', default=[1], type=parse_int_list)
parser.add_argument('--module_stem_padding', default=None, type=parse_int_list)
parser.add_argument('--module_num_layers', default=1, type=int)  # Only mnl=1 currently implemented
parser.add_argument('--module_batchnorm_affine', default=0, type=int)  # 1 overrides other factors
parser.add_argument('--module_dropout', default=5e-2, type=float)
parser.add_argument('--module_input_proj', default=1, type=int)  # Inp conv kernel size (0 for None)
parser.add_argument('--module_kernel_size', default=3, type=int)
parser.add_argument('--condition_method', default='bn-film', type=str,
  choices=['nothing', 'block-input-film', 'block-output-film', 'bn-film', 'concat', 'conv-film', 'relu-film'])
parser.add_argument('--condition_pattern', default=[], type=parse_int_list)  # List of 0/1's (len = # FiLMs)
parser.add_argument('--use_gamma', default=1, type=int)
parser.add_argument('--use_beta', default=1, type=int)
parser.add_argument('--use_coords', default=1, type=int)  # 0: none, 1: low usage, 2: high usage
parser.add_argument('--grad_clip', default=0, type=float)  # <= 0 for no grad clipping
parser.add_argument('--debug_every', default=float('inf'), type=float)  # inf for no pdb
parser.add_argument('--print_verbose_every', default=float('inf'), type=float)  # inf for min print
parser.add_argument('--film_use_attention', default=0, type=int)

#MAC options
parser.add_argument('--mac_write_unit', default='original', type=str)
parser.add_argument('--mac_read_connect', default='last', type=str)
parser.add_argument('--mac_read_unit', default='original', type=str)
parser.add_argument('--mac_vib_start', default=0, type=float)
parser.add_argument('--mac_vib_coof', default=0., type=float)
parser.add_argument('--mac_use_self_attention', default=1, type=int)
parser.add_argument('--mac_use_memory_gate', default=1, type=int)
parser.add_argument('--mac_nonlinearity', default='ELU', type=str)
parser.add_argument('--mac_question2output', default=1, type=int)
parser.add_argument('--mac_train_just_control', action='store_true')

parser.add_argument('--mac_question_embedding_dropout', default=0.08, type=float)
parser.add_argument('--mac_stem_dropout', default=0.18, type=float)
parser.add_argument('--mac_memory_dropout', default=0.15, type=float)
parser.add_argument('--mac_read_dropout', default=0.15, type=float)
parser.add_argument('--mac_use_prior_control_in_control_unit', default=0, type=int)
parser.add_argument('--variational_embedding_dropout', default=0.15, type=float)
parser.add_argument('--mac_embedding_uniform_boundary', default=1., type=float)
parser.add_argument('--hard_code_control', action="store_true")

parser.add_argument('--exponential_moving_average_weight', default=1., type=float)

#NMNFilm2 options
parser.add_argument('--nmn_use_film', default=0, type=int)
parser.add_argument('--nmn_use_simple_block', default=0, type=int)
parser.add_argument('--nmn_module_pool', default='mean', type=str)
parser.add_argument('--nmn_use_gammas', default='identity', type=str)
parser.add_argument('--nmn_learn_control', default=0, type=int)

parser.add_argument('--entropy_coef', default=0.0, type=float)


# CNN options (for baselines)
parser.add_argument('--cnn_res_block_dim', default=128, type=int)
parser.add_argument('--cnn_num_res_blocks', default=0, type=int)
parser.add_argument('--cnn_proj_dim', default=512, type=int)
parser.add_argument('--cnn_pooling', default='maxpool2',
  choices=['none', 'maxpool2'])

# Stacked-Attention options
parser.add_argument('--stacked_attn_dim', default=512, type=int)
parser.add_argument('--num_stacked_attn', default=2, type=int)

# Classifier options
parser.add_argument('--classifier_proj_dim', default=512, type=int)
parser.add_argument('--classifier_downsample', default='maxpool2',
  choices=['maxpool2', 'maxpool3', 'maxpool4', 'maxpool5', 'maxpool7', 'maxpoolfull', 'none',
           'avgpool2', 'avgpool3', 'avgpool4', 'avgpool5', 'avgpool7', 'avgpoolfull', 'aggressive',
           'hybrid'])
parser.add_argument('--classifier_fc_dims', default=[1024], type=parse_int_list)
parser.add_argument('--classifier_batchnorm', default=0, type=int)
parser.add_argument('--classifier_dropout', default=0.0, type=one_or_list(parse_float_list))

# Discriminator options
parser.add_argument('--discriminator_proj_dim', default=512, type=int)
parser.add_argument('--discriminator_downsample', default='maxpool2',
  choices=['maxpool2', 'maxpool3', 'maxpool4', 'maxpool5', 'maxpool7', 'maxpoolfull', 'none',
           'avgpool2', 'avgpool3', 'avgpool4', 'avgpool5', 'avgpool7', 'avgpoolfull', 'aggressive',
           'hybrid'])
parser.add_argument('--discriminator_fc_dims', default=[1024], type=parse_int_list)
parser.add_argument('--discriminator_dropout', default=0.0, type=one_or_list(parse_float_list))

# Optimization options
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--val_batch_size', default=512, type=int)
parser.add_argument('--num_iterations', default=100000, type=int)
parser.add_argument('--optimizer', default='Adam',
  choices=['Adadelta', 'Adagrad', 'Adam', 'Adamax', 'ASGD', 'RMSprop', 'SGD'])
parser.add_argument('--learning_rate', default=5e-4, type=float)
parser.add_argument('--pg_learning_rate', default=None, type=float)
parser.add_argument('--beta1', default=0.9, type=float)
parser.add_argument('--beta2', default=0.999, type=float)
parser.add_argument('--eps', default=1e-8, type=float)
parser.add_argument('--reward_decay', default=0.9, type=float)
parser.add_argument('--same_reward', action="store_true", default=False)
parser.add_argument('--weight_decay', default=0, type=float)
parser.add_argument('--ewa_baseline', default=1, type=int)
parser.add_argument('--enforce_wellformed', default=False, action="store_true")
parser.add_argument('--temperature_increase', default=None, type=float)

# Output options
parser.add_argument('--checkpoint_path', default='{slurmid}.pt')
parser.add_argument('--allow_resume', action='store_true')
parser.add_argument('--load_ee_parameters', default=None, type=str)
parser.add_argument('--randomize_checkpoint_path', type=int, default=0)
parser.add_argument('--avoid_checkpoint_override', default=0, type=int)
parser.add_argument('--record_loss_every', default=1, type=int)
parser.add_argument('--checkpoint_every', default=400, type=int)
parser.add_argument('--validate_every', default=10000, type=int)
parser.add_argument('--time', default=0, type=int)


def main(args):
    if args.validate_every % args.checkpoint_every != 0:
        raise ValueError("must validate at iteration where checkpointing is also done")

    if is_multigpu():
        torch.distributed.init_process_group(backend='nccl')

    global device
    device = (torch.device('cuda:{}'.format(args.local_rank))
              if torch.cuda.is_available()
              else torch.device('cpu'))

    if args.seed is not None:
        torch.manual_seed(args.seed)

    nmn_iwp_code = list(vr.__path__)[0]

    try:
        last_commit = subprocess.check_output(
            'cd {}; git log -n1'.format(nmn_iwp_code), shell=True).decode('utf-8')
        logger.info('LAST COMMIT INFO:')
        logger.info(last_commit)
    except subprocess.CalledProcessError:
        logger.info('Could not figure out the last commit')
    try:
        diff = subprocess.check_output(
            'cd {}; git diff'.format(nmn_iwp_code), shell=True).decode('utf-8')
        if diff:
            logger.info('GIT DIFF:')
            logger.info(diff)
    except subprocess.CalledProcessError:
        logger.info('Could not figure out the last commit')

    logger.info('Will save checkpoints to %s' % args.checkpoint_path)

    args.vocab_json = os.path.join(args.data_dir, args.vocab_json)

    if not args.checkpoint_path:
        raise NotImplementedError('no default checkpoint path')

    args.vocab_json = os.path.join(args.data_dir, args.vocab_json)
    vocab = vr.utils.load_vocab(args.vocab_json)

    logger.info(args)
    question_families = None
    if args.family_split_file is not None:
        with open(args.family_split_file, 'r') as f:
            question_families = json.load(f)


    scenes_needed = args.symbolic_ee
    features_needed = args.model_type != 'PG' and not args.symbolic_ee

    train_question_h5 = os.path.join(args.data_dir, 'train_questions.h5')
    train_features_h5 = os.path.join(args.data_dir, 'train_features.h5')
    train_scenes = os.path.join(args.data_dir, 'train_scenes.json')
    train_loader_kwargs = {
        'question_h5': train_question_h5,
        'feature_h5': train_features_h5 if features_needed else None,
        'scene_path': train_scenes if scenes_needed else None,
        'load_features': args.load_features,
        'vocab': vocab,
        'batch_size': args.batch_size,
        'shuffle': args.shuffle_train_data == 1,
        'question_families': question_families,
        'max_samples': args.num_train_samples,
        'num_workers': args.loader_num_workers,
        'percent_of_data': args.percent_of_data_for_training,
        'oversample': args.oversample,
        'oversample_shift': args.oversample_shift
    }
    train_loader = ClevrDataLoader(**train_loader_kwargs)

    val_loaders = []
    for val_part in args.val_part:
        val_question_h5 = os.path.join(args.data_dir, '{}_questions.h5'.format(val_part))
        val_features_h5 = os.path.join(args.data_dir, '{}_features.h5'.format(val_part))
        val_scenes = os.path.join(args.data_dir, '{}_scenes.json'.format(val_part))
        val_loader_kwargs = {
            'question_h5': val_question_h5,
            'feature_h5': val_features_h5 if features_needed else None,
            'scene_path': val_scenes if scenes_needed else None,
            'load_features': args.load_features,
            'vocab': vocab,
            'batch_size': args.val_batch_size,
            'question_families': question_families,
            'max_samples': args.num_val_samples,
            'num_workers': args.loader_num_workers,
        }
        val_loaders.append(ClevrDataLoader(**val_loader_kwargs))

    try:
        train_loop(args, train_loader, val_loaders)
    finally:
        for loader in [train_loader] + val_loaders:
            loader.close()


def train_loop(args, train_loader, val_loaders):
    vocab = vr.utils.load_vocab(args.vocab_json)
    program_generator, pg_kwargs, pg_optimizer = None, None, None
    execution_engine, ee_kwargs, ee_optimizer = None, None, None
    baseline_model, baseline_kwargs, baseline_optimizer = None, None, None
    baseline_type = None

    stats = {
      'train_losses': [], 'train_rewards': [], 'train_losses_ts': [],
      'train_accs': [], 'val_accs_ts': [], 'alphas' : [], 'grads' : [],
      'model_t': 0, 'model_epoch': 0,
      'entropy': [], 'prog_acc': [], 'compute_time': []
    }
    for val_part in args.val_part:
        stats['best_' + val_part + '_acc'] = -1
        stats[val_part + "_accs"] = []

    models_that_need_pg = ['MAC', 'RTfilm', 'Tfilm', 'FiLM',
                           'PG', 'PG+EE', 'Control-EE', 'RelNet', 'ConvLSTM']
    models_that_need_ee = ['MAC', 'RTfilm', 'Tfilm', 'FiLM', 'EE', 'PG+EE',
                           'Control-EE', 'Hetero', 'SimpleNMN', 'SHNMN', 'RelNet', 'ConvLSTM']

    # Set up model
    if args.allow_resume and os.path.exists(args.checkpoint_path):
        # EITHER resume existing experiment
        logger.info("Trying to resume")
        if args.model_type in models_that_need_pg:
            program_generator, pg_kwargs = vr.utils.load_program_generator(args.checkpoint_path)
            program_generator.to(device)
            if is_multigpu():
                program_generator = DistributedDataParallel(program_generator, device_ids=[args.local_rank])
        if args.model_type in models_that_need_ee:
            if args.symbolic_ee:
                execution_engine, ee_kwargs = get_execution_engine(args)
            else:
                execution_engine, ee_kwargs  = vr.utils.load_execution_engine(args.checkpoint_path)
                execution_engine.to(device)
                if is_multigpu():
                    execution_engine = DistributedDataParallel(execution_engine, device_ids=[args.local_rank])
        with open(args.checkpoint_path + '.json', 'r') as f:
            checkpoint = json.load(f)
        for key in list(stats.keys()):
            if key in checkpoint:
                stats[key] = checkpoint[key]
        stats['model_epoch'] -= 1
        best_pg_state = get_state(program_generator)
        best_ee_state = get_state(execution_engine)
        # no support for PG+EE her
        best_baseline_state = None
    else:
        # OR start a new one
        if args.model_type in models_that_need_pg:
            program_generator, pg_kwargs = get_program_generator(args)

            logger.info('Here is the conditioning network:')
            logger.info(program_generator)
        if args.model_type in models_that_need_ee:
            execution_engine, ee_kwargs = get_execution_engine(args)
            logger.info('Here is the conditioned network:')
            logger.info(execution_engine)
        if args.model_type in ['LSTM', 'CNN+LSTM', 'CNN+LSTM+SA']:
            baseline_model, baseline_kwargs = get_baseline_model(args)
            params = baseline_model.parameters()
            if args.baseline_train_only_rnn == 1:
                params = baseline_model.rnn.parameters()
            logger.info('Here is the baseline model')
            logger.info(baseline_model)
            baseline_type = args.model_type

    if args.load_ee_parameters:
        state = vr.utils.load_cpu(args.load_ee_parameters)
        execution_engine.load_state_dict(state['execution_engine_state'], strict=False)

    optim_method = getattr(torch.optim, args.optimizer)
    if program_generator:
        pg_learning_rate = args.pg_learning_rate
        if pg_learning_rate is None:
            pg_learning_rate = args.learning_rate
        pg_optimizer = optim_method(program_generator.parameters(),
                                    lr=pg_learning_rate,
                                    weight_decay=args.weight_decay,
                                    eps=args.eps)
    if execution_engine and not args.symbolic_ee:
        if args.mac_train_just_control:
            parameters = list(execution_engine.controlUnit.parameters())
            for inpUnit in execution_engine.inputUnits:
                parameters.extend(list(inpUnit.parameters()))
        else:
            parameters = execution_engine.parameters()
        ee_optimizer = optim_method(parameters,
                                    lr=args.learning_rate,
                                    weight_decay=args.weight_decay,
                                    eps=args.eps)
    if baseline_model:
        baseline_optimizer = optim_method(params,
                                          lr=args.learning_rate,
                                          weight_decay=args.weight_decay)

    loss_fn = torch.nn.CrossEntropyLoss().to(device)

    t, epoch, reward_moving_average = stats['model_t'], stats['model_epoch'], 0

    set_mode('train', [program_generator, execution_engine, baseline_model])

    logger.info('train_loader has {} samples'.format(len(train_loader.dataset)))
    for val_part, val_loader in zip(args.val_part, val_loaders):
        logger.info('{}_loader has {} samples'.format(val_part, len(val_loader.dataset)))

    num_checkpoints = 0
    epoch_start_time = 0.0
    epoch_total_time = 0.0
    train_pass_total_time = 0.0
    val_pass_total_time = 0.0
    valB_pass_total_time = 0.0
    running_loss = 0.0


    cache = [lru.LRU(10) for i in range(len(train_loader.dataset))]

    while t < args.num_iterations:
        if (epoch > 0) and (args.time == 1):
            epoch_time = time.time() - epoch_start_time
            epoch_total_time += epoch_time
            logger.info('EPOCH PASS AVG TIME: ' + str(epoch_total_time / epoch), 'white')
            logger.info('Epoch Pass Time      : ' + str(epoch_time), 'white')
        epoch_start_time = time.time()

        fwd_pass_time = 0.
        bwd_pass_time = 0.

        epoch += 1
        logger.info('Starting epoch %d' % epoch)

        batch_start_time = time.time()
        for batch in train_loader:
            compute_start_time = time.time()

            t += 1
            acc = None
            prog_acc = None
            entropy = None

            data_moving_start_time = time.time()
            (questions, indices, feats, scenes, answers, programs) = batch
            if isinstance(questions, list):
                questions = questions[0]
                questions = questions[:, :(questions.sum(0) > 0).sum()]
            questions_var = Variable(questions.to(device))
            feats_var = Variable(feats.to(device))
            answers_var = Variable(answers.to(device))
            if programs[0] is not None:
                programs_var = Variable(programs.to(device))
            data_moving_time = time.time() - compute_start_time

            reward = None
            if args.model_type == 'PG':
                # Train program generator with ground-truth programs
                pg_optimizer.zero_grad()
                loss = program_generator.log_likelihood(questions_var, programs_var).mean()
                loss.backward()
                pg_optimizer.step()
            elif args.model_type in ['EE', 'Hetero']:
                # Train execution engine with ground-truth programs
                ee_optimizer.zero_grad()

                scores, _, _ = execution_engine(feats_var, programs_var, question=questions_var)
                full_loss = loss = loss_fn(scores, answers_var)
                acc = (scores.argmax(1) == answers_var).float().mean()

                full_loss.backward()
                ee_optimizer.step()
            elif args.model_type in ['Control-EE']:
                pg_optimizer.zero_grad()
                ee_optimizer.zero_grad()

                question_repr = program_generator(questions_var)
                scores, _, _ = execution_engine(feats_var, programs_var, question=question_repr)
                loss = loss_fn(scores, answers_var)
                acc = (scores.argmax(1) == answers_var).float().mean()

                loss = loss_fn(scores, answers_var)
                loss.backward()

                pg_optimizer.step()
                ee_optimizer.step()
            elif args.model_type in ['LSTM', 'CNN+LSTM', 'CNN+LSTM+SA']:
                baseline_optimizer.zero_grad()
                baseline_model.zero_grad()
                scores = baseline_model(questions_var, feats_var)
                loss = loss_fn(scores, answers_var)
                loss.backward()
                baseline_optimizer.step()
            elif args.model_type == 'PG+EE':
                programs_pred, token_logprobs = program_generator.forward(questions_var)

                if args.symbolic_ee:
                    preds = execution_engine(scenes, programs_pred)
                else:
                    with torch.set_grad_enabled(bool(args.train_execution_engine)):
                        scores, program_wellformed, _ = execution_engine(feats_var, programs_pred)
                    preds = scores.argmax(1).cpu()
                    if args.enforce_wellformed:
                        preds[~program_wellformed] = -1
                    loss = loss_fn(scores, answers_var)

                raw_reward = (preds == answers).float()
                acc = raw_reward.mean()
                if args.symbolic_ee:
                    loss = -acc
                reward_moving_average *= args.reward_decay
                reward_moving_average += (1.0 - args.reward_decay) * raw_reward.mean()
                centered_reward = raw_reward - (reward_moving_average if args.ewa_baseline else 0.5)
                entropy = -token_logprobs.sum(1).mean()

                min_length = min(programs_var.shape[1], programs_pred.shape[1])
                programs_pred = programs_pred[:, :min_length]
                programs_var = programs_var[:, :min_length]
                correct = (programs_pred == programs_var).int().sum(1) == min_length
                prog_acc = correct.float().mean()


                if args.train_execution_engine == 1:
                    ee_optimizer.zero_grad()
                    loss.backward()
                    ee_optimizer.step()

                if args.train_program_generator == 1:
                    pg_optimizer.zero_grad()
                    weights = centered_reward.to(device)[:, None]
                    if args.entropy_coef:
                        # maximizing entropy = using -logprobs as rewards
                        weights += args.entropy_coef * -token_logprobs.sum(1)[:, None].detach()
                    if args.same_reward:
                        weights = weights.mean()
                    surrogate_loss = (-token_logprobs * weights).sum(1).mean()
                    surrogate_loss.backward()
                    pg_optimizer.step()
            elif args.model_type == 'FiLM' or args.model_type == 'MAC':
                if args.set_execution_engine_eval == 1:
                    set_mode('eval', [execution_engine])

                forward_start_time = time.time()
                programs_pred = program_generator(questions_var)
                scores = execution_engine(feats_var, programs_pred)
                loss = loss_fn(scores, answers_var)
                full_loss = loss.clone()
                fwd_pass_time = time.time() - forward_start_time

                backward_start_time = time.time()
                profile_step = t % 66 == 0
                with torch.autograd.profiler.profile(enabled=profile_step, use_cuda=True) as prof:
                    pg_optimizer.zero_grad()
                    ee_optimizer.zero_grad()
                    if args.debug_every <= -2:
                        pdb.set_trace()
                    full_loss.backward()
                    if args.debug_every < float('inf'):
                        check_grad_num_nans(execution_engine, 'FiLMedNet' if args.model_type == 'FiLM' else args.model_type)
                        check_grad_num_nans(program_generator, 'FiLMGen')
                if profile_step:
                    with open(args.checkpoint_path + '.prof', 'wb') as dest:
                        pickle.dump(prof, dest)
                    print('profile dumped')
                bwd_pass_time = time.time() - backward_start_time

                if args.model_type == 'MAC':
                    if args.train_program_generator == 1 or args.train_execution_engine == 1:
                        if args.grad_clip > 0:
                            allMacParams = itertools.chain(program_generator.parameters(), execution_engine.parameters())
                            torch.nn.utils.clip_grad_norm_(allMacParams, args.grad_clip)
                        pg_optimizer.step()
                        ee_optimizer.step()
                else:
                    if args.train_program_generator == 1:
                        if args.grad_clip > 0:
                            torch.nn.utils.clip_grad_norm(program_generator.parameters(), args.grad_clip)
                        pg_optimizer.step()
                    if args.train_execution_engine == 1:
                        if args.grad_clip > 0:
                            torch.nn.utils.clip_grad_norm(execution_engine.parameters(), args.grad_clip)
                        ee_optimizer.step()
            elif args.model_type == 'Tfilm':
                if args.set_execution_engine_eval == 1:
                    set_mode('eval', [execution_engine])
                programs_pred = program_generator(questions_var)
                scores = execution_engine(feats_var, programs_pred, programs_var)
                loss = loss_fn(scores, answers_var)

                pg_optimizer.zero_grad()
                ee_optimizer.zero_grad()
                if args.debug_every <= -2:
                    pdb.set_trace()
                loss.backward()
                if args.debug_every < float('inf'):
                    check_grad_num_nans(execution_engine, 'TFiLMedNet' if args.model_type == 'Tfilm' else 'NMNFiLMedNet')
                    check_grad_num_nans(program_generator, 'FiLMGen')

                if args.train_program_generator == 1:
                    if args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm(program_generator.parameters(), args.grad_clip)
                    pg_optimizer.step()
                if args.train_execution_engine == 1:
                    if args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm(execution_engine.parameters(), args.grad_clip)
                    ee_optimizer.step()
            elif args.model_type == 'RTfilm':
                if args.set_execution_engine_eval == 1:
                    set_mode('eval', [execution_engine])
                programs_pred = program_generator(questions_var)
                scores = execution_engine(feats_var, programs_pred)
                loss = loss_fn(scores, answers_var)

                pg_optimizer.zero_grad()
                ee_optimizer.zero_grad()
                if args.debug_every <= -2:
                    pdb.set_trace()
                loss.backward()
                if args.debug_every < float('inf'):
                    check_grad_num_nans(execution_engine, 'RTFiLMedNet')
                    check_grad_num_nans(program_generator, 'FiLMGen')

                if args.train_program_generator == 1:
                    if args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm(program_generator.parameters(), args.grad_clip)
                    pg_optimizer.step()
                if args.train_execution_engine == 1:
                    if args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm(execution_engine.parameters(), args.grad_clip)
                    ee_optimizer.step()
            elif args.model_type in ['RelNet', 'ConvLSTM']:
                question_rep = program_generator(questions_var)
                scores = execution_engine(feats_var, question_rep)
                loss = loss_fn(scores, answers_var)

                pg_optimizer.zero_grad()
                ee_optimizer.zero_grad()
                loss.backward()
                pg_optimizer.step()
                ee_optimizer.step()
            else:
                raise ValueError()

            if torch.isnan(loss).item():
                print("NAN!")
                sys.exit(1)

            if t == args.num_iterations:
            # Save the best model separately
                break

            if t % args.record_loss_every == 0:
                if 'miss_mask' in locals():
                    print(miss_mask.sum())

                running_loss += loss.item()
                avg_loss = running_loss / args.record_loss_every
                compute_time = time.time() - compute_start_time
                batch_time = time.time() - batch_start_time
                logger_format = "iter: {} t_b: {:.5f} t_c: {:.5f} t_m: {:.5f} t_fwd: {:.5f} t_bwd: {:.5f} loss: {:.5f}"
                logger_data = (
                    t, batch_time, compute_time,
                    data_moving_time, fwd_pass_time, bwd_pass_time, avg_loss)
                if acc is not None:
                    logger_format += " acc: {:.5f}"
                    logger_data += (acc.item(),)
                if prog_acc is not None:
                    logger_format += " prog_acc: {:.5f}"
                    logger_data += (prog_acc.item(),)
                if entropy is not None:
                    logger_format += " H: {:.9f}"
                    logger_data += (entropy.item(),)
                logger.info(logger_format.format(*logger_data))
                stats['train_losses'].append(avg_loss)
                if prog_acc:
                    stats['prog_acc'].append(prog_acc.item())
                if entropy:
                    stats['entropy'].append(entropy.item())
                stats['train_losses_ts'].append(t)
                if reward is not None:
                    stats['train_rewards'].append(reward.item())
                stats['compute_time'].append(compute_time)
                running_loss = 0.0
            else:
                running_loss += loss.item()

            batch_start_time = time.time()
            if args.local_rank > 0:
                continue

            if t == 1 or t % args.validate_every == 0:
                logger.info('Checking training accuracy ... ')
                start = time.time()
                train_acc = check_accuracy(args, program_generator, execution_engine,
                                           baseline_model, train_loader)
                train_pass_time = (time.time() - start)
                logger.info('train pass time: ' + str(train_pass_time))
                logger.info('train accuracy is {}'.format(train_acc))
                logger.info('Checking validation accuracy ...')
                stats['train_accs'].append(train_acc)

                first_val_acc = None
                for val_part, val_loader in zip(args.val_part, val_loaders):
                    start = time.time()
                    val_acc = check_accuracy(args, program_generator, execution_engine,
                                             baseline_model, val_loader)
                    if first_val_acc is None:
                        first_val_acc = val_acc
                    val_pass_time = (time.time() - start)
                    logger.info('{} pass time: {}'.format(val_part, val_pass_time))
                    logger.info('{} accuracy is {}'.format(val_part, val_acc))
                    stats['{}_accs'.format(val_part)].append(val_acc)
                stats['val_accs_ts'].append(t)

            if t == 1 or t % args.checkpoint_every == 0:
                pg_state = get_state(program_generator)
                ee_state = get_state(execution_engine)
                baseline_state = get_state(baseline_model)

                stats['model_t'] = t
                stats['model_epoch'] = epoch

                checkpoint = {
                    'args': args.__dict__,
                    'program_generator_kwargs': pg_kwargs,
                    'program_generator_state': pg_state,
                    'execution_engine_kwargs': ee_kwargs,
                    'execution_engine_state': ee_state,
                    'baseline_kwargs': baseline_kwargs,
                    'baseline_state': baseline_state,
                    'baseline_type': baseline_type,
                    'vocab': vocab
                }
                for k, v in stats.items():
                    checkpoint[k] = v

                # Save current model
                logger.info('Saving checkpoint to %s' % args.checkpoint_path)
                atomic_torch_save(checkpoint, args.checkpoint_path)

                # Save training status in a human-readable format
                del checkpoint['program_generator_state']
                del checkpoint['execution_engine_state']
                del checkpoint['baseline_state']
                with open(args.checkpoint_path + '.json', 'w') as f:
                    json.dump(checkpoint, f, indent=2, sort_keys=True)

            # Save the best model separately

            if t == 1 or t % args.validate_every == 0:
                for val_part in args.val_part:
                    cur_acc = stats['{}_accs'.format(val_part)][-1]
                    best_acc_key = 'best_{}_acc'.format(val_part)
                    if cur_acc > stats.get(best_acc_key, -1):
                        best_path = '{}.{}.best'.format(args.checkpoint_path, val_part)
                        logger.info('Saving best so far checkpoint to ' + best_path)
                        stats[best_acc_key] = cur_acc
                        checkpoint['program_generator_state'] = pg_state
                        checkpoint['execution_engine_state'] = ee_state
                        checkpoint['baseline_state'] = baseline_state
                        atomic_torch_save(checkpoint, best_path)


def get_state(m):
    if m is None:
        return None
    if isinstance(m, DistributedDataParallel):
        return get_state(m.module)
    if isinstance(m, ClevrExecutor):
        return {}

    state = {}
    for k, v in m.state_dict().items():
        state[k] = v.clone()
    return state


def get_program_generator(args):
    vocab = vr.utils.load_vocab(args.vocab_json)
    if args.program_generator_start_from is not None:
        logger.info('start from pretrained PG')
        pg, kwargs = vr.utils.load_program_generator(args.program_generator_start_from)

        if args.temperature_increase:
            pg.decoder_linear.weight.data /= args.temperature_increase
            pg.decoder_linear.bias.data /= args.temperature_increase
    elif args.ns_vqa:
        pg, kwargs = Seq2seqParser(vocab), {}
    else:
        kwargs = {
          'encoder_vocab_size': len(vocab['question_token_to_idx']),
          'decoder_vocab_size': len(vocab['program_token_to_idx']),
          'wordvec_dim': args.rnn_wordvec_dim,
          'hidden_dim': args.rnn_hidden_dim,
          'rnn_num_layers': args.rnn_num_layers,
          'rnn_dropout': args.rnn_dropout,
        }
        if args.model_type in ['FiLM', 'Tfilm', 'RTfilm', 'MAC', 'Control-EE']:
            kwargs['parameter_efficient'] = args.program_generator_parameter_efficient == 1
            kwargs['output_batchnorm'] = args.rnn_output_batchnorm == 1
            kwargs['bidirectional'] = args.bidirectional == 1
            kwargs['encoder_type'] = args.encoder_type
            kwargs['decoder_type'] = args.decoder_type
            kwargs['gamma_option'] = args.gamma_option
            kwargs['gamma_baseline'] = args.gamma_baseline

            kwargs['use_attention'] = args.film_use_attention == 1

            if args.model_type == 'FiLM' or args.model_type == 'MAC':
                kwargs['num_modules'] = args.num_modules
            elif args.model_type == 'Tfilm':
                kwargs['num_modules'] = args.max_program_module_arity * args.max_program_tree_depth + 1
            elif args.model_type == 'RTfilm':
                treeArities = TreeGenerator().gen(args.tree_type_for_RTfilm)
                kwargs['num_modules'] = len(treeArities)
            if args.model_type == 'MAC' or args.model_type == 'Control-EE':
                kwargs['taking_context'] = True
                kwargs['use_attention'] = False
                kwargs['variational_embedding_dropout'] = args.variational_embedding_dropout
                kwargs['embedding_uniform_boundary'] = args.mac_embedding_uniform_boundary
            kwargs['module_num_layers'] = args.module_num_layers
            kwargs['module_dim'] = args.module_dim
            kwargs['debug_every'] = args.debug_every
            pg = FiLMGen(**kwargs)
        elif args.model_type in ['RelNet', 'ConvLSTM']:
            kwargs['bidirectional'] = args.bidirectional == 1
            kwargs['encoder_type'] = args.encoder_type
            kwargs['taking_context'] = True   # return the last hidden state of LSTM
            pg = FiLMGen(**kwargs)
        elif args.rnn_attention:
            kwargs['autoregressive'] = not args.rnn_nonautoreg
            pg = Seq2SeqAtt(**kwargs)
        else:
            pg = Seq2Seq(**kwargs)
    pg.to(device)
    pg.train()
    if is_multigpu():
        pg = DistributedDataParallel(pg, device_ids=[args.local_rank])
    return pg, kwargs


def get_execution_engine(args):
    vocab = vr.utils.load_vocab(args.vocab_json)
    if args.symbolic_ee:
        return ClevrExecutor(vocab), {}
    if args.execution_engine_start_from is not None:
        logger.info("start from pretrained EE")
        ee, kwargs = vr.utils.load_execution_engine(args.execution_engine_start_from)
    else:
        kwargs = {
            'vocab': vocab,
          'feature_dim': args.feature_dim,
          'stem_batchnorm': args.module_stem_batchnorm == 1,
          'stem_num_layers': args.module_stem_num_layers,
          'stem_subsample_layers': args.module_stem_subsample_layers,
          'stem_kernel_size': args.module_stem_kernel_size,
          'stem_stride': args.module_stem_stride,
          'stem_padding': args.module_stem_padding,
          'stem_dim': args.stem_dim,
          'module_dim': args.module_dim,
          'module_kernel_size': args.module_kernel_size,
          'module_residual': args.module_residual == 1,
          'module_input_proj': args.module_input_proj,
          'module_batchnorm': args.module_batchnorm == 1,
          'classifier_proj_dim': args.classifier_proj_dim,
          'classifier_downsample': args.classifier_downsample,
          'classifier_fc_layers': args.classifier_fc_dims,
          'classifier_batchnorm': args.classifier_batchnorm == 1,
          'classifier_dropout': args.classifier_dropout,
        }
        if args.model_type == 'FiLM':
            kwargs['num_modules'] = args.num_modules
            kwargs['stem_kernel_size'] = args.module_stem_kernel_size
            kwargs['stem_stride'] = args.module_stem_stride
            kwargs['stem_padding'] = args.module_stem_padding
            kwargs['module_num_layers'] = args.module_num_layers
            kwargs['module_intermediate_batchnorm'] = args.module_intermediate_batchnorm == 1
            kwargs['module_batchnorm_affine'] = args.module_batchnorm_affine == 1
            kwargs['module_dropout'] = args.module_dropout
            kwargs['module_input_proj'] = args.module_input_proj
            kwargs['module_kernel_size'] = args.module_kernel_size
            kwargs['use_gamma'] = args.use_gamma == 1
            kwargs['use_beta'] = args.use_beta == 1
            kwargs['use_coords'] = args.use_coords
            kwargs['debug_every'] = args.debug_every
            kwargs['print_verbose_every'] = args.print_verbose_every
            kwargs['condition_method'] = args.condition_method
            kwargs['condition_pattern'] = args.condition_pattern
            ee = FiLMedNet(**kwargs)
        elif args.model_type == 'Tfilm':
            kwargs['num_modules'] = args.max_program_module_arity * args.max_program_tree_depth + 1

            kwargs['max_program_module_arity'] = args.max_program_module_arity
            kwargs['max_program_tree_depth'] = args.max_program_tree_depth

            kwargs['stem_kernel_size'] = args.module_stem_kernel_size
            kwargs['stem_stride'] = args.module_stem_stride
            kwargs['stem_padding'] = args.module_stem_padding
            kwargs['module_num_layers'] = args.module_num_layers
            kwargs['module_intermediate_batchnorm'] = args.module_intermediate_batchnorm == 1
            kwargs['module_batchnorm_affine'] = args.module_batchnorm_affine == 1
            kwargs['module_dropout'] = args.module_dropout
            kwargs['module_input_proj'] = args.module_input_proj
            kwargs['module_kernel_size'] = args.module_kernel_size
            kwargs['use_gamma'] = args.use_gamma == 1
            kwargs['use_beta'] = args.use_beta == 1
            kwargs['use_coords'] = args.use_coords
            kwargs['debug_every'] = args.debug_every
            kwargs['print_verbose_every'] = args.print_verbose_every
            kwargs['condition_method'] = args.condition_method
            kwargs['condition_pattern'] = args.condition_pattern
            ee = TFiLMedNet(**kwargs)
        elif args.model_type == 'RTfilm':
            treeArities = TreeGenerator().gen(args.tree_type_for_RTfilm)
            kwargs['num_modules'] = len(treeArities)
            kwargs['treeArities'] = treeArities
            kwargs['tree_type_for_RTfilm'] = args.tree_type_for_RTfilm
            kwargs['share_module_weight_at_depth'] = args.share_module_weight_at_depth

            kwargs['stem_kernel_size'] = args.module_stem_kernel_size
            kwargs['stem_stride'] = args.module_stem_stride
            kwargs['stem_padding'] = args.module_stem_padding
            kwargs['module_num_layers'] = args.module_num_layers
            kwargs['module_intermediate_batchnorm'] = args.module_intermediate_batchnorm == 1
            kwargs['module_batchnorm_affine'] = args.module_batchnorm_affine == 1
            kwargs['module_dropout'] = args.module_dropout
            kwargs['module_input_proj'] = args.module_input_proj
            kwargs['module_kernel_size'] = args.module_kernel_size
            kwargs['use_gamma'] = args.use_gamma == 1
            kwargs['use_beta'] = args.use_beta == 1
            kwargs['use_coords'] = args.use_coords
            kwargs['debug_every'] = args.debug_every
            kwargs['print_verbose_every'] = args.print_verbose_every
            kwargs['condition_method'] = args.condition_method
            kwargs['condition_pattern'] = args.condition_pattern
            ee = RTFiLMedNet(**kwargs)
        elif args.model_type == 'MAC':
            kwargs = {
                      'vocab': vocab,
                      'feature_dim': args.feature_dim,
                      'stem_num_layers': args.module_stem_num_layers,
                      'stem_batchnorm': args.module_stem_batchnorm == 1,
                      'stem_kernel_size': args.module_stem_kernel_size,
                      'stem_subsample_layers': args.module_stem_subsample_layers,
                      'stem_stride': args.module_stem_stride,
                      'stem_padding': args.module_stem_padding,
                      'num_modules': args.num_modules,
                      'module_dim': args.module_dim,
                      'stem_dim': args.stem_dim,

                      #'module_dropout': args.module_dropout,
                      'question_embedding_dropout': args.mac_question_embedding_dropout,
                      'stem_dropout': args.mac_stem_dropout,
                      'memory_dropout': args.mac_memory_dropout,
                      'read_dropout': args.mac_read_dropout,
                      'write_unit': args.mac_write_unit,
                      'read_connect': args.mac_read_connect,
                      'read_unit': args.mac_read_unit,
                      'question2output': args.mac_question2output,
                      'noisy_controls': bool(args.mac_vib_coof),
                      'use_prior_control_in_control_unit': args.mac_use_prior_control_in_control_unit == 1,
                      'use_self_attention': args.mac_use_self_attention,
                      'use_memory_gate': args.mac_use_memory_gate,
                      'nonlinearity': args.mac_nonlinearity,

                      'classifier_fc_layers': args.classifier_fc_dims,
                      'classifier_batchnorm': args.classifier_batchnorm == 1,
                      'classifier_dropout': args.classifier_dropout,
                      'use_coords': args.use_coords,
                      'debug_every': args.debug_every,
                      'print_verbose_every': args.print_verbose_every,
                      'hard_code_control' : args.hard_code_control
                      }
            ee = MAC(**kwargs)
        elif args.model_type == 'Hetero':
            kwargs = {
              'vocab': vocab,
              'feature_dim': args.feature_dim,
              'stem_batchnorm': args.module_stem_batchnorm == 1,
              'stem_num_layers': args.module_stem_num_layers,
              'stem_kernel_size': args.module_stem_kernel_size,
              'stem_stride': args.module_stem_stride,
              'stem_padding': args.module_stem_padding,
              'module_dim': args.module_dim,
              'stem_dim': args.stem_dim,
              'module_batchnorm': args.module_batchnorm == 1,
            }
            ee = HeteroModuleNet(**kwargs)
        elif args.model_type == 'SimpleNMN':
            kwargs['use_film'] = args.nmn_use_film
            kwargs['forward_func'] = args.nmn_type
            kwargs['use_color'] = args.use_color,
            ee = SimpleModuleNet(**kwargs)

        elif args.model_type == 'SHNMN':
            kwargs = {
              'vocab' : vocab,
              'feature_dim' : args.feature_dim,
              'stem_dim' : args.stem_dim,
              'module_dim': args.module_dim,
              'module_kernel_size' : args.module_kernel_size,
              'stem_subsample_layers': args.module_stem_subsample_layers,
              'stem_num_layers': args.module_stem_num_layers,
              'stem_kernel_size': args.module_stem_kernel_size,
              'stem_padding': args.module_stem_padding,
              'stem_batchnorm': args.module_stem_batchnorm == 1,
              'classifier_fc_layers': args.classifier_fc_dims,
              'classifier_proj_dim': args.classifier_proj_dim,
              'classifier_downsample': args.classifier_downsample,
              'classifier_batchnorm': args.classifier_batchnorm == 1,
              'classifier_dropout' : args.classifier_dropout,
              'hard_code_alpha' : args.hard_code_alpha,
              'hard_code_tau' : args.hard_code_tau,
              'tau_init' : args.tau_init,
              'alpha_init' : args.alpha_init,
              'which_chain' : args.which_chain,
              'model_type' : args.shnmn_type,
              'model_bernoulli' : args.model_bernoulli,
              'num_modules' : 3,
              'use_module' : args.use_module
            }
            ee = SHNMN(**kwargs)
        elif args.model_type == 'RelNet':
            kwargs['module_num_layers'] = args.module_num_layers
            kwargs['rnn_hidden_dim'] = args.rnn_hidden_dim
            ee = RelationNet(**kwargs)
        elif args.model_type == 'ConvLSTM':
            kwargs['rnn_hidden_dim'] = args.rnn_hidden_dim
            ee = ConvLSTM(**kwargs)
        else:
            kwargs['use_film'] = args.nmn_use_film
            kwargs['use_simple_block'] = args.nmn_use_simple_block
            kwargs['mod_id_loss'] = False
            kwargs['kl_loss'] = False
            kwargs['module_pool'] = args.nmn_module_pool
            kwargs['module_num_layers'] = args.module_num_layers
            kwargs['module_use_gammas'] = args.nmn_use_gammas
            kwargs['learn_control'] = args.nmn_learn_control
            kwargs['rnn_dim'] = args.rnn_hidden_dim
            kwargs['type_anonymizer'] = False
            kwargs['discriminator_proj_dim'] = args.discriminator_proj_dim
            kwargs['discriminator_downsample'] = args.discriminator_downsample
            kwargs['discriminator_fc_layers'] = args.discriminator_fc_dims
            kwargs['discriminator_dropout'] = args.discriminator_dropout
            ee = ModuleNet(**kwargs)
    ee.to(device)
    ee.train()
    if is_multigpu():
        ee = DistributedDataParallel(ee, device_ids=[args.local_rank])
    return ee, kwargs


def get_baseline_model(args):
    vocab = vr.utils.load_vocab(args.vocab_json)
    if args.baseline_start_from is not None:
        model, kwargs = vr.utils.load_baseline(args.baseline_start_from)
    elif args.model_type == 'LSTM':
        kwargs = {
          'vocab': vocab,
          'rnn_wordvec_dim': args.rnn_wordvec_dim,
          'rnn_dim': args.rnn_hidden_dim,
          'rnn_num_layers': args.rnn_num_layers,
          'rnn_dropout': args.rnn_dropout,
          'fc_dims': args.classifier_fc_dims,
          'fc_use_batchnorm': args.classifier_batchnorm == 1,
          'fc_dropout': args.classifier_dropout,
        }
        model = LstmModel(**kwargs)
    elif args.model_type == 'CNN+LSTM':
        kwargs = {
            'vocab': vocab,
          'rnn_wordvec_dim': args.rnn_wordvec_dim,
          'rnn_dim': args.rnn_hidden_dim,
          'rnn_num_layers': args.rnn_num_layers,
          'rnn_dropout': args.rnn_dropout,
          'cnn_feat_dim': args.feature_dim,
          'cnn_num_res_blocks': args.cnn_num_res_blocks,
          'cnn_res_block_dim': args.cnn_res_block_dim,
          'cnn_proj_dim': args.cnn_proj_dim,
          'cnn_pooling': args.cnn_pooling,
          'fc_dims': args.classifier_fc_dims,
          'fc_use_batchnorm': args.classifier_batchnorm == 1,
          'fc_dropout': args.classifier_dropout,
        }
        model = CnnLstmModel(**kwargs)
    elif args.model_type == 'CNN+LSTM+SA':
        kwargs = {
            'vocab': vocab,
          'rnn_wordvec_dim': args.rnn_wordvec_dim,
          'rnn_dim': args.rnn_hidden_dim,
          'rnn_num_layers': args.rnn_num_layers,
          'rnn_dropout': args.rnn_dropout,
          'cnn_feat_dim': args.feature_dim,
          'stacked_attn_dim': args.stacked_attn_dim,
          'num_stacked_attn': args.num_stacked_attn,
          'fc_dims': args.classifier_fc_dims,
          'fc_use_batchnorm': args.classifier_batchnorm == 1,
          'fc_dropout': args.classifier_dropout,
        }
        model = CnnLstmSaModel(**kwargs)
    if model.rnn.token_to_idx != vocab['question_token_to_idx']:
        # Make sure new vocab is superset of old
        for k, v in model.rnn.token_to_idx.items():
            assert k in vocab['question_token_to_idx']
            assert vocab['question_token_to_idx'][k] == v
        for token, idx in vocab['question_token_to_idx'].items():
            model.rnn.token_to_idx[token] = idx
        kwargs['vocab'] = vocab
        model.rnn.expand_vocab(vocab['question_token_to_idx'])
    model.to(device)
    model.train()
    return model, kwargs


def set_mode(mode, models):
    assert mode in ['train', 'eval']
    for m in models:
        if m is None or isinstance(m, ClevrExecutor): continue
        if mode == 'train': m.train()
        if mode == 'eval': m.eval()


def check_accuracy(args, program_generator, execution_engine, baseline_model, loader):
    set_mode('eval', [program_generator, execution_engine, baseline_model])
    num_correct, num_samples = 0, 0
    for batch in loader:
        (questions, _, feats, scenes, answers, programs) = batch
        if isinstance(questions, list):
            questions = questions[0]
            questions = questions[:, :(questions.sum(0) > 0).sum()]

        questions_var = questions.to(device)
        feats_var = feats.to(device)
        if programs[0] is not None:
            programs_var = programs.to(device)

        def scope():
            nonlocal num_samples
            nonlocal num_correct

            scores = None  # Use this for everything but PG
            if args.model_type == 'PG':
                #TODO(mnoukhov) change to scores for attention
                vocab = vr.utils.load_vocab(args.vocab_json)
                programs_pred, _ = program_generator.forward(questions_var)
                for i in range(questions.size(0)):
                    program_pred_str = vr.preprocess.decode(programs_pred[i].tolist(), vocab['program_idx_to_token'])
                    program_str = vr.preprocess.decode(programs[i].tolist(), vocab['program_idx_to_token'])
                    if program_pred_str == program_str:
                        num_correct += 1
                    num_samples += 1
                return
            elif args.model_type in ['EE', 'Hetero']:
                scores, _2, _3 = execution_engine(feats_var, programs_var)
            elif args.model_type == 'PG+EE':
                programs_pred, _ = program_generator.forward(questions_var, argmax=True)
                if isinstance(execution_engine, ClevrExecutor):
                    preds = execution_engine(scenes, programs_pred)
                else:
                    scores, _2, _3 = execution_engine(feats_var, programs_pred)
            elif args.model_type == 'Control-EE':
                questions_repr = program_generator(questions_var)
                scores, _2, _3 = execution_engine(feats_var, programs_var, question=questions_repr)
            elif args.model_type == 'FiLM' or args.model_type == 'RTfilm':
                programs_pred = program_generator(questions_var)
                scores = execution_engine(feats_var, programs_pred)
            elif args.model_type == 'Tfilm':
                programs_pred = program_generator(questions_var)
                scores = execution_engine(feats_var, programs_pred, programs_var)
            elif args.model_type == 'MAC':
                programs_pred = program_generator(questions_var)
                scores = execution_engine(feats_var, programs_pred, isTest=True)
            elif args.model_type in ['ConvLSTM', 'RelNet']:
                question_rep = program_generator(questions_var)
                scores = execution_engine(feats_var, question_rep)
            elif args.model_type in ['LSTM', 'CNN+LSTM', 'CNN+LSTM+SA']:
                scores = baseline_model(questions_var, feats_var)
            elif args.model_type in ['SimpleNMN', 'SHNMN']:
                scores = execution_engine(feats_var, questions_var)
            else:
                raise NotImplementedError('model ', args.model_type, ' check_accuracy not implemented')

            if scores is not None:
                _, preds = scores.data.cpu().max(1)

            num_correct += (preds == answers).sum().item()
            num_samples += preds.size(0)

        # dirty trick to make pytorch free memory earlier
        with torch.no_grad():
            scope()

        if args.num_val_samples is not None and num_samples >= args.num_val_samples:
            break

    set_mode('train', [program_generator, execution_engine, baseline_model])
    acc = float(num_correct) / num_samples
    print("num check samples", num_samples)

    return acc


def check_grad_num_nans(model, model_name='model'):
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    num_nans = [np.sum(np.isnan(grad.data.cpu().numpy())) for grad in grads]
    nan_checks = [num_nan == 0 for num_nan in num_nans]
    if False in nan_checks:
        print('Nans in ' + model_name + ' gradient!')
        print(num_nans)
        pdb.set_trace()
        raise(Exception)


if __name__ == '__main__':
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s: %(asctime)s: %(message)s")
    main(args)

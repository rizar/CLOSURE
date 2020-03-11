# This code is released under the MIT License in association with the following paper:
#
# CLOSURE: Assessing Systematic Generalization of CLEVR Models (https://arxiv.org/abs/1912.05783).
#
# Full copyright and license information (including third party attribution) in the NOTICE file (https://github.com/rizar/CLOSURE/NOTICE).

import argparse
import json
import random
import shutil
from termcolor import colored
import time
from tqdm import tqdm
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
import numpy as np
import h5py
from scipy.misc import imread, imresize, imsave

import vr.utils as utils
import vr.programs
from vr.data import ClevrDataset, ClevrDataLoader
from vr.preprocess import tokenize, encode
from vr.models import *


parser = argparse.ArgumentParser()
parser.add_argument('--program_generator', default=None)
parser.add_argument('--execution_engine', default=None)
parser.add_argument('--debug_every', default=float('inf'), type=float)
parser.add_argument('--use_gpu', default=torch.cuda.is_available(), type=int)

# For running on a preprocessed dataset
parser.add_argument('--data_dir', default=None, type=str)
parser.add_argument('--part', default='val', type=str)

# This will override the vocab stored in the checkpoint;
# we need this to run CLEVR models on human data
parser.add_argument('--vocab_json', default=None)

parser.add_argument('--num_examples', default=None, type=int)

# If this is passed, then save all predictions to this file
parser.add_argument('--output_h5', default=None)
parser.add_argument('--output_preds', default=None)

grads = {}
programs = {}  # NOTE: Useful for zero-shot program manipulation when in debug mode

def main(args):
    input_question_h5 = os.path.join(args.data_dir, '{}_questions.h5'.format(args.part))
    input_features_h5 = os.path.join(args.data_dir, '{}_features.h5'.format(args.part))

    pg, _ = utils.load_program_generator(args.program_generator)

    dtype = torch.FloatTensor
    if args.use_gpu == 1:
        dtype = torch.cuda.FloatTensor

    vocab = load_vocab(args)
    loader_kwargs = {
        'question_h5': input_question_h5,
        'feature_h5': input_features_h5,
        'vocab': vocab,
        'batch_size': 128,
    }
    with ClevrDataLoader(**loader_kwargs) as loader:
        run_batch(args, pg, loader, dtype)


def run_batch(args, pg, loader, dtype):
    pg.type(dtype)
    pg.eval()

    all_correct = []
    all_preds = []
    num_samples = 0
    num_correct = 0

    for batch in tqdm(loader):
        questions, images, feats, answers, programs = batch

        if isinstance(questions, list):
            questions_var = questions[0].type(dtype).long()
        else:
            questions_var = questions.type(dtype).long()
        feats_var = feats.type(dtype)
        programs = programs.to(feats_var.device)

        programs_pred, _ = pg.forward(questions_var, argmax=True)
        min_length = min(programs.shape[1], programs_pred.shape[1])
        programs_pred = programs_pred[:, :min_length]
        programs = programs[:, :min_length]

        correct = (programs_pred == programs).int().sum(1) == min_length
        num_correct += correct.sum()
        all_correct.append(correct)
        all_preds.append(programs_pred)

        num_samples += programs.size(0)
        if args.num_examples and num_samples >= args.num_examples:
            break

    acc = float(num_correct) / num_samples
    print('Got %d / %d = %.2f correct' % (num_correct, num_samples, 100 * acc))

    output_path = ('output_' + args.part + "_" + args.program_generator[:-3] + ".h5"
                   if not args.output_h5
                   else args.output_h5)
    preds_path = ('programs_' + args.part + "_" + args.program_generator[:-3] + ".txt"
                   if not args.output_preds
                   else args.output_preds)

    print('Writing output to "%s"' % output_path)
    with h5py.File(output_path, 'w') as fout:
        fout.create_dataset('correct', data=torch.cat(all_correct, 0).cpu().numpy())

    vocab = load_vocab(args)
    all_preds = torch.cat(all_preds, 0).cpu().numpy()
    all_preds_strings = []
    for i in range(len(all_preds)):
        all_preds_strings.append(
            " ".join(vocab['program_idx_to_token'][w] for w in all_preds[i]))

    save_to_file(all_preds_strings, preds_path)

    if args.debug_every <= 1:
        pdb.set_trace()
    return


def load_vocab(args):
    return utils.load_cpu(args.program_generator)['vocab']


def save_grad(name):
    def hook(grad):
        grads[name] = grad
    return hook


def save_to_file(text, filename):
    with open(filename, mode='wt', encoding='utf-8') as myfile:
        myfile.write('\n'.join(text))
        myfile.write('\n')


def get_index(l, index, default=-1):
    try:
        return l.index(index)
    except ValueError:
        return default


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

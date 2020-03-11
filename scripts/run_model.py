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
from vr.ns_vqa.clevr_executor import ClevrExecutor
from vr.ns_vqa.parser import Seq2seqParser
from vr.preprocess import tokenize, encode
from vr.models import *


parser = argparse.ArgumentParser()
parser.add_argument('--program_generator', default=None)
parser.add_argument('--execution_engine', default=None)
parser.add_argument('--baseline_model', default=None)
parser.add_argument('--debug_every', default=float('inf'), type=float)
parser.add_argument('--use_gpu', default=torch.cuda.is_available(), type=int)

# For running on a preprocessed dataset
parser.add_argument('--data_dir', default=None, type=str)
parser.add_argument('--part', default='val', type=str)

# This will override the vocab stored in the checkpoint;
# we need this to run CLEVR models on human data
parser.add_argument('--vocab_json', default=None)

# For running on a single example
parser.add_argument('--question', default=None)
parser.add_argument('--image', default='img/CLEVR_val_000017.png')
parser.add_argument('--cnn_model', default='resnet101')
parser.add_argument('--cnn_model_stage', default=3, type=int)
parser.add_argument('--image_width', default=224, type=int)
parser.add_argument('--image_height', default=224, type=int)
parser.add_argument('--enforce_clevr_vocab', default=1, type=int)

parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--num_samples', default=None, type=int)
parser.add_argument('--num_last_words_shuffled', default=0, type=int)  # -1 for all shuffled
parser.add_argument('--q_family', type=int, action='append')

parser.add_argument('--sample_argmax', type=int, default=1)
parser.add_argument('--temperature', default=None, type=float)

# FiLM models only
parser.add_argument('--gamma_option', default='linear',
  choices=['linear', 'sigmoid', 'tanh', 'exp', 'relu', 'softplus'])
parser.add_argument('--gamma_scale', default=1, type=float)
parser.add_argument('--gamma_shift', default=0, type=float)
parser.add_argument('--gammas_from', default=None)  # Load gammas from file
parser.add_argument('--beta_option', default='linear',
  choices=['linear', 'sigmoid', 'tanh', 'exp', 'relu', 'softplus'])
parser.add_argument('--beta_scale', default=1, type=float)
parser.add_argument('--beta_shift', default=0, type=float)
parser.add_argument('--betas_from', default=None)  # Load betas from file

# If this is passed, then save all predictions to this file
parser.add_argument('--output_h5', default=None)
parser.add_argument('--dump_module_info', action='store_true')
parser.add_argument('--output_preds', default=None)
parser.add_argument('--output_viz_dir', default='img/')
parser.add_argument('--output_program_stats_dir', default=None)

grads = {}
programs = {}  # NOTE: Useful for zero-shot program manipulation when in debug mode

def main(args):
    if not args.program_generator:
        args.program_generator = args.execution_engine
    input_question_h5 = os.path.join(args.data_dir, '{}_questions.h5'.format(args.part))
    input_features_h5 = os.path.join(args.data_dir, '{}_features.h5'.format(args.part))
    input_scenes = os.path.join(args.data_dir, '{}_scenes.json'.format(args.part))
    vocab = load_vocab(args)

    pg, _ = utils.load_program_generator(args.program_generator)
    if pg:
        pg.save_activations = True
    if args.temperature:
        pg.decoder_linear.weight.data /= args.temperature
        pg.decoder_linear.bias.data /= args.temperature
    if args.execution_engine:
        ee, _ = utils.load_execution_engine(
            args.execution_engine, verbose=False)
        ee.noise_enabled = False
    else:
        ee = ClevrExecutor(vocab)

    dtype = torch.FloatTensor
    if args.use_gpu == 1:
        dtype = torch.cuda.FloatTensor

    loader_kwargs = {
        'question_h5': input_question_h5,
        'feature_h5': input_features_h5,
        'scene_path': input_scenes if isinstance(ee, ClevrExecutor) else None,
        'vocab': vocab,
        'batch_size': args.batch_size,
    }
    if args.num_samples is not None and args.num_samples > 0:
        loader_kwargs['max_samples'] = args.num_samples
    if args.q_family:
        loader_kwargs['question_families'] = args.q_family
    with ClevrDataLoader(**loader_kwargs) as loader:
        with torch.no_grad():
            run_batch(args, pg, ee, loader, dtype)


def run_batch(args, pg, ee, loader, dtype):
    if pg:
        pg.type(dtype)
        pg.eval()
    if ee and not isinstance(ee, ClevrExecutor):
        ee.type(dtype)
        ee.eval()

    all_scores = []
    all_probs = []
    all_preds = []
    all_correct = []

    all_programs = []
    all_groundtruth_programs = []
    all_questions = []
    all_correct_programs = []

    all_seq2seq_attentions = []

    num_samples = 0
    total_nll = 0
    total_prob = 0

    start = time.time()
    for batch in tqdm(loader):
        assert(not pg or not pg.training)
        assert(isinstance(ee, ClevrExecutor) or not ee.training)

        questions, images, feats, scenes, answers, programs = batch
        questions_var = questions[0].type(dtype).long()
        questions_var = questions_var[:, :(questions_var.sum(0) > 0).sum()]
        feats_var = feats.type(dtype)
        programs_var = programs.to(feats_var.device)

        question_repr = None
        programs_pred = None

        # PG
        if isinstance(pg, FiLMGen):
            question_repr = pg(questions_var)
        if isinstance(pg, (Seq2seqParser, Seq2SeqAtt)):
            programs_pred, _ = pg(questions_var, argmax=True)
            all_groundtruth_programs.append(F.pad(programs_var, (0, 30 - programs_var.shape[1], 0, 0)))
            all_programs.append(F.pad(programs_pred, (0, 30 - programs_pred.shape[1], 0, 0)))
            all_questions.append(F.pad(questions_var, (0, 50 - questions_var.shape[1], 0, 0)))
            for _ in range(30 - len(pg._attn_weights)):
                pg._attn_weights.append(torch.zeros_like(pg._attn_weights[0]))
            attn_weights = [F.pad(a, (0, 50 - a.shape[2], 0, 0, 0, 0)) for a in pg._attn_weights]
            all_seq2seq_attentions.append(torch.cat(attn_weights, 1))

            nlls = pg.log_likelihood(questions_var, programs_var)
            total_nll += nlls.sum()
            total_prob += torch.exp(-nlls).sum()
        else:
            programs_pred = programs_var

        # EE
        # arg 1
        if isinstance(ee, ClevrExecutor):
            pos_args = [scenes]
        else:
            pos_args = [feats_var]
        # arg 2
        if isinstance(ee, (ModuleNet, ClevrExecutor)):
            pos_args.append(programs_pred)
        else:
            pos_args.append(question_repr)
        # kwargs
        kwargs = ({'save_activations': True}
                  if isinstance(ee, (FiLMedNet, ModuleNet, MAC))
                  else {})
        if isinstance(ee, ModuleNet) and  ee.learn_control:
            kwargs['question'] = question_repr

        result = ee(*pos_args, **kwargs)

        # unpack outputs
        preds = scores = None
        if isinstance(ee, ModuleNet):
            scores, _2, mod_id_targets = result
        elif isinstance(ee, ClevrExecutor):
            preds = result
        else:
            scores = result

        # compute predictions
        if preds is None:
            probs = F.softmax(scores, dim=1)
            _, preds = scores.data.cpu().max(1)

        all_preds.append(preds.cpu().clone())
        all_correct.append(preds == answers)
        if programs_pred is not None:
            min_length = min(programs_var.shape[1], programs_pred.shape[1])
            programs_pred = programs_pred[:, :min_length]
            programs_var = programs_var[:, :min_length]
            correct_programs = (programs_pred == programs_var).int().sum(1) == min_length
            all_correct_programs.append(correct_programs.cpu().clone())

        if args.dump_module_info:
            all_module_outputs.append(ee.module_outputs.cpu().detach())
            all_mod_id_targets.append(mod_id_targets.cpu().detach())
        num_samples += preds.size(0)

    num_correct = torch.cat(all_correct, 0).sum().item()
    acc = float(num_correct) / num_samples
    print('Got %d / %d = %.2f correct' % (num_correct, num_samples, 100 * acc))

    if all_correct_programs:
        num_correct_programs = torch.cat(all_correct_programs, 0).sum().item()
        prog_acc = float(num_correct_programs) / num_samples
        print('Got %d / %d = %.2f programs correct' % (num_correct_programs, num_samples, 100 * prog_acc))

    if total_nll:
        print("GT program NLL: {}".format(total_nll / num_samples))
        print("Average probability of sampling a GT program: {}".format(total_prob / num_samples))

    print('%.2fs to evaluate' % (start - time.time()))

    model = args.execution_engine if args.execution_engine else args.program_generator
    output_path = ('output_' + args.part + "_" + model.split('.')[0].replace('/', '_') + ".h5"
                   if not args.output_h5
                   else args.output_h5)

    print('Writing output to "%s"' % output_path)
    with h5py.File(output_path, 'w') as fout:
        fout.create_dataset('correct', data=torch.cat(all_correct, 0).numpy())
        if all_scores:
            fout.create_dataset('scores', data=torch.cat(all_scores, 0).numpy())
            fout.create_dataset('probs', data=torch.cat(all_probs, 0).numpy())
        if all_correct_programs:
            fout.create_dataset('correct_programs', data=torch.cat(all_correct_programs, 0).numpy())
        if all_seq2seq_attentions:
            fout.create_dataset('seq2seq_attentions', data=torch.cat(all_seq2seq_attentions, 0).cpu().numpy())
        if all_programs:
            fout.create_dataset('programs', data=torch.cat(all_programs, 0).cpu().numpy())
        if all_groundtruth_programs:
            fout.create_dataset('groundtruth_programs', data=torch.cat(all_groundtruth_programs, 0).cpu().numpy())
        if all_questions:
            fout.create_dataset('questions', data=torch.cat(all_questions, 0).cpu().numpy())

    if args.output_preds is not None:
        all_preds_strings = []
        for i in range(len(all_preds)):
            all_preds_strings.append(vocab['answer_idx_to_token'][all_preds[i]])
        save_to_file(all_preds_strings, args.output_preds)

    if args.debug_every <= 1:
        pdb.set_trace()
    return


def load_vocab(args):
    path = None
    if args.baseline_model is not None:
        path = args.baseline_model
    elif args.program_generator is not None:
        path = args.program_generator
    elif args.execution_engine is not None:
        path = args.execution_engine
    return utils.load_cpu(path)['vocab']


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

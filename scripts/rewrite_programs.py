import h5py
import os
import shutil
import sys
import json
import copy
import numpy
import time
import timeit
from vr.data import ClevrDataset
from vr.utils import load_vocab


def needs_shortcut(token):
    return (token.startswith('filter_') # because "pointers" need not contain attribute info
            or token.startswith('query') # -//-
            or token.startswith('same') # -//-
            or token.startswith('relate')) # -//- because "pointers" need not contain position info


def add_shortcuts(program, vocab):
    token = vocab['program_idx_to_token'][program[0]]
    cur_arity = vocab['program_token_arity'][token]
    if cur_arity == 0:
        return 1, [program[0]]
    if cur_arity == 1:
        shift, subtree = add_shortcuts(program[1:], vocab)
        if needs_shortcut(token):
            return 1 + shift, [program[0], vocab['program_token_to_idx']['scene']] + subtree
        else:
            return 1 + shift, [program[0]] + subtree
    if cur_arity == 2:
        left_shift, left_subtree = add_shortcuts(program[1:], vocab)
        right_shift, right_subtree = add_shortcuts(program[1 + left_shift:], vocab)
        return 1 + left_shift + right_shift, [program[0]] + left_subtree + right_subtree
    raise ValueError()


def rewrite_programs(src_dir, dst_dir):
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    vocab = load_vocab(os.path.join(src_dir, 'vocab.json'))
    old_vocab = copy.deepcopy(vocab)
    arity = vocab['program_token_arity']
    program_vocab = vocab['program_idx_to_token']
    question_vocab = vocab['question_idx_to_token']

    # Step 1: change the arity of filters
    for func in arity.keys():
        if needs_shortcut(func):
            arity[func] = 2
    with open(os.path.join(dst_dir, 'vocab.json'), 'w') as dst:
        json.dump(vocab, dst)

    for part in ['train', 'val', 'test']:
        src_questions = "{}/{}_questions.h5".format(src_dir, part)
        dst_questions = "{}/{}_questions.h5".format(dst_dir, part)
        with h5py.File(src_questions) as src_file:
            programs = src_file['programs']
            prog_wshortcuts = []
            for i in range(len(programs)):
                prog_wshortcuts.append(add_shortcuts(programs[i], old_vocab)[1])
            new_max_program_len = max(len(p) for p in prog_wshortcuts)

            shutil.copyfile(src_questions, dst_questions)
            with h5py.File(dst_questions, 'a') as dst_file:
                del dst_file['programs']
                program_dataset = dst_file.create_dataset(
                    'programs', (len(prog_wshortcuts), new_max_program_len), dtype=numpy.int64)
                for i in range(len(prog_wshortcuts)):
                    program_dataset[i, :len(prog_wshortcuts[i])] = prog_wshortcuts[i]


if __name__ == '__main__':
    src_dir = sys.argv[1]
    dst_dir = sys.argv[2]
    rewrite_programs(src_dir, dst_dir)

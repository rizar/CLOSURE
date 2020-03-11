#!/usr/bin/env python3

# This code is released under the MIT License in association with the following paper:
#
# CLOSURE: Assessing Systematic Generalization of CLEVR Models (https://arxiv.org/abs/1912.05783).
#
# Full copyright and license information (including third party attribution) in the NOTICE file (https://github.com/rizar/CLOSURE/NOTICE).

import inspect
import json
import torch

from vr.models import (ModuleNet,
                       Seq2Seq,
                       Seq2SeqAtt,
                       LstmModel,
                       CnnLstmModel,
                       CnnLstmSaModel,
                       FiLMedNet,
                       FiLMGen,
                       MAC)
from vr.ns_vqa.parser import Seq2seqParser
from vr.ns_vqa.clevr_executor import ClevrExecutor

def invert_dict(d):
    return {v: k for k, v in d.items()}


def load_vocab(path):
    with open(path, 'r') as f:
        vocab = json.load(f)
        vocab['question_idx_to_token'] = invert_dict(vocab['question_token_to_idx'])
        vocab['program_idx_to_token'] = invert_dict(vocab['program_token_to_idx'])
        vocab['answer_idx_to_token'] = invert_dict(vocab['answer_token_to_idx'])
    # Sanity check: make sure <NULL>, <START>, and <END> are consistent
    assert vocab['question_token_to_idx']['<NULL>'] == 0
    assert vocab['question_token_to_idx']['<START>'] == 1
    assert vocab['question_token_to_idx']['<END>'] == 2
    assert vocab['program_token_to_idx']['<NULL>'] == 0
    assert vocab['program_token_to_idx']['<START>'] == 1
    assert vocab['program_token_to_idx']['<END>'] == 2
    return vocab


def load_cpu(path):
    """
    Loads a torch checkpoint, remapping all Tensors to CPU
    """
    return torch.load(path, map_location={'cuda:0': 'cpu'})


def load_program_generator(path):
    checkpoint = load_cpu(path)
    model_type = checkpoint['args']['model_type']
    kwargs = checkpoint['program_generator_kwargs']
    state = checkpoint['program_generator_state']
    if model_type in ['FiLM', 'MAC', 'RelNet', 'Control-EE']:
        model = FiLMGen(**kwargs)
    elif model_type == 'PG+EE' or model_type == 'PG':
        if checkpoint['args']['ns_vqa']:
            model = Seq2seqParser(checkpoint['vocab'])
        else:
            model = Seq2SeqAtt(**kwargs)
    else:
        model = None
    if model is not None:
        model.load_state_dict(state)
    return model, kwargs


def load_execution_engine(path, verbose=True):
    checkpoint = load_cpu(path)
    if checkpoint['args'].get('symbolic_ee'):
        vocab = load_vocab(checkpoint['args']['vocab_json'])
        ee = ClevrExecutor(vocab)
        return ee, {}
    model_type = checkpoint['args']['model_type']
    kwargs = checkpoint['execution_engine_kwargs']
    state = checkpoint['execution_engine_state']
    kwargs['verbose'] = verbose
    if model_type == 'FiLM':
        model = FiLMedNet(**kwargs)
    elif model_type in ['PG+EE', 'EE', 'Control-EE']:
        kwargs.pop('sharing_patterns', None)
        kwargs.setdefault('module_pool', 'mean')
        kwargs.setdefault('module_use_gammas', 'linear')
        model = ModuleNet(**kwargs)
    elif model_type == 'MAC':
        kwargs.setdefault('write_unit', 'original')
        kwargs.setdefault('read_connect', 'last')
        kwargs.setdefault('read_unit', 'original')
        kwargs.setdefault('noisy_controls', False)
        kwargs.pop('sharing_params_patterns', None)
        model = MAC(**kwargs)
    elif model_type == 'RelNet':
        model = RelationNet(**kwargs)
    elif model_type == 'SHNMN':
        model = SHNMN(**kwargs)
    elif model_type == 'SimpleNMN':
        model = SimpleModuleNet(**kwargs)
    else:
        raise ValueError()
    cur_state = model.state_dict()
    model.load_state_dict(state)
    return model, kwargs


def load_baseline(path):
    model_cls_dict = {
      'LSTM': LstmModel,
      'CNN+LSTM': CnnLstmModel,
      'CNN+LSTM+SA': CnnLstmSaModel,
    }
    checkpoint = load_cpu(path)
    baseline_type = checkpoint['baseline_type']
    kwargs = checkpoint['baseline_kwargs']
    state = checkpoint['baseline_state']

    model = model_cls_dict[baseline_type](**kwargs)
    model.load_state_dict(state)
    return model, kwargs


def get_updated_args(kwargs, object_class):
    """
    Returns kwargs with renamed args or arg valuesand deleted, deprecated, unused args.
    Useful for loading older, trained models.
    If using this function is neccessary, use immediately before initializing object.
    """
    # Update arg values
    for arg in arg_value_updates:
        if arg in kwargs and kwargs[arg] in arg_value_updates[arg]:
            kwargs[arg] = arg_value_updates[arg][kwargs[arg]]

    # Delete deprecated, unused args
    valid_args = inspect.getargspec(object_class.__init__)[0]
    new_kwargs = {valid_arg: kwargs[valid_arg] for valid_arg in valid_args if valid_arg in kwargs}
    return new_kwargs

class EMA():
    def __init__(self, mu):
        self.mu = mu
        self.shadow = {}

    def register(self, cat, name, val):
        self.shadow[cat + '-' + name] = val.clone()

    def __call__(self, cat, name, x):
        name = cat + '-' + name
        assert name in self.shadow
        new_average = self.mu * x + (1.0 - self.mu) * self.shadow[name]
        self.shadow[name] = new_average.clone()
        return new_average


def load_tbd_net(checkpoint, vocab):
    """ Convenience function to load a TbD-Net model from a checkpoint file.

    Parameters
    ----------
    checkpoint : Union[pathlib.Path, str]
        The path to the checkpoint.

    vocab : Dict[str, Dict[any, any]]
        The vocabulary file associated with the TbD-Net. For an extended description, see above.

    Returns
    -------
    torch.nn.Module
        The TbD-Net model.

    Notes
    -----
    This pushes the TbD-Net model to the GPU if a GPU is available.
    """
    tbd_net = TbDNet(vocab)
    tbd_net.load_state_dict(torch.load(str(checkpoint), map_location={'cuda:0': 'cpu'}))
    if torch.cuda.is_available():
        tbd_net.cuda()
    return tbd_net

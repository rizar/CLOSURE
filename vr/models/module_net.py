#!/usr/bin/env python3

# This code is released under the MIT License in association with the following paper:
#
# CLOSURE: Assessing Systematic Generalization of CLEVR Models (https://arxiv.org/abs/1912.05783).
#
# Full copyright and license information (including third party attribution) in the NOTICE file (https://github.com/rizar/CLOSURE/NOTICE).

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models

from vr.models.layers import (
    init_modules, ResidualBlock, GlobalAveragePool, Flatten,
    build_classifier, build_stem, ConcatBlock, SimpleConcatBlock)
import vr.programs

from torch.nn.init import kaiming_normal, kaiming_uniform, xavier_uniform, xavier_normal, constant
from torch.autograd import Function
from vr.models.filmed_net import FiLM, FiLMedResBlock, ConcatFiLMedResBlock, coord_map, SharedFiLMedModule, FiLMModule
from vr.models.maced_net import MACControl


class ModuleNet(nn.Module):
    def __init__(self, vocab, feature_dim,
                 use_film,
                 use_simple_block,
                 stem_num_layers,
                 stem_batchnorm,
                 stem_subsample_layers,
                 stem_kernel_size,
                 stem_stride,
                 stem_padding,
                 stem_dim,
                 module_dim,
                 module_pool,
                 module_use_gammas,
                 module_kernel_size,
                 module_input_proj,
                 module_residual=True,
                 module_batchnorm=False,
                 module_num_layers=1,
                 mod_id_loss=False,
                 kl_loss=False,
                 learn_control=False,
                 rnn_dim=None,
                 classifier_proj_dim=512,
                 classifier_downsample='maxpool2',
                 classifier_fc_layers=(1024,),
                 classifier_batchnorm=False,
                 classifier_dropout=0,
                 discriminator_proj_dim=None,
                 discriminator_downsample=None,
                 discriminator_fc_layers=None,
                 discriminator_dropout=None,
                 verbose=True,
                 type_anonymizer=False):
        super(ModuleNet, self).__init__()

        if discriminator_proj_dim is None:
            discriminator_proj_dim = classifier_proj_dim
        if discriminator_downsample is None:
            discriminator_downsample = classifier_downsample
        if discriminator_fc_layers is None:
            discriminator_fc_layers = classifier_fc_layers
        if discriminator_dropout is None:
            discriminator_dropout = classifier_dropout

        self.module_dim = module_dim
        self.use_film = use_film
        self.use_simple_block = use_simple_block
        self.mod_id_loss = mod_id_loss
        self.kl_loss = kl_loss
        self.learn_control = learn_control

        self.stem = build_stem(feature_dim[0], stem_dim, module_dim,
                               num_layers=stem_num_layers,
                               subsample_layers=stem_subsample_layers,
                               kernel_size=stem_kernel_size,
                               padding=stem_padding,
                               with_batchnorm=stem_batchnorm)
        tmp = self.stem(Variable(torch.zeros([1, feature_dim[0], feature_dim[1], feature_dim[2]])))
        module_H = tmp.size(2)
        module_W = tmp.size(3)

        self.coords = coord_map((module_H, module_W))

        if verbose:
            print('Here is my stem:')
            print(self.stem)

        classifier_kwargs = dict(module_C=module_dim, module_H=module_H, module_W=module_W,
                                 num_answers=len(vocab['answer_idx_to_token']),
                                 fc_dims=classifier_fc_layers,
                                 proj_dim=classifier_proj_dim,
                                 downsample=classifier_downsample,
                                 with_batchnorm=classifier_batchnorm,
                                 dropout=classifier_dropout)
        discriminator_kwargs = dict(module_C=module_dim, module_H=module_H, module_W=module_W,
                                    num_answers=len(vocab['program_idx_to_token']),
                                    fc_dims=discriminator_fc_layers,
                                    proj_dim=discriminator_proj_dim,
                                    downsample=discriminator_downsample,
                                    with_batchnorm=False,
                                    dropout=discriminator_dropout)
        if self.use_film:
            classifier_kwargs['module_H'] = 1
            classifier_kwargs['module_W'] = 1
            discriminator_kwargs['module_H'] = 1
            discriminator_kwargs['module_W'] = 1

        self.classifier = build_classifier(**classifier_kwargs)
        if self.mod_id_loss:
            self.module_identifier = build_classifier(**discriminator_kwargs)

        if verbose:
            print('Here is my classifier:')
            print(self.classifier)

        self.function_modules = {}
        self.function_modules_num_inputs = {}
        self.vocab = vocab

        shared_block = None
        if type_anonymizer:
            shared_block = ResidualBlock(module_dim,
                                         kernel_size=module_kernel_size,
                                         with_residual=module_residual,
                                         with_batchnorm=module_batchnorm)
        elif use_film == 1:
            assert module_W == module_H
            shared_block = SharedFiLMedModule(module_dim,
                                              kernel_size=module_kernel_size,
                                              num_layers=module_num_layers,
                                              with_residual=module_residual,
                                              pool=module_pool,
                                              use_gammas=module_use_gammas,
                                              post_linear=kl_loss,
                                              learn_embeddings=not learn_control)
        if shared_block:
            self.shared_block = shared_block
            self.add_module('shared', shared_block)

        for fn_str, fn_idx in vocab['program_token_to_idx'].items():
            num_inputs = vocab['program_token_arity'][fn_str]
            self.function_modules_num_inputs[fn_str] = num_inputs

            def create_module():
                if num_inputs > 2:
                    raise Exception('Not implemented!')

                if use_film == 1:
                    return FiLMModule(shared_block, fn_idx)

                if use_film == 2:
                    separate_core_block = SharedFiLMedModule(module_dim, module_W,
                                                             kernel_size=module_kernel_size,
                                                             with_residual=module_residual)
                    return FiLMModule(separate_core_block, fn_idx)

                if use_simple_block:
                    # brutally simple concatentation block
                    # with 2 layers, no residual connection
                    return SimpleConcatBlock(
                        module_dim,
                        kernel_size=module_kernel_size)

                if num_inputs in [0, 1]:
                    return ResidualBlock(
                            module_dim,
                            kernel_size=module_kernel_size,
                            with_residual=module_residual,
                            with_batchnorm=module_batchnorm,
                            shared_block=shared_block,
                            post_linear=kl_loss)
                else:
                    return ConcatBlock(
                            module_dim,
                            kernel_size=module_kernel_size,
                            with_residual=module_residual,
                            with_batchnorm=module_batchnorm,
                            shared_block=shared_block,
                            post_linear=kl_loss)

            mod = create_module()
            if mod is not None:
                self.add_module(fn_str, mod)
                self.function_modules[fn_str] = mod

        self.save_module_outputs = False
        self.noise_enabled = True

        if learn_control:
            self.controller = MACControl(30, rnn_dim, module_dim)

    def _forward_modules_ints_helper(self, feats, program, i, j, module_outputs):
        used_fn_j = True
        orig_j = j
        if j < program.size(1):
            fn_idx = program.data[i, j]
            fn_str = self.vocab['program_idx_to_token'][fn_idx.item()]
        else:
            used_fn_j = False
            fn_str = 'scene'
        if fn_str == '<NULL>':
            used_fn_j = False
            fn_str = 'scene'
        elif fn_str == '<START>':
            used_fn_j = False
            return self._forward_modules_ints_helper(feats, program, i, j + 1, module_outputs)
        if used_fn_j:
            self.used_fns[i, j] = 1
        j += 1

        num_inputs = self.function_modules_num_inputs[fn_str]
        if fn_str == 'scene':
            num_inputs = 1

        module = self.function_modules[fn_str]

        if fn_str == 'scene':
            module_inputs = [feats[i:i+1]]
        else:
            module_inputs = []
            while len(module_inputs) < num_inputs:
                cur_input, j = self._forward_modules_ints_helper(feats, program, i, j, module_outputs)
                module_inputs.append(cur_input)
            if self.use_film:
                module_inputs = [feats[i:i+1]] + module_inputs

        if self.use_simple_block:
            # simple block must have 3 inputs
            if len(module_inputs) < 2:
                module_inputs.append(torch.zeros_like(module_inputs[0]))
            module_inputs = [feats[i:i+1]] + module_inputs

        module_output = module(*module_inputs)
        if self.kl_loss:
            mu = module_output[:, :self.module_dim]
            logvar = module_output[:, self.module_dim:] - 5
            logvar = torch.min(logvar, torch.ones_like(logvar))
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(logvar) if self.noise_enabled else 0
            module_output = mu + std * eps
            self._mus.append(mu)
            self._logvars.append(logvar)
        # a module is uniquely identified by an (i, orig_j)
        if used_fn_j:
            module_outputs[(i, orig_j)] = module_output
        return module_output, j

    def _forward_modules_ints(self, feats, program):
        """
        feats: FloatTensor of shape (N, C, H, W) giving features for each image
        program: LongTensor of shape (N, L) giving a prefix-encoded program for
          each image.
        """
        N = feats.size(0)
        final_module_outputs = []
        self.used_fns = torch.Tensor(program.size()).fill_(0)

        module_outputs = {}

        for i in range(N):
            cur_output, _ = self._forward_modules_ints_helper(feats, program, i, 0, module_outputs)
            final_module_outputs.append(cur_output)
        final_module_outputs = torch.cat(final_module_outputs, 0)
        self.used_fns = self.used_fns.type_as(program.data).float()


        return final_module_outputs, module_outputs

    def _forward_batch(self, feats, program, question, save_activations=False):
        cur = None
        batch_size = program.shape[0]
        max_program_len = program.shape[1]
        stacks = [[] for j in range(batch_size)]
        program_wellformed = torch.ones(batch_size, dtype=torch.bool)

        zero_inp = torch.zeros_like(feats)[:, :, 0, 0]
        memory = zero_inp[None, :]

        if question is not None:
            controls, control_scores = self.controller(question)
            assert max_program_len <= controls.shape[1]
            lengths = (program > 0).sum(1)
            new_controls = []
            for j, leng in zip(range(batch_size), lengths):
                 #shift controls so that the last control goes to the first module
                 new_controls.append(
                    torch.cat([controls[j, -leng:],
                               torch.zeros((max_program_len - leng, controls.shape[2]),
                                           device=controls.device)],
                              0))
            controls = torch.cat([c[None, :] for c in new_controls], 0)

        # skip <START> at the position 0
        for i in reversed(range(1, max_program_len)):
            fn_names = [self.vocab['program_idx_to_token'][program[j, i].item()]
                        for j in range(batch_size)]
            mask = torch.ones_like(program[:, 0])
            for j in range(batch_size):
                if fn_names[j] in ['<END>', '<NULL>']:
                    mask[j] = 0
            num_inputs = [self.function_modules_num_inputs[fn_name] if mask[j] else 0
                          for j, fn_name in enumerate(fn_names)]

            # prepare inputs
            input_indices = [[max_program_len, max_program_len] for j in range(batch_size)]
            for j in range(batch_size):
                for k in range(num_inputs[j]):
                    if stacks[j]:
                        input_indices[j][k] = stacks[j].pop()
                    else:
                        program_wellformed[j] = False
            inputs = []
            for k in range(2):
                indices = [input_indices[j][k] - i - 1 for j in range(batch_size)]
                inputs.append(memory[indices, range(batch_size)])

            # run the batched compute
            control_i = controls[:, i] if question else program[:, i]
            cur = self.shared_block(feats, control_i, inputs[0], inputs[1])

            memory = torch.cat([cur[None, :], memory])

            # push the new results onto the stack
            for j in range(batch_size):
                if mask[j]:
                    stacks[j].append(i)

        for j in range(batch_size):
            if len(stacks[j]) != 1:
                program_wellformed[j] = False

        if save_activations and self.learn_control:
            self.control_scores = control_scores

        return cur, program_wellformed

    def forward(self, x, program, save_activations=False, question=None):
        N = x.size(0)
        assert N == len(program)

        feats = self.stem(x)

        program_wellformed = None
        if self.use_film == 1:
            final_module_outputs, program_wellformed = self._forward_batch(
                feats, program, question=question if self.learn_control else None,
                save_activations=save_activations)
            #check = self._forward_modules_ints(feats, program)
            #print(abs(final_module_outputs - check[0]).sum())
        else:
            final_module_outputs, _ = self._forward_modules_ints(feats, program)

        scores = self.classifier(final_module_outputs)
        return scores, program_wellformed, None

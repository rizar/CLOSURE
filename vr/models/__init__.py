#!/usr/bin/env python3

# This code is released under the MIT License in association with the following paper:
#
# CLOSURE: Assessing Systematic Generalization of CLEVR Models (https://arxiv.org/abs/1912.05783).
#
# Full copyright and license information (including third party attribution) in the NOTICE file (https://github.com/rizar/CLOSURE/NOTICE).

from vr.models.module_net import ModuleNet
from vr.models.filmed_net import FiLMedNet
from vr.models.seq2seq import Seq2Seq
from vr.models.seq2seq_att import Seq2SeqAtt
from vr.models.film_gen import FiLMGen
from vr.models.maced_net import MAC
from vr.models.baselines import LstmModel, CnnLstmModel, CnnLstmSaModel
from vr.models.convlstm import ConvLSTM

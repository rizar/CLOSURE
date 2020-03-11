import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from .base_rnn import BaseRNN
from .attention import Attention


def logical_or(x, y):
    return (x + y).clamp_(0, 1)

def logical_not(x):
    return x == 0


class Decoder(BaseRNN):
    """Decoder RNN module
    To do: add docstring to methods
    """

    def __init__(self, vocab_size, max_len, word_vec_dim, hidden_size,
                 n_layers, start_id=1, end_id=2, rnn_cell='lstm',
                 bidirectional=False, input_dropout_p=0,
                 dropout_p=0, use_attention=False):
        super(Decoder, self).__init__(vocab_size, max_len, hidden_size,
                                      input_dropout_p, dropout_p, n_layers, rnn_cell)

        self.max_length = max_len
        self.output_size = vocab_size
        self.hidden_size = hidden_size
        self.word_vec_dim = word_vec_dim
        self.bidirectional_encoder = bidirectional
        if bidirectional:
            self.hidden_size *= 2
        self.use_attention = use_attention
        self.start_id = start_id
        self.end_id = end_id

        self.embedding = nn.Embedding(self.output_size, self.word_vec_dim)
        self.rnn = self.rnn_cell(self.word_vec_dim, self.hidden_size, n_layers, batch_first=True, dropout=dropout_p)
        self.out_linear = nn.Linear(self.hidden_size, self.output_size)
        if use_attention:
            self.attention = Attention(self.hidden_size)

    def forward_step(self, input_var, hidden, encoder_outputs):
        batch_size = input_var.size(0)
        output_size = input_var.size(1)
        embedded = self.embedding(input_var)
        embedded = self.input_dropout(embedded)
        output, hidden = self.rnn(embedded, hidden)

        attn = None
        if self.use_attention:
            output, attn = self.attention(output, encoder_outputs)

        output = self.out_linear(output.contiguous().view(-1, self.hidden_size))
        predicted_softmax = F.log_softmax(output.view(batch_size, output_size, -1), 2)
        return predicted_softmax, hidden, attn

    def forward(self, y, encoder_outputs, encoder_hidden):
        decoder_hidden = self._init_state(encoder_hidden)
        decoder_outputs, decoder_hidden, attn = self.forward_step(y, decoder_hidden, encoder_outputs)
        return decoder_outputs, decoder_hidden

    def forward_sample(self, encoder_outputs, encoder_hidden, reinforce_sample=False):
        if isinstance(encoder_hidden, tuple):
            batch_size = encoder_hidden[0].size(1)
        else:
            batch_size = encoder_hidden.size(1)
        decoder_hidden = self._init_state(encoder_hidden)
        decoder_input = Variable(torch.LongTensor(batch_size, 1).fill_(self.start_id))
        decoder_input = decoder_input.to(encoder_hidden[0].device)

        output_symbols = [decoder_input.squeeze()]
        output_logprobs = [torch.zeros(batch_size).to(decoder_input.device)]
        done = torch.ByteTensor(batch_size).fill_(0).to(decoder_input.device)

        def decode(i, output, reinforce_sample=reinforce_sample):
            nonlocal done
            if reinforce_sample:
                dist = torch.distributions.Categorical(probs=torch.exp(output.view(batch_size, -1))) # better initialize with logits
                symbols = dist.sample().unsqueeze(1)
            else:
                symbols = output.topk(1)[1].view(batch_size, -1)
            symbol_logprobs = output[:, 0, :][torch.arange(batch_size), symbols[:, 0]]
            not_done = logical_not(done)
            output_logprobs.append(not_done.float() * symbol_logprobs)
            output_symbols.append(symbols.squeeze())
            done = logical_or(done, symbols[:, 0] == self.end_id)
            return symbols

        for i in range(self.max_length):
            decoder_output, decoder_hidden, step_attn = self.forward_step(decoder_input, decoder_hidden, encoder_outputs)
            decoder_input = decode(i, decoder_output)

        return output_symbols, output_logprobs

    def _init_state(self, encoder_hidden):
        if encoder_hidden is None:
            return None
        if isinstance(encoder_hidden, tuple):
            encoder_hidden = tuple([self._cat_directions(h) for h in encoder_hidden])
        else:
            encoder_hidden = self._cat_directions(encoder_hidden)
        return encoder_hidden

    def _cat_directions(self, h):
        if self.bidirectional_encoder:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h

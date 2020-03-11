#!/usr/bin/env python3

# This code is released under the MIT License in association with the following paper:
#
# CLOSURE: Assessing Systematic Generalization of CLEVR Models (https://arxiv.org/abs/1912.05783).
#
# Full copyright and license information (including third party attribution) in the NOTICE file (https://github.com/rizar/CLOSURE/NOTICE).
import math

import torch
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import (pack_padded_sequence,
                                pad_packed_sequence)


class Attn(nn.Module):
    def __init__(self, hidden_size):
        super(Attn, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 3, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)

    def forward(self, output, encoder_outputs, encoder_mask):
        seq_len = encoder_outputs.size(1)
        keys = output.repeat(seq_len, 1, 1).transpose(0,1)
        attn_energies = self.score(keys, encoder_outputs) # B*1*T
        attn_energies -= 1000 * (encoder_mask[:, None, :] == 0).float()
        return F.softmax(attn_energies, dim=2)

    def score(self, hidden, encoder_outputs):
        energy = torch.tanh(self.attn(torch.cat([hidden, encoder_outputs], 2))) # [B*T*2H]->[B*T*H]
        energy = energy.transpose(2,1) # [B*H*T]
        v = self.v.repeat(encoder_outputs.data.shape[0],1).unsqueeze(1) #[B*1*H]
        energy = torch.bmm(v, energy) # [B*1*T]
        return energy


class Seq2SeqAtt(nn.Module):
    def __init__(self,
      null_token=0,
      start_token=1,
      end_token=2,
      encoder_vocab_size=100,
      decoder_vocab_size=100,
      wordvec_dim=300,
      hidden_dim=256,
      rnn_num_layers=2,
      rnn_dropout=0,
      autoregressive=True,
    ):
        super().__init__()
        self.encoder_embed = nn.Embedding(encoder_vocab_size, wordvec_dim)
        self.encoder_rnn = nn.LSTM(wordvec_dim, hidden_dim, rnn_num_layers,
                                   dropout=rnn_dropout, bidirectional=True, batch_first=True)
        self.decoder_embed = nn.Embedding(decoder_vocab_size, wordvec_dim)
        self.decoder_rnn = nn.LSTM(wordvec_dim + 2 * hidden_dim, hidden_dim, rnn_num_layers,
                                   dropout=rnn_dropout, batch_first=True)
        self.decoder_linear = nn.Linear(3 * hidden_dim, decoder_vocab_size)
        self.decoder_attn = Attn(hidden_dim)
        self.rnn_num_layers = rnn_num_layers
        self.NULL = null_token
        self.START = start_token
        self.END = end_token
        self.multinomial_outputs = None
        self.autoregressive = autoregressive

        self.save_activations = False

    def expand_encoder_vocab(self, token_to_idx, word2vec=None, std=0.01):
        expand_embedding_vocab(self.encoder_embed, token_to_idx,
                               word2vec=word2vec, std=std)

    def get_dims(self, x=None, y=None):
        V_in = self.encoder_embed.num_embeddings
        V_out = self.decoder_embed.num_embeddings
        D = self.encoder_embed.embedding_dim
        H = self.encoder_rnn.hidden_size
        L = self.encoder_rnn.num_layers

        N = x.size(0) if x is not None else None
        N = y.size(0) if N is None and y is not None else N
        T_in = x.size(1) if x is not None else None
        T_out = y.size(1) if y is not None else None
        return V_in, V_out, D, H, L, N, T_in, T_out

    def encoder(self, x):
        x, x_lengths, inverse_index = sort_for_rnn(x, null=self.NULL)
        embed = self.encoder_embed(x)
        packed = pack_padded_sequence(embed, x_lengths, batch_first=True)
        out_packed, hidden = self.encoder_rnn(packed)
        out, _ = pad_packed_sequence(out_packed, batch_first=True)

        out = out[inverse_index]
        hidden = [h[:,inverse_index] for h in hidden]

        return out, hidden

    def decoder(self, word_inputs, prev_hidden, encoder_outputs, encoder_mask):
        hn, cn, an = prev_hidden

        # 1 - rnn transition
        word_embedded = self.decoder_embed(word_inputs)
        if not self.autoregressive:
            word_embedded = torch.zeros_like(word_embedded)
        rnn_input = torch.cat((word_embedded, an), 1)[:, None, :]
        output, (hnext, cnext) = self.decoder_rnn(rnn_input, (hn, cn))
        output = output[:, 0, :]

        # 2 - perform attention
        attn_weights = self.decoder_attn(output, encoder_outputs, encoder_mask)
        anext = attn_weights.bmm(encoder_outputs)[:, 0, :]
        if self.save_activations:
            self._attn_weights.append(attn_weights)

        # 3 - compute output logits
        logits = self.decoder_linear(torch.cat([output, anext], 1))

        return logits, (hnext, cnext, anext)

    def compute_loss(self, output_logprobs, y):
        """
        Compute loss. We assume that the first element of the output sequence y is
        a start token, and that each element of y is left-aligned and right-padded
        with self.NULL out to T_out. We want the output_logprobs to predict the
        sequence y, shifted by one timestep so that y[0] is fed to the network and
        then y[1] is predicted. We also don't want to compute loss for padded
        timesteps.

        Inputs:
        - output_logprobs: Variable of shape (N, T_out, V_out)
        - y: LongTensor Variable of shape (N, T_out)
        """
        V_in, V_out, D, H, L, N, T_in, T_out = self.get_dims(y=y)

        output_logprobs = output_logprobs[:, :-1].contiguous()
        y = y[:, 1:].contiguous()
        losses = F.cross_entropy(output_logprobs.view(-1, V_out), y.view(-1), reduction='none')
        losses = losses.view(N, T_out - 1)
        losses *= (y != self.NULL).float()
        return losses.sum(1)

    def log_likelihood(self, x, y):
        V_in, V_out, D, H, L, N, T_in, T_out = self.get_dims(y=y)

        encoder_outputs, _ = self.encoder(x)
        encoder_mask = x != self.NULL

        decoder_inputs = y
        decoder_hidden = (torch.zeros(L, N, H).to(x.device),
                          torch.zeros(L, N, H).to(x.device),
                          torch.zeros(N, 2 * H).to(x.device)) # attention state
        decoder_outputs = []
        for t in range(T_out):
            decoder_out, decoder_hidden = self.decoder(
                decoder_inputs[:,t], decoder_hidden,
                encoder_outputs, encoder_mask)
            decoder_outputs.append(decoder_out)

        decoder_outputs = torch.stack(decoder_outputs, dim=1)
        loss = self.compute_loss(decoder_outputs, y)
        return loss

    def forward(self, x, max_length=30, temperature=1.0, argmax=False):
        self._attn_weights = []

        V_in, V_out, D, H, L, N, T_in, T_out = self.get_dims(x=x)
        T_out = max_length

        encoded, _ = self.encoder(x)
        encoder_mask = x != self.NULL
        h, c, a = (torch.zeros(L, N, H).to(x.device), # hidden state
                   torch.zeros(L, N, H).to(x.device), # cell state
                   torch.zeros(N, 2 * H).to(x.device)) # attention state

        # buffers (on CPU currently)
        cur_input = Variable(x.data.new(N).fill_(self.START))
        y = torch.LongTensor(N, T_out).fill_(self.NULL)
        y[:, 0] = cur_input
        y_logprobs = torch.zeros((N, T_out))
        done = torch.ByteTensor(N).fill_(0)

        for t in range(1, T_out):
            # generate output
            logprobs, (h, c, a) = self.decoder(cur_input, (h, c, a), encoded, encoder_mask)
            logprobs = logprobs / temperature
            logprobs = F.log_softmax(logprobs, dim=1)
            if argmax:
                _, cur_output = logprobs.max(1)
            else:
                cur_output = torch.exp(logprobs).multinomial(1)[:, 0]

            # save output
            cur_output_data = cur_output.data.cpu()
            not_done = logical_not(done)
            y[not_done, t] = cur_output_data[not_done]
            y_logprobs[:, t] = logprobs[torch.arange(N), cur_output]
            done = logical_or(done, (cur_output_data == self.END).byte())
            cur_input = cur_output

            # stop if fully done
            if done.sum() == N:
                break
        return y.to(x.device), y_logprobs.to(x.device)

def logical_or(x, y):
    return (x + y).clamp_(0, 1)

def logical_not(x):
    return x == 0

def sort_for_rnn(x, null=0):
    lengths = torch.sum(x != null, dim=1).long()
    sorted_lengths, sorted_idx = torch.sort(lengths, dim=0, descending=True)
    sorted_lengths = sorted_lengths.data.tolist() # remove for pytorch 0.4+
    # ugly
    inverse_sorted_idx = torch.LongTensor(sorted_idx.shape).fill_(0).to(x.device)
    for i, v in enumerate(sorted_idx):
        inverse_sorted_idx[v.data] = i

    return x[sorted_idx], sorted_lengths, inverse_sorted_idx

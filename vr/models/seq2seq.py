#!/usr/bin/env python3

# This code is released under the MIT License in association with the following paper:
#
# CLOSURE: Assessing Systematic Generalization of CLEVR Models (https://arxiv.org/abs/1912.05783).
#
# Full copyright and license information (including third party attribution) in the NOTICE file (https://github.com/rizar/CLOSURE/NOTICE).

import torch
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Seq2Seq(nn.Module):
    def __init__(self,
      encoder_vocab_size=100,
      decoder_vocab_size=100,
      wordvec_dim=300,
      hidden_dim=256,
      rnn_num_layers=2,
      rnn_dropout=0,
      null_token=0,
      start_token=1,
      end_token=2,
      encoder_embed=None
    ):
        super(Seq2Seq, self).__init__()
        self.encoder_embed = nn.Embedding(encoder_vocab_size, wordvec_dim)
        self.encoder_rnn = nn.LSTM(wordvec_dim, hidden_dim, rnn_num_layers,
                                   dropout=rnn_dropout, batch_first=True)
        self.decoder_embed = nn.Embedding(decoder_vocab_size, wordvec_dim)
        self.decoder_rnn = nn.LSTM(wordvec_dim + hidden_dim, hidden_dim, rnn_num_layers,
                                   dropout=rnn_dropout, batch_first=True)
        self.decoder_rnn_new = nn.LSTM(hidden_dim, hidden_dim, rnn_num_layers,
                                       dropout=rnn_dropout, batch_first=True)
        self.decoder_linear = nn.Linear(hidden_dim, decoder_vocab_size)
        self.NULL = null_token
        self.START = start_token
        self.END = end_token
        self.multinomial_outputs = None

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

    def before_rnn(self, x, replace=0):
        # TODO: Use PackedSequence instead of manually plucking out the last
        # non-NULL entry of each sequence; it is cleaner and more efficient.
        N, T = x.size()
        idx = torch.LongTensor(N).fill_(T - 1)

        # Find the last non-null element in each sequence. Is there a clean
        # way to do this?
        x_cpu = x.cpu()
        for i in range(N):
            for t in range(T - 1):
                if x_cpu.data[i, t] != self.NULL and x_cpu.data[i, t + 1] == self.NULL:
                    idx[i] = t
                    break
        idx = idx.type_as(x.data)
        x[x.data == self.NULL] = replace
        return x, Variable(idx)

    def encoder(self, x):
        V_in, V_out, D, H, L, N, T_in, T_out = self.get_dims(x=x)
        x, idx = self.before_rnn(x)
        embed = self.encoder_embed(x)
        h0 = Variable(torch.zeros(L, N, H).type_as(embed.data))
        c0 = Variable(torch.zeros(L, N, H).type_as(embed.data))

        out, _ = self.encoder_rnn(embed, (h0, c0))

        # Pull out the hidden state for the last non-null value in each input
        idx = idx.view(N, 1, 1).expand(N, 1, H)
        return out.gather(1, idx).view(N, H)

    def decoder(self, encoded, y, h0=None, c0=None):
        V_in, V_out, D, H, L, N, T_in, T_out = self.get_dims(y=y)

        if T_out > 1:
            y, _ = self.before_rnn(y)
        y_embed = self.decoder_embed(y)
        encoded_repeat = encoded.view(N, 1, H).expand(N, T_out, H)
        rnn_input = torch.cat([encoded_repeat, y_embed], 2)
        if h0 is None:
            h0 = Variable(torch.zeros(L, N, H).type_as(encoded.data))
        if c0 is None:
            c0 = Variable(torch.zeros(L, N, H).type_as(encoded.data))
        rnn_output, (ht, ct) = self.decoder_rnn(rnn_input, (h0, c0))

        rnn_output_2d = rnn_output.contiguous().view(N * T_out, H)
        output_logprobs = self.decoder_linear(rnn_output_2d).view(N, T_out, V_out)

        return output_logprobs, ht, ct

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
        self.multinomial_outputs = None
        V_in, V_out, D, H, L, N, T_in, T_out = self.get_dims(y=y)
        mask = y.data != self.NULL
        y_mask = Variable(torch.Tensor(N, T_out).fill_(0).type_as(mask))
        y_mask[:, 1:] = mask[:, 1:]
        y_masked = y[y_mask]
        out_mask = Variable(torch.Tensor(N, T_out).fill_(0).type_as(mask))
        out_mask[:, :-1] = mask[:, 1:]
        out_mask = out_mask.view(N, T_out, 1).expand(N, T_out, V_out)
        out_masked = output_logprobs[out_mask].view(-1, V_out)
        loss = F.cross_entropy(out_masked, y_masked)
        return loss

    def forward(self, x, y):
        encoded = self.encoder(x)

        V_in, V_out, D, H, L, N, T_in, T_out = self.get_dims(x=x)
        T_out = 15
        encoded_repeat = encoded.view(N, 1, H).expand(N, T_out, H)
        h0 = Variable(torch.zeros(L, N, H).type_as(encoded.data))
        c0 = Variable(torch.zeros(L, N, H).type_as(encoded.data))
        rnn_output, (ht, ct) = self.decoder_rnn_new(encoded_repeat, (h0, c0))

        output_logprobs, _, _ = self.decoder(encoded, y)
        loss = self.compute_loss(output_logprobs, y)
        return loss

    def reinforce_sample(self, x, max_length=30, temperature=1.0, argmax=False):
        N, T = x.size(0), max_length
        encoded = self.encoder(x)
        y = torch.LongTensor(N, T).fill_(self.NULL)
        y_logprobs = torch.Tensor(N, T).fill_(0.)
        done = torch.ByteTensor(N).fill_(0)
        cur_input = Variable(x.data.new(N, 1).fill_(self.START))
        h, c = None, None
        for t in range(T):
            # generate output
            logprobs, h, c = self.decoder(encoded, cur_input, h0=h, c0=c)
            logprobs = logprobs[:, 0, :]
            logprobs = logprobs / temperature
            logprobs = F.log_softmax(logprobs, dim=1)
            if argmax:
                _, cur_output = logprobs.max(1, keepdim=True)
            else:
                cur_output = torch.exp(logprobs).multinomial(1) # Now N x 1

            # save output
            cur_output_data = cur_output.data.cpu()
            not_done = logical_not(done)
            y[not_done, t] = cur_output_data[not_done, 0]
            y_logprobs[not_done, t] = logprobs[not_done, cur_output[:, 0]]
            done = logical_or(done, cur_output_data[:, 0] == self.END)
            cur_input = cur_output

            # stop if fully done
            if done.sum() == N:
                break
        return y.type_as(x.data), y_logprobs.to(x.device)


def logical_and(x, y):
    return x * y

def logical_or(x, y):
    return (x + y).clamp_(0, 1)

def logical_not(x):
    return x == 0

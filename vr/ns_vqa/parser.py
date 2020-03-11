import torch
import torch.nn as nn
from torch.autograd import Variable

from . import create_seq2seq_net, TrainOptions


class Seq2seqParser(nn.Module):
    """Model interface for seq2seq parser"""

    def __init__(self, vocab):
        super().__init__()
        self.opt = TrainOptions().parse()
        self.vocab = vocab
        self.net_params = self._get_net_params(self.opt, self.vocab)
        self.seq2seq = create_seq2seq_net(**self.net_params)
        self.variable_lengths = self.net_params['variable_lengths']
        self.end_id = self.net_params['end_id']
        #self.gpu_ids = opt.gpu_ids
        self.criterion = nn.NLLLoss()

    def set_input(self, x, y=None):
        input_lengths, idx_sorted = None, None
        if self.variable_lengths:
            x, y, input_lengths, idx_sorted = self._sort_batch(x, y)
        #self.x = self._to_var(x)
        #if y is not None:
            #self.y = self._to_var(y)
        #else:
            #self.y = None
        self.x = x
        self.y = y
        self.input_lengths = input_lengths
        self.idx_sorted = idx_sorted

    def log_likelihood(self, x, y):
        self.set_input(x, y)
        assert self.y is not None, 'Must set y value'
        output_logprob = self.seq2seq(self.x, self.y, self.input_lengths)
        loss = self.criterion(output_logprob[:,:-1,:].contiguous().view(-1, output_logprob.size(2)),
                                   self.y[:,1:].contiguous().view(-1))
        return loss

    def forward(self, x, argmax=False):
        self.set_input(x)
        rl_seq, logprobs = self.seq2seq.reinforce_forward(self.x, self.input_lengths, argmax=argmax)
        rl_seq = self._restore_order(rl_seq.data.cpu())
        logprobs = self._restore_order(logprobs)
        self.reward = None # Need to recompute reward from environment each time a new sequence is sampled
        return rl_seq.to(x.device), logprobs

    def reinforce_backward(self, entropy_factor=0.0):
        assert self.reward is not None, 'Must run forward sampling and set reward before REINFORCE'
        self.seq2seq.reinforce_backward(self.reward, entropy_factor)

    def parse(self):
        output_sequence = self.seq2seq.sample_output(self.x, self.input_lengths)
        output_sequence = self._restore_order(output_sequence.data.cpu())
        return output_sequence

    def _get_net_params(self, opt, vocab):
        net_params = {
            'input_vocab_size': len(vocab['question_token_to_idx']),
            'output_vocab_size': len(vocab['program_token_to_idx']),
            'hidden_size': opt.hidden_size,
            'word_vec_dim': opt.word_vec_dim,
            'n_layers': opt.n_layers,
            'bidirectional': opt.bidirectional,
            'variable_lengths': opt.variable_lengths,
            'use_attention': opt.use_attention,
            'encoder_max_len': opt.encoder_max_len,
            'decoder_max_len': opt.decoder_max_len,
            'start_id': opt.start_id,
            'end_id': opt.end_id,
            'word2vec_path': opt.word2vec_path,
            'fix_embedding': opt.fix_embedding,
        }
        return net_params

    def _sort_batch(self, x, y):
        _, lengths = torch.eq(x, self.end_id).max(1)
        lengths += 1
        lengths_sorted, idx_sorted = lengths.sort(0, descending=True)
        x_sorted = x[idx_sorted]
        y_sorted = None
        if y is not None:
            y_sorted = y[idx_sorted]
        lengths_list = lengths_sorted.cpu().numpy()
        return x_sorted, y_sorted, lengths_list, idx_sorted

    def _restore_order(self, x):
        if self.idx_sorted is not None:
            inv_idxs = self.idx_sorted.clone()
            inv_idxs.scatter_(0, self.idx_sorted,
                              torch.arange(x.size(0)).to(inv_idxs.device).long())
            return x[inv_idxs]
        return x

    def _to_var(self, x):
        if len(self.gpu_ids) > 0 and torch.cuda.is_available():
            x = x.cuda()
        return Variable(x)

    def _to_numpy(self, x):
        return x.data.cpu().numpy().astype(float)

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
max_seq_length = 100
PAD_TAG = "<PAD>"
UNKNOWN_TOKEN = "<UNK>"
START_TAG = "<SOS>"
STOP_TAG = "<EOS>"
tag_to_ix = {PAD_TAG:0,START_TAG:1,STOP_TAG:2,
             "O": 3, "B-D": 4, "B-T": 5,"B-S": 6,"B-C": 7,"B-P": 8,"B-B": 9,
             "D": 10, "T": 11,"S": 12,"C": 13,"P": 14,"B": 15
             }
class crf(nn.Module):
    def __init__(self, num_tags,batch_size):
        super().__init__()
        self.num_tags = num_tags
        self.batch_size = batch_size
        # matrix of transition scores from j to i
        # self.trans = nn.Parameter(torch.Tensor(num_tags, num_tags).cuda()).data.zero_()
        self.trans = nn.Parameter(torch.rand(num_tags, num_tags).cuda())
        self.trans.data[tag_to_ix[START_TAG], :] = -10000 # no transition to SOS
        self.trans.data[:, tag_to_ix[STOP_TAG]] = -10000 # no transition from EOS except to PAD
        self.trans.data[:, tag_to_ix[PAD_TAG]] = -10000 # no transition from PAD except to PAD
        self.trans.data[tag_to_ix[PAD_TAG], :] = -10000 # no transition to PAD except from EOS
        self.trans.data[tag_to_ix[PAD_TAG], tag_to_ix[STOP_TAG]] = 0
        self.trans.data[tag_to_ix[PAD_TAG], tag_to_ix[PAD_TAG]] = 0

    def forward(self, h, mask): # forward algorithm
        # initialize forward variables in log space
        score = torch.Tensor(self.batch_size, self.num_tags).fill_(-10000).cuda()
        score[:, tag_to_ix[START_TAG]] = 0.
        trans = self.trans.unsqueeze(0) # [1, C, C]
        for t in range(h.size(1)): # recursion through the sequence
            mask_t = mask[:, t].unsqueeze(1)
            emit_t = h[:, t].unsqueeze(2) # [B, C, 1]
            score_t = score.unsqueeze(1) + emit_t + trans # [B, 1, C] -> [B, C, C]
            score_t = log_sum_exp(score_t) # [B, C, C] -> [B, C]
            score = score_t * mask_t + score * (1 - mask_t)
        score = log_sum_exp(score + self.trans[tag_to_ix[STOP_TAG]])
        return score # partition function

    def score(self, h, y, mask): # calculate the score of a given sequence
        score = torch.Tensor(self.batch_size).fill_(0.).cuda()
        h = h.unsqueeze(3)
        trans = self.trans.unsqueeze(2)
        for t in range(h.size(1)): # recursion through the sequence
            mask_t = mask[:, t]
            emit_t = torch.cat([h[t, y[t + 1]] for h, y in zip(h, y)])
            trans_t = torch.cat([trans[y[t + 1], y[t]] for y in y])
            score += (emit_t + trans_t) * mask_t
        last_tag = y.gather(1, mask.sum(1).long().unsqueeze(1)).squeeze(1)
        score += self.trans[tag_to_ix[STOP_TAG], last_tag]
        return score

    def decode(self, h, mask): # Viterbi decoding
        # initialize backpointers and viterbi variables in log space
        bptr = torch.LongTensor().cuda()
        score = torch.Tensor(self.batch_size, self.num_tags).fill_(-10000).cuda()
        score[:, tag_to_ix[START_TAG]] = 0.

        for t in range(h.size(1)): # recursion through the sequence
            mask_t = mask[:, t].unsqueeze(1)
            score_t = score.unsqueeze(1) + self.trans # [B, 1, C] -> [B, C, C]
            score_t, bptr_t = score_t.max(2) # best previous scores and tags
            score_t += h[:, t] # plus emission scores
            bptr = torch.cat((bptr, bptr_t.unsqueeze(1)), 1)
            score = score_t * mask_t + score * (1 - mask_t)
        score += self.trans[tag_to_ix[STOP_TAG]]
        best_score, best_tag = torch.max(score, 1)

        # back-tracking
        bptr = bptr.tolist()
        best_path = [[i] for i in best_tag.tolist()]
        for b in range(self.batch_size):
            x = best_tag[b] # best tag
            y = int(mask[b].sum().item())
            for bptr_t in reversed(bptr[b][:y]):
                x = bptr_t[x]
                best_path[b].append(x)
            best_path[b].pop()
            best_path[b].reverse()

        return best_path

def log_sum_exp(x):
    m = torch.max(x, -1)[0]
    return m + torch.log(torch.sum(torch.exp(x - m.unsqueeze(-1)), -1))
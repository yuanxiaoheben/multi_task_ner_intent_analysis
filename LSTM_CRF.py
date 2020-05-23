import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from CRF import crf
max_seq_length = 100
PAD_TAG = "<PAD>"
UNKNOWN_TOKEN = "<UNK>"
START_TAG = "<SOS>"
STOP_TAG = "<EOS>"
tag_to_ix = {PAD_TAG:0,START_TAG:1,STOP_TAG:2,
             "O": 3, "B-D": 4, "B-T": 5,"B-S": 6,"B-C": 7,"B-P": 8,"B-B": 9,
             "D": 10, "T": 11,"S": 12,"C": 13,"P": 14,"B": 15
             }


class BiLSTM(nn.Module):

    def __init__(self, vocab_size, max_len, embedding_dim, hidden_dim, batch_size, dropout_prob, word_embeddings=None):
        super(BiLSTM, self).__init__()
        if not word_embeddings is None:
            self.word_embeds,self.word_embedding_dim = self._load_embeddings(word_embeddings)
        else:
            self.word_embedding_dim = embedding_dim
            self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.dropout = nn.Dropout(p=dropout_prob)
        self.max_len = max_len
        self.lstm = torch.nn.LSTM(self.word_embedding_dim,hidden_dim // 2, 2, batch_first=True,bidirectional=True, dropout=dropout_prob)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (Variable(torch.zeros(4,self.batch_size, self.hidden_dim // 2).cuda()),
                Variable(torch.zeros(4,self.batch_size, self.hidden_dim // 2).cuda()))
    
    def _load_embeddings(self,embeddings):
        word_embeddings = torch.nn.Embedding(embeddings.size(0), embeddings.size(1))
        word_embeddings.weight = torch.nn.Parameter(embeddings)
        emb_dim = embeddings.size(1)
        return word_embeddings,emb_dim
    
    def forward(self, sentence):  
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        embeds = embeds.view(self.batch_size,self.max_len,-1)
        embeds = self.dropout(embeds)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)

        return lstm_out

class rnn_crf(nn.Module):
    def __init__(self, bilstm, num_tags,batch_size,max_len,dropout_prob):
        super().__init__()
        self.batch_size = batch_size
        self.max_len = max_len
        self.bilstm = bilstm
        self.crf = crf(num_tags,batch_size)
        self.hidden2tag = nn.Linear(bilstm.r + bilstm.lstm_hid_dim, num_tags)
        self.dropout = nn.Dropout(p=dropout_prob)
    def _get_lstm_features(self, sentence,mask):
        _,_,attention = self.bilstm(sentence)
        attention = self.dropout(attention)
        lstm_feats = self.hidden2tag(attention)
        lstm_feats *= mask.unsqueeze(2)
        return lstm_feats
    def forward(self, x, y):
        mask = x.gt(0).float()
        lstm_feats = self._get_lstm_features(x,mask)
        Z = self.crf.forward(lstm_feats, mask)
        score = self.crf.score(lstm_feats, y, mask)
        return torch.mean(Z - score) # NLL loss

    def decode(self, x): # for inference 
        mask = x.gt(0).float()
        h = self._get_lstm_features(x, mask)
        return self.crf.decode(h, mask)

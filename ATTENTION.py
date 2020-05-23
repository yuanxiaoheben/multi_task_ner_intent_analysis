
import torch
import torch.nn as nn
import torch.nn.functional as F
class SelfAttention(torch.nn.Module):
    """
    The class is an implementation of the paper A Structured Self-Attentive Sentence Embedding including regularization
    and without pruning. Slight modifications have been done for speedup
    """
   
    def __init__(self,batch_size,d_a,r,n_classes, bilstm):
        super(SelfAttention,self).__init__()
       
        self.lstm = bilstm
        self.lstm_hid_dim = self.lstm.hidden_dim
        self.linear_first = torch.nn.Linear(self.lstm_hid_dim,d_a)
        self.linear_first.bias.data.fill_(0)
        self.linear_second = torch.nn.Linear(d_a,r)
        self.linear_second.bias.data.fill_(0)
        self.n_classes = n_classes
        self.linear_final = torch.nn.Linear(self.lstm_hid_dim,self.n_classes)
        self.batch_size = batch_size
        self.r = r
        self.type = type     
        
    def softmax(self,input, axis=1):
        input_size = input.size()
        trans_input = input.transpose(axis, len(input_size)-1)
        trans_size = trans_input.size()
        input_2d = trans_input.contiguous().view(-1, trans_size[-1])
        soft_max_2d = F.softmax(input_2d,dim=1)
        soft_max_nd = soft_max_2d.view(*trans_size)
        return soft_max_nd.transpose(axis, len(input_size)-1)       
        
    def forward(self,x):
        outputs = self.lstm(x)
        x = torch.tanh(self.linear_first(outputs))       
        x = self.linear_second(x)       
        x = self.softmax(x,1)
        concat_out = torch.cat((outputs, x), 2)       
        attention = x.transpose(1,2)       
        sentence_embeddings = attention@outputs 
        avg_sentence_embeddings = torch.sum(sentence_embeddings,1)/self.r
        output = self.linear_final(avg_sentence_embeddings)
        return output,attention,concat_out

    def l2_matrix_norm(self,m):
        return torch.sum(torch.sum(torch.sum(m**2,1),1)**0.5).type(torch.DoubleTensor)

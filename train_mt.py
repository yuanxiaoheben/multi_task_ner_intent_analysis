import torch.optim as optim
from torch.utils.data import Dataset,DataLoader,TensorDataset
from LSTM_CRF import *
from utils import *
import pandas as pd
from ATTENTION import *
import os
import json
from sklearn.metrics import f1_score

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--train_intent_corpus",
                        default=None,
                        type=str,
                        required=True,
                        help="The input train corpus.")

parser.add_argument("--train_ner_corpus",
                        default=None,
                        type=str,
                        required=True,
                        help="The input train corpus for NER.")

parser.add_argument("--intent_LR",
                        default=0.001,
                        type=float,
                        required=False,
                        help="intent analysis learning rate.")

parser.add_argument("--NER_LR",
                        default=0.005,
                        type=float,
                        required=False,
                        help="NER analysis learning rate.")

parser.add_argument("--epochs",
                        default=100,
                        type=int,
                        required=False,
                        help="model training epochs.")

parser.add_argument("--dropout_prob",
                        default=0.55,
                        type=float,
                        required=False,
                        help="Dropout")

parser.add_argument("--lstm_hid_dim",
                        default=1000,
                        type=int,
                        required=True,
                        help="knowledge label model path.")

parser.add_argument("--embedding_dim",
                        default=300,
                        type=str,
                        required=True,
                        help="embedding size")

parser.add_argument("--word2vec_path",
                        default=None,
                        type=str,
                        required=False,
                        help="word2vec path")

parser.add_argument("--max_seq_length",
                        default='90',
                        type=int,
                        required=False,
                        help="maximum sequence length")

args = parser.parse_args()





int_train_data = pd.read_csv(args.train_intent_corpus)
ner_train_data = pd.read_csv(args.train_ner_corpus)

train_int_set = gen_data_list(int_train_data)
train_ner_set = ner_list(ner_train_data)

with open('char_to_ix.json', 'r') as result_file:
    char_to_ix = json.load(result_file)
char_size = len(char_to_ix)
word_embedding = load_word2vec_embeddings(args.word2vec_path, char_to_ix, args.embedding_dim)

int_char_train = torch.LongTensor(np.array([prepare_sequence(x[0], char_to_ix, args.max_seq_length) for x in train_int_set])).cuda()
int_y_train = torch.FloatTensor(np.array([x[1] for x in train_int_set])).cuda()
int_train_dataset = TensorDataset(int_char_train,int_y_train)


ner_char_train = torch.LongTensor(np.array([prepare_sequence(x[0], char_to_ix, args.max_seq_length) for x in train_ner_set])).cuda()
ner_y_train = torch.LongTensor(np.array([prepare_tag(x[1], tag_to_ix, args.max_seq_length) for x in train_ner_set])).cuda()
ner_train_dataset = TensorDataset(ner_char_train,ner_y_train)


def training_loop(model, loss, optimizer, epochs, train_dataset):
    train_loader = DataLoader(dataset=train_dataset,batch_size=args.batch_size,shuffle=True,drop_last=True)
    for j in range(epochs):
        loss_sum = 0
        for i,data in enumerate(train_loader):
            model.train()
            char_data,labels = data
            #input_data.enable_grad()
            #labels.enable_grad()

            model.zero_grad()
            output,_,_ = model(char_data)
            lossy = loss(output, labels)
            lossy.backward()
            optimizer.step()
            loss_sum = loss_sum + lossy.detach().cpu().numpy()
        print( "Loss %f" %(loss_sum))



def tag_processing(input_batch):
    curr_arr = input_batch.detach().cpu().numpy()
    output_arr = []
    for row in curr_arr:
        new_sentence = []
        for tag in row:
            if tag == tag_to_ix[START_TAG] or tag == tag_to_ix[PAD_TAG]:
                continue
            else:
                new_sentence.append(tag)
        output_arr.append(new_sentence)
    return output_arr
    
def predict(model, ner_dataset):
    train_loader = DataLoader(dataset=ner_dataset,batch_size=args.batch_size,shuffle=True,drop_last=True)
    target = []
    output = []
    for i,data in enumerate(train_loader):
        char_data,tags = data
        result = model.decode(char_data)[:args.batch_size]
        target =  target + tag_processing(tags)
        output = output + result
    return target,output

def training_CRF(model, optimizer, epochs, train_dataset):
    train_loader = DataLoader(dataset=train_dataset,batch_size=args.batch_size,shuffle=True,drop_last=True)
    for j in range(epochs):
        loss_sum = 0
        for i,data in enumerate(train_loader):
            model.train()
            char_data,tags = data
            model.zero_grad()
            loss =  model(char_data, tags)
            loss.backward()
            loss_sum = loss_sum+loss.sum().item()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr']  / (1 + (j + 1) * 0.01)
        
        print( "Loss %f" %(loss_sum))
        
# Hyper Parameters
hyper_parameters = {
    'num_labels':7,
    'd_a':100,
    'r':400
}
print(hyper_parameters)


lstm_model = BiLSTM(char_size, args.max_seq_length,args.embedding_dim, args.lstm_hid_dim, args.batch_size,
    args.dropout_prob, word_embedding)

atention_model = SelfAttention(args.batch_size,hyper_parameters['d_a'],hyper_parameters['r'],hyper_parameters['num_labels'], lstm_model).cuda()
    
# Loss and Optimizer
loss_class = nn.BCEWithLogitsLoss()  
optimizer_class = torch.optim.Adam(atention_model.parameters(), lr=args.intent_LR)

atention_model2 = SelfAttention(args.batch_size,hyper_parameters['d_a'],hyper_parameters['r'],hyper_parameters['num_labels'], lstm_model).cuda()
crf_model = rnn_crf(atention_model2, len(tag_to_ix),args.batch_size,max_seq_length,
    args.dropout_prob).cuda()
# Loss and Optimizer
optimizer = torch.optim.Adam(crf_model.parameters(), lr=args.NER_LR)

# Train the model
def training_all(epochs):
    for j in range(epochs):
        print( "Epochs %i" %(j+1))
        training_loop(atention_model, loss_class, optimizer_class, 1, int_train_dataset)
        training_CRF(crf_model, optimizer,1, ner_train_dataset)
        
training_all(args.epochs)
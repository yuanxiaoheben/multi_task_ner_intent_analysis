import torch
import torch.autograd as autograd

from torch.utils.data import Dataset,DataLoader,TensorDataset
import numpy as np
import re
import pandas as pd

PAD_TAG = "<PAD>"
UNKNOWN_TOKEN = "<UNK>"
START_TAG = "<SOS>"
STOP_TAG = "<EOS>"
tag_to_ix = {PAD_TAG:0,START_TAG:1,STOP_TAG:2,
             "O": 3, "B-D": 4, "B-T": 5,"B-S": 6,"B-C": 7,"B-P": 8,"B-B": 9,
             "D": 10, "T": 11,"S": 12,"C": 13,"P": 14,"B": 15
             }

emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
def text_processing(text):
    text = re.sub(r'[\s\t\r\n]', "", text)
    text = re.sub(r'[\"\']', "", text)
    text = emoji_pattern.sub(r'', text)
    return [c for c in text]

# change ouput tag style 'S T' --> [0,1,1,0,0,0,0]
tags_mapping = ['C','S','T','R','M','D','P']
def generate_tags(values):
    values = values.upper()
    values = values.replace('\s','')
    out = [0,0,0,0,0,0,0]
    for i in range(len(tags_mapping)):
        if tags_mapping[i] in values:
            out[i] = 1
    return out 

def gen_data_list(df):
    data_list = []
    for idx,row in df.iterrows():
        curr = [text_processing(row['Question']), generate_tags(row['Intention'])]
        data_list.append(curr)
    return data_list
def ner_list(df):
    data_list = []
    for idx,row in df.iterrows():
        curr = [[x for x in row['text']], row['tags'].split(' ')]
        data_list.append(curr)
    return data_list

def build_dictionary(dataset):
    """
    input structure: [word, tags, label]
    """      
    vocabulary_char = set()
    for row in dataset:
        for word in row[0]:
            vocabulary_char.add(word)
    
    vocabulary_char = list(vocabulary_char)
    vocabulary_char = [PAD_TAG,START_TAG,STOP_TAG,UNKNOWN_TOKEN] + vocabulary_char
    char_indices = dict(zip(vocabulary_char, range(len(vocabulary_char))))

    return char_indices, len(vocabulary_char)
def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, to_ix, max_seq_length):
    idxs = []
    if len(seq) >= max_seq_length:
        idxs = [to_ix[seq[i]] for i in range(max_seq_length)]
        return idxs
    for i in range(max_seq_length):
        if i < len(seq):
            if seq[i-1] in to_ix:
                idxs.append(to_ix[seq[i-1]])
            else:
                idxs.append(to_ix[UNKNOWN_TOKEN])
        else:
            idxs.append(to_ix[PAD_TAG])
    return idxs
def prepare_tag(seq, to_ix, max_seq_length):
    idxs = []
    if len(seq) >= max_seq_length:
        idxs = [to_ix[START_TAG]] + [to_ix[seq[i]] for i in range(max_seq_length)]
        return idxs
    for i in range(max_seq_length+1):
        if i == 0:
            idxs.append(to_ix[START_TAG])
        elif i < len(seq)+1:
            idxs.append(to_ix[seq[i-1]])
        else:
            idxs.append(to_ix[PAD_TAG])
    return idxs

def load_word2vec_embeddings(path, word2idx, embedding_dim):
    """Loading the word2vec embeddings"""
    with open(path, encoding='utf-8') as f:
        idx = 0
        embeddings = np.zeros((len(word2idx), embedding_dim))
        for line in f.readlines():
            if idx == 0:
                print(line)
            idx = idx + 1
            values = line.split()
            word = values[0]
            index = word2idx.get(word)
            if index:
                vector = np.array(values[1:], dtype='float32')
                if vector.shape[-1] != embedding_dim:
                    raise Exception('Dimension not matching.')
                embeddings[index] = vector
        return torch.from_numpy(embeddings).float()
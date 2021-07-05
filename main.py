#%% Preprocessing
import torch
import torchtext
from torchtext import data
import sys
import numpy as np
import pandas as pd
import spacy
import random
from yaml import safe_load

from src.preprocessing import *
from src.model import *
from src.train import *

print(f"System version: {sys.version}")
print(f"Numpy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")
print(f"Torch version: {torch.__version__}")
print(f"Torchtext version: {torchtext.__version__}")
print(f"Spacy version: {spacy.__version__}")

with open("parameters.yml","r",encoding='utf8') as stream:
    parameters = safe_load(stream)

train_path = parameters['preprocessing_parameters']['train_path']
test_path = parameters['preprocessing_parameters']['test_path']
vectors = parameters['preprocessing_parameters']['vectors']
tokenizer_language = parameters['preprocessing_parameters']['tokenizer_language']

SEED = parameters['preprocessing_parameters']['SEED']
MAX_VOCAB_SIZE = parameters['preprocessing_parameters']['MAX_VOCAB_SIZE']
BATCH_SIZE = parameters['preprocessing_parameters']['BATCH_SIZE']

torch.backends.cudnn.deterministic = parameters['preprocessing_parameters']['deterministic']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TEXT = data.Field(tokenize = 'spacy',
                  tokenizer_language = tokenizer_language,
                  include_lengths = True)

LABEL = data.LabelField(dtype = torch.float)

train_data, valid_data, test_data = split_data(train_path,test_path,TEXT,LABEL,random.seed(SEED))

build_vocab(TEXT=TEXT, LABEL=LABEL, train_data=train_data,MAX_VOCAB_SIZE=MAX_VOCAB_SIZE,vectors=vectors)

train_iterator, valid_iterator, test_iterator = get_iterators(train_data,valid_data,test_data,BATCH_SIZE=BATCH_SIZE)

#%% Build the model
 
INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = parameters['model_parameters']['EMBEDDING_DIM']
HIDDEN_DIM = parameters['model_parameters']['HIDDEN_DIM']
OUTPUT_DIM = parameters['model_parameters']['OUTPUT_DIM']
N_LAYERS = parameters['model_parameters']['N_LAYERS']
BIDIRECTIONAL = parameters['model_parameters']['BIDIRECTIONAL']
DROPOUT = parameters['model_parameters']['DROPOUT']
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

model = RNN(INPUT_DIM, 
            EMBEDDING_DIM, 
            HIDDEN_DIM, 
            OUTPUT_DIM, 
            N_LAYERS, 
            BIDIRECTIONAL, 
            DROPOUT, 
            PAD_IDX)



print(f'The model has {count_parameters(model):,} trainable parameters')

pretrained_embeddings = TEXT.vocab.vectors

print(pretrained_embeddings.shape)

model.embedding.weight.data.copy_(pretrained_embeddings)

UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]

model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

print(model.embedding.weight.data)

#%% Train the Model

import torch.optim as optim

optimizer = optim.Adam(model.parameters())

criterion = nn.BCEWithLogitsLoss()

model = model.to(device)
criterion = criterion.to(device)

import time

N_EPOCHS = parameters['model_parameters']['N_EPOCHS']
model_dict_name = parameters['model_parameters']['model_dict_name']
best_valid_loss = float('inf')

import pandas as pd

train_history=pd.DataFrame({'train_loss':[],'train_acc':[]})
valid_history=pd.DataFrame({'valid_loss':[],'valid_acc':[]})

for epoch in range(N_EPOCHS):

    start_time = time.time()
    
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
    
    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        print(f"Saving model parameters to [{model_dict_name}]")
        torch.save(model.state_dict(), model_dict_name)
    train_result_dict = {'train_loss':train_loss,'train_acc':train_acc}
    valid_result_dict = {'valid_loss':valid_loss,'valid_acc':valid_acc}
    
    train_history[epoch]=train_result_dict
    valid_history[epoch]=valid_result_dict
    
    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

from matplotlib import pyplot as plt

train_hist = train_history.plot(title=f"Training History - {model_dict_name}")
plt.savefig(f"{model_dict_name}_train_hist.png")

valid_hist = valid_history.plot(title=f"Validation History - {model_dict_name}")
plt.savefig(f"{model_dict_name}_val_hist.png")

model.load_state_dict(torch.load(model_dict_name))

test_loss, test_acc = evaluate(model, test_iterator, criterion)

print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')

model_name = parameters['model_parameters']['model_name']

print(f"Saving model as [{model_name}]")
torch.save(model, model_name)
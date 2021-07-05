import torch
from torchtext import data

def split_data(train,test,TEXT,LABEL,SEED,path='',format='csv',split_ratio=0.8):    
    fields = [('Text', TEXT), ('Label', LABEL)]
    
    train_data, test_data = data.TabularDataset.splits(
        path = path,
        train = train,
        test = test,
        format = format, #'tsv' for tabs, 'csv' for commas
        fields = fields,
        skip_header=True)
    
    train_data, valid_data = train_data.split(split_ratio=split_ratio,random_state=SEED)
    
    return train_data, test_data, valid_data

def build_vocab(TEXT, LABEL, train_data, MAX_VOCAB_SIZE = 25_000, vectors = "glove.6B.100d"):
    MAX_VOCAB_SIZE = 25_000

    TEXT.build_vocab(train_data, 
                    max_size = MAX_VOCAB_SIZE, 
                    vectors = vectors, 
                    unk_init = torch.Tensor.normal_)

    LABEL.build_vocab(train_data)


def get_iterators(train_data, valid_data, test_data,BATCH_SIZE = 64):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size = BATCH_SIZE,
        sort_key=lambda x: len(x.Text),
        sort_within_batch = True,
        device = device)
    
    return train_iterator, valid_iterator, test_iterator
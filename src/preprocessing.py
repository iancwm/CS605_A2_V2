import torch
import re
from torchtext import data
from yaml import safe_load
import pickle
import pandas as pd


def clean_text(text):
    text_new = re.sub(r'[^\w\s.]', '', text)  # only keep words, numbers and .
    text_new = [word for word in text_new.split(
        ' ') if word != 'br']  # br comes from < \br>
    text_new = " ".join(text_new[:128]+text_new[-382:]
                        if len(text_new) > 512 else text_new)
    # This step is used to deal with long text. I will keep the head and tail when the length
    # of tokens is greater than 512
    return(text_new)


def preprocess_text(folder, file_path):

    df = pd.read_csv(f"{folder}\{file_path}")
    df['Text_preprocessed'] = df['Text'].apply(lambda x: clean_text(x))
    df[['Text_preprocessed', 'Label']].to_csv(
        f"{folder}\preprocessed_{file_path}")
    return None


def load_parameters(yaml_path):
    """Loads YAML parameters for experiments    
    """
    with open(yaml_path, "r", encoding='utf8') as stream:
        parameters = safe_load(stream)
    return parameters


def split_data(train_path, test_path, TEXT, LABEL, SEED, path='', format='csv', split_ratio=0.8):
    """Returns a tuple containing the split data from

    Args:
        train_path (str):               String containing filename of training data
        test_path (str):                String containing filename of testing data
        TEXT (torchtext Field):         torchtext Field
        LABEL (torchtext LabelField):   torchtext LabelField
        SEED (int):                     Random seed for reproducibility
        path (str):                     Folder path (default blank for root)
        format (str):                   Format of source data (default csv)
        split_ratio (float<1.0):        Fraction of data to split at (default 0.8)

    Yields:
        train_data (torchtext.data.TabularDataset):      training data
        test_data (torchtext.data.TabularDataset):       testing data
        valid_data (torchtext.data.TabularDataset):      validation data
    """
    fields = [('Text_preprocessed', TEXT), ('Label', LABEL)]

    train_data, test_data = data.TabularDataset.splits(
        path=path,
        train=train_path,
        test=test_path,
        format=format,  # 'tsv' for tabs, 'csv' for commas
        fields=fields,
        skip_header=True)

    train_data, valid_data = train_data.split(
        split_ratio=split_ratio, random_state=SEED)

    return train_data, test_data, valid_data


def build_vocab(TEXT, LABEL, train_data, MAX_VOCAB_SIZE=25_000, vectors="glove.6B.100d"):
    """Builds vocabulary of torchtext fields in place

    Args:
        TEXT (torchtext Field):                     torchtext Field
        LABEL (torchtext LabelField):               torchtext LabelField
        train_data (torchtext.data.TabularDataset): Training dataset
        MAX_VOCAB_SIZE (int):                       The maximum size of the vocabulary, or None for no maximum.
        vectors (str):                              Prebuilt word vectors to use

    Yields:
        None

    Notes: 
        Generates Vocab object in place.
    """
    MAX_VOCAB_SIZE = 25_000

    TEXT.build_vocab(train_data,
                     max_size=MAX_VOCAB_SIZE,
                     vectors=vectors,
                     unk_init=torch.Tensor.normal_)

    LABEL.build_vocab(train_data)


def get_iterators(train_data, valid_data, test_data, BATCH_SIZE=64):
    """Prepares iterators to be used in the models

    Args:
        train_data (torchtext.data.TabularDataset): Training dataset
        valid_data (torchtext.data.TabularDataset): Validation dataset
        test_data (torchtext.data.TabularDataset):  Test dataset
        BATCH_SIZE (int):                           Number of examples in batch

    Yields:
        train_iterator (torchtext.data.Iterator):   Iterator for training data
        valid_iterator (torchtext.data.Iterator):   Iterator for validation data
        test_iterator (torchtext.data.Iterator):    Iterator for testing data

    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=BATCH_SIZE,
        sort_key=lambda x: len(x.Text_preprocessed),
        sort_within_batch=True,
        device=device)

    return train_iterator, valid_iterator, test_iterator


def save_vocab(filename):
    """Dumps vocabulary file for use in prediction
    """
    with open(filename, 'wb') as f:
        pickle.dump(TEXT, f)


def load_vocab(filename):
    """Loads vocabulary file for predictions

    Args:
        filename (str):                 Path to Vocab object

    Yields:
        TEXT_load (torchtext.Vocab):    Vocab object for use in prediction

    """
    with open(filename, 'rb') as f:
        TEXT_load = pickle.load(f)
    return TEXT_load

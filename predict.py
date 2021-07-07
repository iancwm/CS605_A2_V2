import spacy
import torch
import pickle
from src.preprocessing import load_parameters
from src.model import *

parameters = load_parameters('parameters.yml')

filename = parameters['persistence_parameters']['vocab_filename']

with open(filename, 'rb') as f:
    TEXT_load = pickle.load(f)

model_name = parameters['model_parameters']['model_name']
tokenizer_language = parameters['preprocessing_parameters']['tokenizer_language']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

nlp = spacy.load(tokenizer_language)

model = torch.load(model_name)
model.eval()

predict_sentiment(model, "This film is terrible", TEXT_load, nlp, device)

predict_sentiment(model, "This film is great", TEXT_load, nlp, device)
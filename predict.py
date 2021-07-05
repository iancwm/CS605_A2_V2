import spacy
import torch
from yaml import safe_load

with open("parameters.yml","r",encoding='utf8') as stream:
    parameters = safe_load(stream)

model_name = parameters['model_parameters']['model_name']
tokenizer_language = parameters['preprocessing_parameters']['tokenizer_language']

nlp = spacy.load(tokenizer_language)

def predict_sentiment(model, sentence):
    model.eval()
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    length = [len(indexed)]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(1)
    length_tensor = torch.LongTensor(length)
    prediction = torch.sigmoid(model(tensor, length_tensor))
    result = prediction.item()
    print(f"Results for {sentence}")
    print(result)
    return result

model = torch.load(model_name)
model.eval()

predict_sentiment(model, "This film is terrible")

predict_sentiment(model, "This film is great")
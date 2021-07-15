# Instructions [IMPORTANT]

## Installation and setup

Run the following in command line to setup the virtual environment

1. `pip install -r requirements.txt`
2. `python -m spacy download en_core_web_sm`

Next, open up a python interpreter and run the following to set up `nltk`:

1. `nltk.download('stopwords')`
2. `nltk.download('wordnet')`

Packages specified in `requirements.txt` are:

|package|version|comments|
|---|---|---|
|`torch`|`1.7.0+cu101`|Use `pip` in anaconda|
|`torchtext`|`0.8.0`||
|`numpy`|`1.19.5`||
|`spacy`|`3.0.6`|Install `en_core_web_sm`|
|`nltk`|`3.6.2`|Download packages via `nltk.download()`|

## Summary
1. Place data (e.g. `test.csv`) into `data` folder
2. With the virtual environment activated, run the following to predict (replace `Test-format.csv` with file name):

> ```python predict.py 2_layer_512.pt test.csv```

# Layout

## Folder structure

The project consists of the following folders:

- `src`:        Stores python modules containing code for pipeline such as preprocessing, model specifications, prediction etc.
- `data`:       Contains all data files, including preprocessed train data. Please place unseen data for prediction here

The code will create the following folders:

- `training_results`:   Stores the results of the training for each model
- `charts`:             Visualizations for the training process
- `model`:              Saves the models here
- `vocab`:              Saves the generated Vocab object used for prediction
- `output`:             Saves all critical model output such as `.csv` files containing *unpreprocessed* prediction text and sentiment classification (`0` for negative or `1` for positive) for each run of the prediction script, and one `training_results.csv` of performance of models trained

## YAML parameters

The repo contains a `parameters.yml` file that contains all critical parameters and values.

Change important experiment parameters in this file, including:
- model architecture
- file directories
- prebuilt word vectors and more.

**IMPORTANT**: Change `model_name` and `model_dict_name` on each training run to avoid overwriting other experiments.

----

# Instructions

For all three stages, parameters will be extracted from `parameters.yml`, and the relevant folders created for output.

## 1. Preprocessing
This is straight forward - run `preprocess.py`. The preprocessed train and validation data will be output to the specified `folders:data_path` folder under `parameters.yml`, with a prefix `preprocessed_`.

## 2. Training the model
This is straight forward - run `run.py`.

The model will be trained on the pre-processed output from step 1, which is now output to the specified `folders:data_path` folder under `parameters.yml`.

Graphs of training and validation performance will also be output to the specified `folders:chart_dir` folder specified in `parameters.yml`

## 3. Prediction
This file is run from the command line, and the required syntax is:

> `python predict.py <model_name> <target_file>`

The script will:
1. Look for `folders:model_name` in the directory specified in `parameters.yml`. By default this is `model`
2. Look for `folders:data_path` in the directory specified in `parameters.yml`. By default this is `data`
3. Parse the file and preprocess the text
4. Load the saved `Vocab` object and model specified in `parameters.yml`. By default, this is `vocab`
5. Use the model to predict the sentiment of the input. If the score is above the threshold set in parameters (default is 0.5), it will return 1 (positive sentiment), else it will return 0 (negative sentiment).
6. The results will be written to a `.csv` under the `folders:output_path` entry in `parameters.yml`. By default this is `output`. The naming convention is `prediction_{model_name.pt}`. The format of the output will be without headers `'unpreprocessed text, prediction'`

----
# Description of Model

The network is constructed using `pytorch 1.7` and `torchtext 0.8.1`, and consists of the following layers and hyperparameters:

1. `nn.Embedding()`
    - `vocab_size` - taken from `parameters.yml`
    - `embedding_dim`- taken from `parameters.yml`
    - `pad_idx` - taken from `parameters.yml`
2. `nn.LSTM()`
    - `embedding_dim` - taken from `parameters.yml`
    - `hidden_dim` - taken from `parameters.yml`
    - `n_layers` - taken from `parameters.yml`
    - `bidrectional` - taken from `parameters.yml`
3. `nn.Linear`
4. `nn.Dropout`
    - `dropout` - taken from `parameters.yml`

## Optimizer

The optimizer used was Adam (<u>**Ada**</u>ptive <u>**M**</u>oment Estimation), an extension to the commonly used SGD. Where SGD only uses a single learning rate, Adam maintains a learning rate for each parameter, which improves performance on problems with sparse gradients such as NLP.

## Loss

The loss criterion chosen was `BCEWithLogitsLoss()`, which is a binary crossentropy loss bundled with a sigmoid layer.
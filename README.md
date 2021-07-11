# Layout

## Folder structure

The project consists of the following folders:

- `src`:        Stores python modules containing code for pipeline such as preprocessing, model specifications, prediction etc.
- `data`:       Contains all data files, including preprocessed train data. Please place unseen data for prediction here

The code will create the following folders:

- `results`:    Stores the results of the training for each model
- `charts`:     Visualizations for the training process
- `model`:      Saves the models here
- `vocab`:      Saves the generated Vocab object used for prediction

## YAML parameters

The repo contains a `parameters.yml` file that contains all critical parameters and values.

Change important experiment parameters in this file, including:
- model architecture
- file directories
- prebuilt word vectors and more.

**IMPORTANT**: Change `model_name` and `model_dict_name` to avoid overwriting other experiments

----

# Directions

## 1. Training the model
This is straight forward - run `main.py`

## 2. Prediction
This file is run from the command line, and the required syntax is:

`python predict.py <model_name> <target_file>`

The script will look for `model_name` in the directory specified in the `YAML` parameter `model_parameters`>`model_dir`

The script will look for the data in the `data` folder.
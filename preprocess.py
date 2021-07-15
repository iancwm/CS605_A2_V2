from src.preprocessing import *

parameters = load_parameters('parameters.yml')
# %% Text Preprocessing
train_path = parameters['folders']['train_path']
test_path = parameters['folders']['test_path']
data_path = parameters['folders']['data_path']
output_path = parameters['folders']['output_path']

preprocess_text(data_path, train_path)
preprocess_text(data_path, test_path)

print("All text preprocessed! Run run.py next!")

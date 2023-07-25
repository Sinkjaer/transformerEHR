# %%
import json
import random


# %% Split synthetic data into train and val
def split_synthetic(file_path, train_prob=0.8):
    with open(file_path) as f:
        data = json.load(f)
    train_data = {}
    val_data = {}
    for patient, sequences in data.items():
        if random.random() < train_prob:
            train_data[patient] = sequences
        else:
            val_data[patient] = sequences
    train_file_path = file_path.replace(".json", "_train.json")
    val_file_path = file_path.replace(".json", "_val.json")
    with open(train_file_path, "w") as f:
        json.dump(train_data, f)
    with open(val_file_path, "w") as f:
        json.dump(val_data, f)
    return train_data, val_data


file_path = "/Users/mikkelsinkjaer/Library/Mobile Documents/com~apple~CloudDocs/transEHR/Code/transformerEHR/data/syntheticData.json"
train_data, val_data = split_synthetic(file_path, train_prob=0.8)

# %%

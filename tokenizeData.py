# %% Load
from torchtext.vocab import vocab
from collections import Counter

import json
from dateutil.relativedelta import relativedelta
from datetime import datetime

# %% global variables
file_path = "H:/Code/transformerEHR/syntheticData.json"
SEP_TOKEN = "<SEP>"
PAD_TOKEN = "<PAD>"
# %% Make vocab


# unique a list
def unique_strings(string_list):
    unique_list = list(set(string_list))
    return unique_list


def build_vocab(filename):
    with open(filename) as f:
        data = json.load(f)
    events = []
    for key, value in data.items():
        list_of_events = value["events"]
        for n in range(len(list_of_events)):
            events.append(list_of_events[n]["codes"])

    # build vocabulary
    events = unique_strings(events)
    counter = Counter(events)
    vocab_list = vocab(counter, specials=(SEP_TOKEN, PAD_TOKEN, "<UNK>", "<MASK>"))

    return vocab_list


# %% Tokenize including dates


vocab_list = None  # build_vocab(file_path)


def process_data(filename, vocab_list):
    with open(filename) as f:
        data = json.load(f)

    max_length = 0
    processed_data = {}

    # Variables for calculating age
    today = datetime.today()

    for patient, patient_data in data.items():
        age = datetime.strptime(patient_data["age"], "%Y-%m-%d")
        events = patient_data["events"]
        events.sort(key=lambda x: x["admdate"])  # Sort events by 'admdate'

        # Group 'admdate' and 'codes' with same 'admid' together
        admid_groups = {}
        for event in events:
            if event["admid"] in admid_groups:
                admid_groups[event["admid"]][0].append(event["admdate"])
                admid_groups[event["admid"]][1].append(event["codes"])
            else:
                admid_groups[event["admid"]] = [[event["admdate"]], [event["codes"]]]

        # Arrange 'admdate' and 'codes' into sequences with '[SEP]' separating different 'admid' groups
        date_sequence = []
        age_sequence = []
        code_sequence = []
        for date_list, code_list in admid_groups.values():
            date_sequence += date_list + [SEP_TOKEN]
            age_sequence += [
                relativedelta(datetime.strptime(date, "%Y-%m-%d"), age).years
                for date in date_list
            ] + [SEP_TOKEN]
            code_sequence += code_list + [SEP_TOKEN]

        # Remove the last '[SEP]' from the sequences
        date_sequence = date_sequence[:-1]
        age_sequence = age_sequence[:-1]
        code_sequence = code_sequence[:-1]

        if len(date_sequence) > max_length:
            max_length = len(date_sequence)

        processed_data[patient] = {
            "age": age_sequence,
            "dates": date_sequence,
            "codes": code_sequence,
        }

    # Padding all sequences to the max length*
    for patient, sequences in processed_data.items():
        for key in sequences:
            if len(sequences[key]) < max_length:
                sequences[key] += [PAD_TOKEN] * (max_length - len(sequences[key]))
        # Make embeddiding events

    return processed_data


# Restult of process_data
# data = process_data(file_path)

# %% DataLoader

from torch.utils.data import Dataset, DataLoader


class HealthDataset(Dataset):
    def __init__(self, data):
        self.data = list(data.values())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        dates = self.data[idx]["dates"]
        age = self.data[idx]["age"]
        codes = self.data[idx]["codes"]
        return dates, age, codes


# Define data_loader
data = process_data(file_path)
dataset = HealthDataset(data)
data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

# Test data_loader
for batch_idx, (age, dates, codes) in enumerate(data_loader):
    # Do something with the batch of data
    print(f"Batch {batch_idx}: Dates: {dates}, Age: {age}, Codes: {codes}")
# %%

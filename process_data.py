# %% Import libraries and initialize global variables

from torchtext.vocab import vocab
from collections import Counter

import json
from dateutil.relativedelta import relativedelta
from datetime import datetime, date

import torch
from torch.utils.data import Dataset, DataLoader

from utils import random_masking

# global variables
file_path = "H:/Code/transformerEHR/syntheticData.json"
ref_date = datetime(1900, 1, 1)
START_TOKEN_NS = 0  # non-sequence start token
START_TOKEN = "<CLS>"
SEP_TOKEN = "<SEP>"
# SEP_TOKEN_NS = # this is not used
PAD_TOKEN = "<PAD>"
PAD_TOKEN_NS = 0
MASK_TOKEN = "<MASK>"
MASK_TOKEN_NS = 0
UNKNOWN_TOKEN = "<UNK>"


# %% Make vocab


# Functions finds unique strings in a list
def unique_strings(string_list):
    unique_list = list(set(string_list))
    return unique_list


# Function should only take data file
def build_vocab(data):
    events = []
    for key, value in data.items():
        list_of_events = value["events"]
        for n in range(len(list_of_events)):
            events.append(list_of_events[n]["codes"])

    # build vocabulary
    events = unique_strings(events)
    counter = Counter(events)
    vocab_list = vocab(
        counter, specials=(UNKNOWN_TOKEN, START_TOKEN, SEP_TOKEN, PAD_TOKEN, MASK_TOKEN)
    )
    vocab_list.set_default_index(vocab_list[UNKNOWN_TOKEN])

    word_to_idx = vocab_list.get_stoi()

    return vocab_list, word_to_idx


# %% Test Tokenizer
# with open(file_path) as f:
#     data = json.load(f)
# vocab_list, word_to_idx = build_vocab(data)
# print("Known token:", vocab_list["Q12"])
# print("Unknown token:", vocab_list["Q1"])


# %% DataLoader
class MaskedDataset(Dataset):
    def __init__(self, data):
        self.data = list(data.values())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        dates = torch.tensor(self.data[idx]["dates"])
        age = torch.tensor(self.data[idx]["age"])
        codes = torch.tensor(self.data[idx]["codes_masked"])
        position = torch.tensor(self.data[idx]["position"])
        segment = torch.tensor(self.data[idx]["segment"])
        return codes, dates, age, position, segment


# %% Tokenize including dates
def process_data_MLM(data, vocab_list, word_to_idx):
    """
    Function to process the data from the json file to
    Data is processed for MLM (Masked Language Model) task
    """

    max_length = 1000
    processed_data = {}

    for patient, patient_data in data.items():
        birth_date = datetime.strptime(patient_data["birthdate"], "%Y-%m-%d")
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

        # Initialize sequences and insert start token
        date_sequence = [START_TOKEN_NS]
        age_sequence = [START_TOKEN_NS]
        code_sequence = [START_TOKEN]
        position_sqeuence = [START_TOKEN_NS]
        segment_sequence = [START_TOKEN_NS]

        position = 1
        segment = 1
        for date_list, code_list in admid_groups.values():
            date_sequence += [
                (datetime.strptime(date, "%Y-%m-%d") - ref_date).days
                for date in date_list
            ]
            age_sequence += [
                relativedelta(datetime.strptime(date, "%Y-%m-%d"), birth_date).years
                for date in date_list
            ]
            code_sequence += code_list + [SEP_TOKEN]
            position_sqeuence += [position] * (len(code_list) + 1)
            segment_sequence += [segment] * (len(code_list) + 1)
            position += 1
            segment *= -1

            # Add segment date and age where seprator is added
            date_sequence += [date_sequence[-1]]
            age_sequence += [age_sequence[-1]]

        # Remove the last '[SEP]' from the sequences
        date_sequence = date_sequence[:-1]
        age_sequence = age_sequence[:-1]
        code_sequence = code_sequence[:-1]
        segment_sequence = segment_sequence[:-1]
        position_sqeuence = position_sqeuence[:-1]

        processed_data[patient] = {
            "dates": date_sequence,
            "age": age_sequence,
            "codes": code_sequence,
            "position": position_sqeuence,
            "segment": segment_sequence,
        }

    for patient, sequences in processed_data.items():
        # Padding all sequences to the max length*
        for key in sequences:
            if key == "codes":
                sequences[key] += [PAD_TOKEN] * (max_length - len(sequences[key]))
            else:
                sequences[key] += [PAD_TOKEN_NS] * (max_length - len(sequences[key]))

        # Mask codes
        codes_masked, label, masked_index = random_masking(
            sequences["codes"], vocab_list, word_to_idx
        )
        sequences["codes_masked"] = codes_masked
        sequences["label"] = label
        sequences["masked_index"] = masked_index
    return processed_data


# %% Test process_data_MLM
# with open(file_path) as f:
#     data = json.load(f)

# # Build vocabulary
# vocab_list, token_to_idx = build_vocab(data)

# data = process_data_MLM(data, vocab_list, token_to_idx)
# dataset = MaskedDataset(data)
# data_loader = DataLoader(dataset, batch_size=2, shuffle=False)

# # Test data_loader
# sample = next(iter(data_loader))
# %%

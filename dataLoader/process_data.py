# %% Import libraries and initialize global variables

from torchtext.vocab import vocab
from collections import Counter

import json
from dateutil.relativedelta import relativedelta
from datetime import datetime, date

import torch
from torch.utils.data import Dataset, DataLoader

from utils import random_masking


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
        attension_mask = torch.tensor(self.data[idx]["attention_mask"])
        return codes, dates, age, position, segment, attension_mask


# %% Tokenize including dates
def process_data_MLM(
    data,
    vocab_list,
    word_to_idx,
    START_TOKEN="<CLS>",
    SEP_TOKEN="<SEP>",
    PAD_TOKEN="<PAD>",
    EMPTY_TOKEN_NS=0,
    ref_date=datetime(1900, 1, 1),
    max_length=512,
):
    """
    Function to process the data from the json file to
    Data is processed for MLM (Masked Language Model) task
    """

    max_length = 512  # maximum length of sequence - currently given by BERT
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
        date_sequence = [EMPTY_TOKEN_NS]
        age_sequence = [EMPTY_TOKEN_NS]
        code_sequence = [START_TOKEN]
        position_sqeuence = [EMPTY_TOKEN_NS]
        segment_sequence = [EMPTY_TOKEN_NS]

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
                sequences[key] += [EMPTY_TOKEN_NS] * (max_length - len(sequences[key]))
        # Attentions masks
        sequences["attention_mask"] = [1] * len(sequences["codes"]) + [0] * (
            max_length - len(sequences["codes"])
        )

        # Mask codes
        codes_masked, label, masked_index = random_masking(
            sequences["codes"], vocab_list, word_to_idx
        )

        sequences["codes_masked"] = codes_masked
        sequences["label"] = label
        sequences["masked_index"] = masked_index
    return processed_data


# # %% Test process_data_MLM
# file_path = "H:/Code/transformerEHR/syntheticData.json"
# with open(file_path) as f:
#     data = json.load(f)

# # Build vocabulary
# vocab_list, token_to_idx = build_vocab(data)

# data = process_data_MLM(data, vocab_list, token_to_idx)
# dataset = MaskedDataset(data)
# data_loader = DataLoader(dataset, batch_size=2, shuffle=False)

# # Test data_loader
# sample = next(iter(data_loader))

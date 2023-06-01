# %% Load
from torchtext.vocab import vocab
from collections import Counter

import json
from dateutil.relativedelta import relativedelta
from datetime import datetime, date

from torch.utils.data import Dataset, DataLoader

# %% global variables
file_path = "H:/Code/transformerEHR/syntheticData.json"
SEP_TOKEN = "<SEP>"
PAD_TOKEN = "<PAD>"


# %% Make vocab
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
    vocab_list = vocab(counter, specials=(SEP_TOKEN, PAD_TOKEN, "<MASK>"))

    return vocab_list


# %% Test
# v = build_vocab(file_path)
# v.lookup_indices(["Q12"])


# %% Tokenize including dates
def process_data(filename):
    """
    Function to process the data from the json file to
    """
    with open(filename) as f:
        data = json.load(f)

    max_length = 0
    processed_data = {}

    # Variables for calculating age
    today = datetime.today()

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
        date_sequence = []
        age_sequence = []
        code_sequence = []
        position_sqeuence = []
        segment_sequence = []

        position = 1
        segment = 1
        for date_list, code_list in admid_groups.values():
            date_sequence += date_list
            age_sequence += [
                str(
                    relativedelta(datetime.strptime(date, "%Y-%m-%d"), birth_date).years
                )
                for date in date_list
            ]
            code_sequence += code_list + [SEP_TOKEN]
            position_sqeuence += [position] * (len(code_list) + 1)
            segment_sequence += [segment] * (len(code_list) + 1)
            position += 1
            segment *= -1

            # Add segment date and age where seprator is added
            date_sequence += date(1900, 1, 1)
            age_sequence += -1

        # Remove the last '[SEP]' from the sequences
        date_sequence = date_sequence[:-1]
        age_sequence = age_sequence[:-1]
        code_sequence = code_sequence[:-1]

        if len(date_sequence) > max_length:
            max_length = len(date_sequence)

        processed_data[patient] = {
            "dates": date_sequence,
            "age": age_sequence,
            "codes": code_sequence,
            "position": position_sqeuence,
            "segment": segment_sequence,
        }

    # Padding all sequences to the max length*
    for patient, sequences in processed_data.items():
        for key in sequences:
            if len(sequences[key]) < max_length:
                if key == "codes":
                    sequences[key] += [PAD_TOKEN] * (max_length - len(sequences[key]))
                else:  # pad with zeros for the other sequences
                    sequences[key] += [0] * (max_length - len(sequences[key]))

    return processed_data


# Restult of process_data
# data = process_data(file_path)


# %% DataLoader
class HealthDataset(Dataset):
    def __init__(self, data):
        self.data = list(data.values())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        dates = self.data[idx]["dates"]
        age = self.data[idx]["age"]
        codes = self.data[idx]["codes"]
        position = self.data[idx]["position"]
        segment = self.data[idx]["segment"]
        return dates, age, codes, position, segment


# # Define data_loader
# vocab_list = build_vocab(file_path)
# data = process_data(file_path)
# dataset = HealthDataset(data)
# data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

# # Test data_loader
# sample = next(iter(data_loader))
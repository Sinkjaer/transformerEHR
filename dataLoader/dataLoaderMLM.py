# %% Import libraries and initialize global variables

from torchtext.vocab import vocab
from collections import Counter

import json
from dateutil.relativedelta import relativedelta
from datetime import datetime, date

import torch
from torch.utils.data import Dataset, DataLoader

from dataLoader.utils import random_masking

from tqdm import tqdm


# %% DataLoader
class MaskedDataset(Dataset):
    def __init__(self, data):
        self.data = list(data.values())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        dates = torch.tensor(self.data[idx]["dates"])
        age = torch.tensor(self.data[idx]["age"])
        masked_codes = torch.tensor(self.data[idx]["masked_codes"])
        position = torch.tensor(self.data[idx]["position"])
        segment = torch.tensor(self.data[idx]["segment"])
        attension_mask = torch.tensor(self.data[idx]["attention_mask"])
        output_labels = torch.tensor(self.data[idx]["output_labels"])
        # patient = torch.tensor(self.data[idx]["patient"])
        return (
            dates,
            age,
            masked_codes,
            position,
            segment,
            attension_mask,
            output_labels,
            # patient,
        )


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
    mask_prob=0.15,
    Azure=False,
):
    """
    Function to process the data from the json file.
    Data is processed for the MLM (Masked Language Model) task.
    """
    if Azure:
        # Azure specific name mappings
        names = {
            "event_id": "EncounterKey",
            "event_date": "Time",
            "birth_date": "BirthDate",
            "events": "Events",
            "codes": "Type",
        }
    else:
        # Non-Azure name mappings
        names = {
            "event_id": "admid",
            "event_date": "admdate",
            "birth_date": "birthdate",
            "events": "events",
            "codes": "codes",
        }

    processed_data = {}
    count = 0

    for patient, patient_data in tqdm(data.items()):
        count += 1
        if count == 10000:
            break
        # Process birth date and events
        birth_date = datetime.strptime(patient_data[names["birth_date"]], "%Y-%m-%d")
        events = patient_data[names["events"]]
        events = [
            event for event in events if event.get(names["event_date"])
        ]  # Remove empty items
        events.sort(key=lambda x: x[names["event_date"]])  # Sort events by dates

        # Group date and 'codes' with the same ID together
        admid_groups = {}
        for event in events:
            if event[names["event_id"]] in admid_groups:
                admid_groups[event[names["event_id"]]][0].append(
                    event[names["event_date"]]
                )
                admid_groups[event[names["event_id"]]][1].append(event[names["codes"]])
            else:
                admid_groups[event[names["event_id"]]] = [
                    [event[names["event_date"]]],
                    [event[names["codes"]]],
                ]

        # Initialize sequences and insert start token
        date_sequence = [EMPTY_TOKEN_NS]
        age_sequence = [EMPTY_TOKEN_NS]
        code_sequence = [START_TOKEN]
        position_sequence = [EMPTY_TOKEN_NS]
        segment_sequence = [EMPTY_TOKEN_NS]

        position = 1
        segment = 1
        total_length = 0

        for date_list, code_list in admid_groups.values():
            # Check if adding the current group exceeds the max_length
            if (
                total_length + len(code_list) + 2 > max_length
            ):  # 2 for SEP_TOKEN and potential final [SEP] after loop
                break

            # Add date and code sequences
            date_sequence += [
                (datetime.strptime(date[:10], "%Y-%m-%d") - ref_date).days
                for date in date_list
            ]
            age_sequence += [
                relativedelta(
                    datetime.strptime(date[:10], "%Y-%m-%d"), birth_date
                ).years
                + relativedelta(
                    datetime.strptime(date[:10], "%Y-%m-%d"), birth_date
                ).months
                / 12
                for date in date_list
            ]
            code_sequence += code_list + [SEP_TOKEN]
            position_sequence += [position] * (len(code_list) + 1)
            segment_sequence += [segment] * (len(code_list) + 1)

            # Update position, segment, and total length
            position += 1
            segment = position % 2
            total_length += len(code_list) + 2

        # Remove the last '[SEP]' from the sequences
        date_sequence = date_sequence[:-1]
        age_sequence = age_sequence[:-1]
        code_sequence = code_sequence[:-1]
        segment_sequence = segment_sequence[:-1]
        position_sequence = position_sequence[:-1]

        processed_data[patient] = {
            "dates": date_sequence,
            "age": age_sequence,
            "codes": code_sequence,
            "position": position_sequence,
            "segment": segment_sequence,
        }

    for patient, sequences in processed_data.items():
        # Padding sequences to the max_length
        for key in sequences:
            if key == "codes":
                sequences[key] += [PAD_TOKEN] * (max_length - len(sequences[key]))
            else:
                sequences[key] += [EMPTY_TOKEN_NS] * (max_length - len(sequences[key]))

        # Attention masks
        sequences["attention_mask"] = [1] * len(sequences["codes"]) + [0] * (
            max_length - len(sequences["codes"])
        )

        # Mask codes
        true_codes, masked_codes, output_labels = random_masking(
            sequences["codes"], vocab_list, word_to_idx, probability=mask_prob
        )

        sequences["true_codes"] = true_codes
        sequences["masked_codes"] = masked_codes
        sequences["output_labels"] = output_labels
        sequences["patient"] = int(patient)

    return processed_data


# # %% Test process_data_MLM
# file_path = "H:/Code/transformerEHR/data/syntheticData.json"
# with open(file_path) as f:
#     data = json.load(f)

# # Build vocabulary
# vocab_list, token_to_idx = build_vocab(data)

# data = process_data_MLM(data, vocab_list, token_to_idx)
# dataset = MaskedDataset(data)
# data_loader = DataLoader(dataset, batch_size=2, shuffle=False)

# # Test data_loader
# sample = next(iter(data_loader))

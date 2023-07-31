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
import random


# %% MLM data loader
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

    for patient, patient_data in tqdm(data.items(), desc="Processing data"):
        # count += 1
        # if count == 5:
        #     break
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

        # Ensure that sequence is not to large
        date_sequence = date_sequence[-max_length:]
        age_sequence = age_sequence[-max_length:]
        code_sequence = code_sequence[-max_length:]
        segment_sequence = segment_sequence[-max_length:]
        position_sequence = position_sequence[-max_length:]

        processed_data[patient] = {
            "dates": date_sequence,
            "age": age_sequence,
            "codes": code_sequence,
            "position": position_sequence,
            "segment": segment_sequence,
        }

    for patient, sequences in processed_data.items():
        # Attention masks
        sequences["attention_mask"] = [1] * len(sequences["codes"]) + [0] * (
            max_length - len(sequences["codes"])
        )
        # Padding sequences to the max_length
        for key in sequences:
            if key == "codes":
                sequences[key] += [PAD_TOKEN] * (max_length - len(sequences[key]))
            else:
                sequences[key] += [EMPTY_TOKEN_NS] * (max_length - len(sequences[key]))

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


# %% Coercion risk data loader
class CoercionRiskDataset(Dataset):
    def __init__(self, data):
        self.data = list(data.values())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        dates = torch.tensor(self.data[idx]["dates"])
        age = torch.tensor(self.data[idx]["age"])
        input_sequence = torch.tensor(self.data[idx]["input_sequence"])
        position = torch.tensor(self.data[idx]["position"])
        segment = torch.tensor(self.data[idx]["segment"])
        attension_mask = torch.tensor(self.data[idx]["attention_mask"])
        classification_labels = torch.tensor(self.data[idx]["classification_labels"])
        return (
            dates,
            age,
            input_sequence,
            position,
            segment,
            attension_mask,
            classification_labels,
        )


def process_data_CoercionRisk(
    data,
    vocab_list,
    word_to_idx,
    START_TOKEN="<CLS>",
    SEP_TOKEN="<SEP>",
    PAD_TOKEN="<PAD>",
    EMPTY_TOKEN_NS=0,
    ref_date=datetime(1900, 1, 1),
    max_length=512,
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

    for patient, patient_data in tqdm(data.items()):
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
                    [event[names["event_id"]]],
                ]

        # Initialize sequences and insert start token
        date_sequence = [EMPTY_TOKEN_NS]
        age_sequence = [EMPTY_TOKEN_NS]
        code_sequence = [START_TOKEN]
        position_sequence = [EMPTY_TOKEN_NS]
        segment_sequence = [EMPTY_TOKEN_NS]
        event_id_sequence = [EMPTY_TOKEN_NS]

        position = 1
        segment = 1
        total_length = 0
        coercion_label = -1

        for date_list, code_list, event_id in admid_groups.values():
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
            event_id_sequence += [event_id] * (len(code_list) + 1)

            # Update position, segment, and total length
            position += 1
            segment = position % 2
            total_length += len(code_list) + 2

        # Get coercion and psych admission events
        coercion = [
            index
            for index, value in enumerate(code_sequence)
            if value == "coercion_2_start" or value == "coercion_3_start"
        ]
        psych_admission = [
            index
            for index, value in enumerate(code_sequence)
            if value == "psych_admission"
        ]

        if len(coercion) > 0:
            # sample a random encounter with coercion and get the first index where the coercion is present in
            if random.random() < 1:
                coercion_label = 1

                sample_index = random.sample(coercion, 1)[0]

                # Trim sequence
                code_sequence = code_sequence[:sample_index]
                date_sequence = date_sequence[:sample_index]
                segment_sequence = segment_sequence[:sample_index]
                age_sequence = age_sequence[:sample_index]
                position_sequence = position_sequence[:sample_index]

            # Sample random psych admission
            else:
                coercion_label = 0
                sample_index = random.sample(psych_admission, 1)[0]
                # Trim sequence
                code_sequence = code_sequence[:sample_index]
                date_sequence = date_sequence[:sample_index]
                segment_sequence = segment_sequence[:sample_index]
                age_sequence = age_sequence[:sample_index]
                position_sequence = position_sequence[:sample_index]
        else:
            coercion_label = 0
            sample_index = random.sample(psych_admission, 1)[0]

            # Trim sequence
            code_sequence = code_sequence[:sample_index]
            date_sequence = date_sequence[:sample_index]
            segment_sequence = segment_sequence[:sample_index]
            age_sequence = age_sequence[:sample_index]
            position_sequence = position_sequence[:sample_index]

        # Remove the last '[SEP]' from the sequences
        date_sequence = date_sequence[:-1]
        age_sequence = age_sequence[:-1]
        code_sequence = code_sequence[:-1]
        segment_sequence = segment_sequence[:-1]
        position_sequence = position_sequence[:-1]

        # Ensure that sequence is not to large, only keep the latest events
        date_sequence = date_sequence[-max_length:]
        age_sequence = age_sequence[-max_length:]
        code_sequence = code_sequence[-max_length:]
        segment_sequence = segment_sequence[-max_length:]
        position_sequence = position_sequence[-max_length:]

        processed_data[patient] = {
            "dates": date_sequence,
            "age": age_sequence,
            "codes": code_sequence,
            "position": position_sequence,
            "segment": segment_sequence,
            "classification_labels": coercion_label,
        }

    for patient, sequences in processed_data.items():
        # Attention masks
        sequences["attention_mask"] = [1] * len(sequences["codes"]) + [0] * (
            max_length - len(sequences["codes"])
        )

        # Padding sequences to the max_length
        for key in sequences:
            if key == "codes":
                sequences[key] += [PAD_TOKEN] * (max_length - len(sequences[key]))
            elif key != "classification_labels":
                sequences[key] += [EMPTY_TOKEN_NS] * (max_length - len(sequences[key]))

        # Tokenize
        sequences["input_sequence"] = [vocab_list[code] for code in sequences["codes"]]

    return processed_data

# %% Import libraries and initialize global variables

from torchtext.vocab import vocab
from collections import Counter

import json
from dateutil.relativedelta import relativedelta
from datetime import datetime, date

import torch
from torch.utils.data import Dataset, DataLoader

from dataLoader.utils import random_masking

import random
import numpy as np


# %% MLM data loader
class MaskedDataset(Dataset):
    def __init__(self, data, vocab_list, word_to_idx):
        self.data = data
        self.vocab_list = vocab_list
        self.word_to_idx = word_to_idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        patient, patient_data = list(self.data.items())[idx]
        processed_patient_data = process_data_MLM(
            patient_data, patient, self.vocab_list, self.word_to_idx
        )

        dates = torch.tensor(processed_patient_data["dates"])
        age = torch.tensor(processed_patient_data["age"])
        masked_codes = torch.tensor(processed_patient_data["masked_codes"])
        position = torch.tensor(processed_patient_data["position"])
        segment = torch.tensor(processed_patient_data["segment"])
        attention_mask = torch.tensor(processed_patient_data["attention_mask"])
        output_labels = torch.tensor(processed_patient_data["output_labels"])
        return (
            dates,
            age,
            masked_codes,
            position,
            segment,
            attention_mask,
            output_labels,
        )


def process_data_MLM(
    patient_data,
    patient,
    vocab_list,
    word_to_idx,
    START_TOKEN="<CLS>",
    SEP_TOKEN="<SEP>",
    PAD_TOKEN="<PAD>",
    EMPTY_TOKEN_NS=0,
    ref_date=datetime(1900, 1, 1),
    max_length=512,
    mask_prob=0.15,
):
    """
    Function to process the data from the json file.
    Data is processed for the MLM (Masked Language Model) task.
    """

    names = {
        "event_id": "EncounterKey",
        "event_date": "Time",
        "birth_date": "BirthDate",
        "events": "Events",
        "codes": "Type",
    }

    # remove empty events and sort events by date
    events = patient_data[names["events"]]
    events = [
        event for event in events if event.get(names["codes"])
    ]  # Remove empty items
    events.sort(
        key=lambda x: (x[names["event_date"]], x[names["codes"]])
    )  # Sort events by dates and names

    birth_date = patient_data[names["birth_date"]]
    ref_date = events[-1][names["event_date"]]  # Set reference date to last event date
    minutes_per_year = 60 * 24 * 365.25  # Minutes per year
    minutes_per_day = 60 * 24  # Minutes per day

    # Group date and 'codes' with the same ID together
    admid_groups = {}
    for event in events:
        if event[names["event_id"]] in admid_groups:
            admid_groups[event[names["event_id"]]][0].append(event[names["event_date"]])
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
        date_sequence.extend(
            [int((ref_date - date) // minutes_per_day) for date in date_list]
        )
        age_sequence.extend(
            [max(0, int((date - birth_date) // minutes_per_year)) for date in date_list]
        )

        code_sequence.extend(code_list + [SEP_TOKEN])
        position_sequence.extend([position] * (len(code_list) + 1))
        segment_sequence.extend([segment] * (len(code_list) + 1))

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

    # Ensure that sequence is not too large
    if total_length > max_length:
        date_sequence = date_sequence[-max_length:]
        age_sequence = age_sequence[-max_length:]
        code_sequence = code_sequence[-max_length:]
        segment_sequence = segment_sequence[-max_length:]
        position_sequence = position_sequence[-max_length:]

        # Remove entries prior to the first '[SEP]'
        index = code_sequence.index(SEP_TOKEN)
        date_sequence = date_sequence[index:]
        age_sequence = age_sequence[index:]
        code_sequence = code_sequence[index:]
        segment_sequence = segment_sequence[index:]
        position_sequence = position_sequence[index:]

        # Scale position_sequence to the new length
        min_pos = position_sequence[0] - 1
        position_sequence = [i - min_pos for i in position_sequence]

    processed_data = {
        "dates": date_sequence,
        "age": age_sequence,
        "codes": code_sequence,
        "position": position_sequence,
        "segment": segment_sequence,
    }

    # Attention masks
    processed_data["attention_mask"] = [1] * len(processed_data["codes"]) + [0] * (
        max_length - len(processed_data["codes"])
    )
    # Padding sequences to the max_length
    for key in processed_data:
        if key == "codes":
            processed_data[key] += [PAD_TOKEN] * (max_length - len(processed_data[key]))
        else:
            processed_data[key] += [EMPTY_TOKEN_NS] * (
                max_length - len(processed_data[key])
            )

    # Mask codes
    true_codes, masked_codes, output_labels = random_masking(
        processed_data["codes"], vocab_list, word_to_idx, probability=mask_prob
    )

    processed_data["true_codes"] = true_codes
    processed_data["masked_codes"] = masked_codes
    processed_data["output_labels"] = output_labels
    processed_data["patient"] = int(patient)

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
    def __init__(self, data, vocab_list, word_to_idx):
        self.data = data
        self.vocab_list = vocab_list
        self.word_to_idx = word_to_idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        patient, patient_data = list(self.data.items())[idx]
        processed_patient_data = process_data_CoercionRisk(
            patient_data, patient, self.vocab_list, self.word_to_idx
        )
        dates = torch.tensor(processed_patient_data["dates"])
        age = torch.tensor(processed_patient_data["age"])
        input_sequence = torch.tensor(processed_patient_data["input_sequence"])
        position = torch.tensor(processed_patient_data["position"])
        segment = torch.tensor(processed_patient_data["segment"])
        attention_mask = torch.tensor(processed_patient_data["attention_mask"])
        classification_labels = torch.tensor(
            processed_patient_data["classification_labels"]
        )
        return (
            dates,
            age,
            input_sequence,
            position,
            segment,
            attention_mask,
            classification_labels,
        )


def process_data_CoercionRisk(
    patient_data,
    patient,
    vocab_list,
    word_to_idx,
    START_TOKEN="<CLS>",
    SEP_TOKEN="<SEP>",
    PAD_TOKEN="<PAD>",
    EMPTY_TOKEN_NS=0,
    ref_date=datetime(1900, 1, 1),
    max_length=512,
    mask_prob=0.15,
):
    """
    Function to process the data from the json file.
    Data is processed for the MLM (Masked Language Model) task.
    """

    names = {
        "event_id": "EncounterKey",
        "event_date": "Time",
        "birth_date": "BirthDate",
        "events": "Events",
        "codes": "Type",
    }

    # Get index of ceroction and psych admission
    coercion = list(
        set(patient_data["Coercion_2_idx"] + patient_data["Coercion_3_idx"])
    )
    psych_admission = patient_data["Psych_admission_idx"]

    psych_admission_no_coercion = [
        idx for idx in psych_admission if idx not in coercion
    ]

    # sample event
    if len(coercion) > 0:
        if (random.random() < 0.8) or (len(psych_admission_no_coercion) == 0):
            sample_index = random.sample(coercion, 1)[0]
            coercion_label = 1
        else:
            sample_index = random.sample(psych_admission_no_coercion, 1)[0]
            coercion_label = 0
    else:
        sample_index = random.sample(psych_admission, 1)[0]
        coercion_label = 0

    # replace sampled index with sample token
    patient_data["Events"][sample_index]["Type"] = "sample"

    # Process birth date and events
    events = patient_data[names["events"]]
    events = [
        event for event in events if event.get(names["event_date"])
    ]  # Remove empty items
    events.sort(
        key=lambda x: (x[names["event_date"]], x[names["codes"]])
    )  # Sort events by dates and names

    birth_date = patient_data[names["birth_date"]]
    ref_date = patient_data[names["events"]][sample_index][
        "Time"
    ]  # Set reference date to the sample
    minutes_per_year = 60 * 24 * 365.25  # Minutes per year
    minutes_per_day = 60 * 24  # Minutes per day

    # Group date and 'codes' with the same ID together
    admid_groups = {}
    for event in events:
        if event[names["event_id"]] in admid_groups:
            admid_groups[event[names["event_id"]]][0].append(event[names["event_date"]])
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
        # Stop if sample token is reached
        if "sample" in code_list:
            break
        # Add date and code sequences
        date_sequence.extend(
            [int((ref_date - date) // minutes_per_day) for date in date_list]
        )
        age_sequence.extend(
            [max(0, int((date - birth_date) // minutes_per_year)) for date in date_list]
        )

        code_sequence.extend(code_list + [SEP_TOKEN])
        position_sequence.extend([position] * (len(code_list) + 1))
        segment_sequence.extend([segment] * (len(code_list) + 1))

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

    # Remove events that happens after the sample
    if any(np.array(date_sequence) < 0):
        negative_date_idx = list(np.where(np.array(date_sequence) < 0)[0])
        date_sequence = [
            date
            for idx, date in enumerate(date_sequence)
            if idx not in negative_date_idx
        ]
        age_sequence = [
            age for idx, age in enumerate(age_sequence) if idx not in negative_date_idx
        ]
        code_sequence = [
            code
            for idx, code in enumerate(code_sequence)
            if idx not in negative_date_idx
        ]
        segment_sequence = [
            segment
            for idx, segment in enumerate(segment_sequence)
            if idx not in negative_date_idx
        ]
        position_sequence = [
            position
            for idx, position in enumerate(position_sequence)
            if idx not in negative_date_idx
        ]

    # Ensure that sequence is not to large
    if len(code_sequence) > max_length:
        date_sequence = date_sequence[-max_length:]
        age_sequence = age_sequence[-max_length:]
        code_sequence = code_sequence[-max_length:]
        segment_sequence = segment_sequence[-max_length:]
        position_sequence = position_sequence[-max_length:]

        # Remove entries prior to the first '[SEP]'
        try:  # Is used if the sequence don't contain a sep token
            index = code_sequence.index(SEP_TOKEN)
        except:
            print(code_sequence)
        date_sequence = date_sequence[index:]
        age_sequence = age_sequence[index:]
        code_sequence = code_sequence[index:]
        segment_sequence = segment_sequence[index:]
        position_sequence = position_sequence[index:]

        # Scale position_sequence to the new length
        min_pos = position_sequence[0] - 1
        position_sequence = [i - min_pos for i in position_sequence]

    processed_data = {
        "dates": date_sequence,
        "age": age_sequence,
        "codes": code_sequence,
        "position": position_sequence,
        "segment": segment_sequence,
    }

    # Attention masks
    processed_data["attention_mask"] = [1] * len(processed_data["codes"]) + [0] * (
        max_length - len(processed_data["codes"])
    )
    # Padding sequences to the max_length
    for key in processed_data:
        if key == "codes":
            processed_data[key] += [PAD_TOKEN] * (max_length - len(processed_data[key]))
        else:
            processed_data[key] += [EMPTY_TOKEN_NS] * (
                max_length - len(processed_data[key])
            )

    # Tokenize
    processed_data["input_sequence"] = [
        vocab_list[code] for code in processed_data["codes"]
    ]
    processed_data["classification_labels"] = [coercion_label]

    return processed_data


# %%

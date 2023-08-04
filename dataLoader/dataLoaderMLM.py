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

    # Process birth date and events
    birth_date = datetime.strptime(patient_data[names["birth_date"]], "%Y-%m-%d")
    events = patient_data[names["events"]]
    events = [
        event for event in events if event.get(names["event_date"])
    ]  # Remove empty items
    # TODO This does not sort the events correctly as they are strings
    events.sort(key=lambda x: x[names["event_date"]])  # Sort events by dates

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
        # TODO use timestaps instead of dates
        date_list = [datetime.strptime(date[:10], "%Y-%m-%d") for date in date_list]
        date_sequence.extend([(date - ref_date).days for date in date_list])
        # patient can for som reason be younger than 0
        age_sequence.extend(
            [max(0, relativedelta(date, birth_date).years) for date in date_list]
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

    # Process birth date and events
    birth_date = datetime.strptime(patient_data[names["birth_date"]], "%Y-%m-%d")
    events = patient_data[names["events"]]
    events = [
        event for event in events if event.get(names["event_date"])
    ]  # Remove empty items
    # TODO This does not sort the events correctly as they are strings
    events.sort(key=lambda x: x[names["event_date"]])  # Sort events by dates

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
        # TODO use timestaps instead of dates
        date_list = [datetime.strptime(date[:10], "%Y-%m-%d") for date in date_list]
        date_sequence.extend([(date - ref_date).days for date in date_list])
        # patient can for som reason be younger than 0
        age_sequence.extend(
            [max(0, relativedelta(date, birth_date).years) for date in date_list]
        )

        code_sequence.extend(code_list + [SEP_TOKEN])
        position_sequence.extend([position] * (len(code_list) + 1))
        segment_sequence.extend([segment] * (len(code_list) + 1))

        # Update position, segment, and total length
        position += 1
        segment = position % 2
        total_length += len(code_list) + 2

    coercion_label = -1
    # Remove the last '[SEP]' from the sequences
    date_sequence = date_sequence[:-1]
    age_sequence = age_sequence[:-1]
    code_sequence = code_sequence[:-1]
    segment_sequence = segment_sequence[:-1]
    position_sequence = position_sequence[:-1]

    # Get coercion and psych admission events
    coercion = [
        index
        for index, value in enumerate(code_sequence)
        if value == "coercion_2_start" or value == "coercion_3_start"
    ]
    psych_admission = [
        index for index, value in enumerate(code_sequence) if value == "psych_admission"
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

    # Ensure that sequence is not to large
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

    # Tokenize
    processed_data["input_sequence"] = [
        vocab_list[code] for code in processed_data["codes"]
    ]
    processed_data["classification_labels"] = [coercion_label]

    return processed_data


# %%

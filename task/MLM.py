# %%
import sys

sys.path.insert(0, "../")

import os

# Get the current working directory
current_directory = os.getcwd()

# Print the current working directory
print(current_directory)

Azure = True

# %%
from common.common import create_folder
from common.pytorch import load_model
from dataLoader.build_vocab import load_vocab, build_vocab
import pytorch_pretrained_bert as Bert
from common.common import load_obj
from dataLoader.dataLoaderMLM import MaskedDataset, process_data_MLM
from torch.utils.data import DataLoader
import pandas as pd
from model.transModel import BertForMaskedLM
from model.optimiser import adam
import sklearn.metrics as skm
import numpy as np
import torch
import time
import torch.nn as nn
import os
import json


# %%
class BertConfig(Bert.modeling.BertConfig):
    def __init__(self, config):
        super(BertConfig, self).__init__(
            vocab_size_or_config_json_file=config.get("vocab_size"),
            hidden_size=config["hidden_size"],
            num_hidden_layers=config.get("num_hidden_layers"),
            num_attention_heads=config.get("num_attention_heads"),
            intermediate_size=config.get("intermediate_size"),
            hidden_act=config.get("hidden_act"),
            hidden_dropout_prob=config.get("hidden_dropout_prob"),
            attention_probs_dropout_prob=config.get("attention_probs_dropout_prob"),
            max_position_embeddings=config.get("max_position_embedding"),
            initializer_range=config.get("initializer_range"),
        )
        self.seg_vocab_size = config.get("seg_vocab_size")
        self.age_vocab_size = config.get("age_vocab_size")


# %%
if Azure:
    file_config = {
        "vocab": "../dataloader/vocab.txt",  # vocabulary idx2token, token2idx
        "data_train": "../../EHR_data/data/pre_train_training_set.json",  # formated data
        "data_val": "../../EHR_data/data/pre_train_validation_set.json",  # formated data
        "model_path": "MLM/model1",  # where to save model
        "model_name": "model",  # model name
        "file_name": "log.txt",  # log path
        "use_cuda": True,
        "device": "cuda:0",
    }
else:
    file_config = {
        "vocab": "/Users/mikkelsinkjaer/Library/Mobile Documents/com~apple~CloudDocs/transEHR/Code/transformerEHR/data/vocab.txt",  # vocabulary idx2token, token2idx
        "data_train": "/Users/mikkelsinkjaer/Library/Mobile Documents/com~apple~CloudDocs/transEHR/Code/transformerEHR/data/syntheticData_train.json",
        "data_val": "/Users/mikkelsinkjaer/Library/Mobile Documents/com~apple~CloudDocs/transEHR/Code/transformerEHR/data/syntheticData_val.json",
        "model_path": "MLM/model1",  # where to save model
        "model_name": "model",  # model name
        "file_name": "log.txt",  # log path
        "use_cuda": False,
        "device": "cpu",
    }
create_folder(file_config["model_path"])

global_params = {"max_seq_len": 512, "gradient_accumulation_steps": 1}

optim_param = {"lr": 3e-6, "warmup_proportion": 0.1, "weight_decay": 0.01}

train_params = {
    "batch_size": 128,
    "use_cuda": file_config["use_cuda"],
    "max_len_seq": global_params["max_seq_len"],
    "device": file_config["device"],
}

# vocab_list, word_to_idx = load_vocab(file_config["vocab"])

with open(file_config["data_train"]) as f:
    data_json = json.load(f)

# Build vocab
vocab_list, word_to_idx = build_vocab(data_json, Azure=Azure)

# %%
# Data loader
data = process_data_MLM(data_json, vocab_list, word_to_idx, mask_prob=0.20, Azure=Azure)
masked_data = MaskedDataset(data)
sample = next(iter(masked_data))

# %%
trainload = DataLoader(
    dataset=masked_data,
    batch_size=train_params["batch_size"],
    shuffle=True,
    num_workers=16,
)

# Data loader for validation set
with open(file_config["data_val"]) as f:
    data_val_json = json.load(f)

data_val = process_data_MLM(
    data_val_json, vocab_list, word_to_idx, mask_prob=0.20, Azure=Azure
)
masked_data_val = MaskedDataset(data_val)

valload = DataLoader(
    dataset=masked_data_val,
    batch_size=train_params["batch_size"],
    shuffle=False,
)

model_config = {
    "vocab_size": len(vocab_list),  # number of disease + symbols for word embedding
    "hidden_size": 288,  # word embedding and seg embedding hidden size
    "seg_vocab_size": 2,  # number of vocab for seg embedding
    "age_vocab_size": 144,  # number of vocab for age embedding
    "max_position_embedding": train_params["max_len_seq"],  # maximum number of tokens
    "hidden_dropout_prob": 0.1,  # dropout rate
    "num_hidden_layers": 6,  # number of multi-head attention layers required
    "num_attention_heads": 12,  # number of attention heads
    "attention_probs_dropout_prob": 0.1,  # multi-head attention dropout rate
    "intermediate_size": 512,  # the size of the "intermediate" layer in the transformer encoder
    "hidden_act": "gelu",  # The non-linear activation function in the encoder and the pooler "gelu", 'relu', 'swish' are supported
    "initializer_range": 0.02,  # parameter weight initializer range
}

conf = BertConfig(model_config)
model = BertForMaskedLM(conf)

model = model.to(train_params["device"])
optim = adam(params=list(model.named_parameters()), config=optim_param)


def cal_acc(label, pred):
    label = label.cpu().numpy()
    ind = np.where(label != -1)[0]
    truepred = pred.detach().cpu().numpy()
    truepred = truepred[ind]
    truelabel = label[ind]
    truepred = torch.from_numpy(truepred)
    truepred = torch.nn.functional.log_softmax(truepred, dim=1)
    outs = [np.argmax(pred_x) for pred_x in truepred.numpy()]
    # Consider only the non-padded tokens
    outs = np.array(outs)[truelabel != 3]
    truelabel = truelabel[truelabel != 3]
    precision = skm.precision_score(truelabel, outs, average="micro")
    return precision


def train(e, loader):
    tr_loss = 0
    temp_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    cnt = 0
    start = time.time()

    for step, batch in enumerate(loader):
        cnt += 1
        batch = tuple(t.to(train_params["device"]) for t in batch)

        (
            dates_ids,
            age_ids,
            input_ids,
            posi_ids,
            segment_ids,
            attMask,
            output_labels,
        ) = batch
        loss, pred, label = model(
            input_ids,
            dates_ids=dates_ids,
            age_ids=age_ids,
            seg_ids=segment_ids,
            posi_ids=posi_ids,
            attention_mask=attMask,
            masked_lm_labels=output_labels,
        )
        if global_params["gradient_accumulation_steps"] > 1:
            loss = loss / global_params["gradient_accumulation_steps"]
        loss.backward()

        temp_loss += loss.item()
        tr_loss += loss.item()

        nb_tr_examples += input_ids.size(0)
        nb_tr_steps += 1

        if step % 100 == 0:
            print(
                "epoch: {}\t| cnt: {}\t|Loss: {}\t| precision: {:.4f}\t| time: {:.2f}".format(
                    e,
                    cnt,
                    temp_loss / 100,
                    cal_acc(label, pred),
                    time.time() - start,
                )
            )
            temp_loss = 0
            start = time.time()

        if (step + 1) % global_params["gradient_accumulation_steps"] == 0:
            optim.step()
            optim.zero_grad()

    print("** ** * Saving fine - tuned model ** ** * ")
    model_to_save = (
        model.module if hasattr(model, "module") else model
    )  # Only save the model it-self
    create_folder(file_config["model_path"])
    output_model_file = os.path.join(
        file_config["model_path"], file_config["model_name"]
    )

    torch.save(model_to_save.state_dict(), output_model_file)

    cost = time.time() - start
    return (
        tr_loss / nb_tr_examples,
        cost,
    )  # Scale the loss by number of training examples


def validation(loader):
    model.eval()  # Set model to evaluation mode
    total_acc = 0.0
    total_loss = 0.0
    total_count = 0
    nb_val_examples = 0
    with torch.no_grad():
        for batch in loader:
            batch = tuple(t.to(train_params["device"]) for t in batch)

            (
                dates_ids,
                age_ids,
                input_ids,
                posi_ids,
                segment_ids,
                attMask,
                output_labels,
            ) = batch
            loss, pred, label = model(
                input_ids,
                dates_ids=dates_ids,
                age_ids=age_ids,
                seg_ids=segment_ids,
                posi_ids=posi_ids,
                attention_mask=attMask,
                masked_lm_labels=output_labels,
            )

            total_acc += cal_acc(label, pred)
            total_loss += loss.item()
            total_count += 1
            nb_val_examples += input_ids.size(0)

    model.train()  # Set model back to train mode
    return total_loss / nb_val_examples, total_acc / total_count


f = open(os.path.join(file_config["model_path"], file_config["file_name"]), "w")
f.write("{}\t{}\t{}\t{}\t{}\n".format("epoch", "loss", "time", "val_loss", "val_acc"))
for e in range(5):
    loss, time_cost = train(e, trainload)
    loss = loss / 1  # data_len
    val_loss, val_acc = validation(valload)  # Calculate validation loss and accuracy
    print(f"Validation loss at epoch {e} is {val_loss}, accuracy is {val_acc}")
    f.write(
        "{}\t{}\t{}\t{}\t{}\n".format(e, loss, time_cost, val_loss, val_acc)
    )  # Log validation loss and accuracy
f.close()

# %%

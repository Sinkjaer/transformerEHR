# %%
import sys

sys.path.insert(0, "../")

import os

# Get the current working directory
current_directory = os.getcwd()

# Print the current working directory
print(current_directory)

Azure = False

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
import sklearn
from sklearn.preprocessing import MultiLabelBinarizer


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
        "data_val": "../../EHR_data/data/pre_train_validation.json",  # formated data
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
        "model_path": "MLM/model1/",  # where to save model
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
    shuffle=False,
    # num_workers=1,
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


def load_model(path, model):
    # load pretrained model and update weights
    pretrained_dict = torch.load(path)
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)
    return model


model = load_model(
    os.path.join(file_config["model_path"], file_config["model_name"]), model
)

model = model.to(train_params["device"])
optim = adam(params=list(model.named_parameters()), config=optim_param)


# Metrics
def precision(logits, label):
    sig = nn.Sigmoid()
    output = sig(logits)
    label, output = label.cpu(), output.detach().cpu()
    tempprc = sklearn.metrics.average_precision_score(
        label.numpy(), output.numpy(), average="samples"
    )
    return tempprc, output, label


def precision_test(logits, label):
    sig = nn.Sigmoid()
    output = sig(logits)
    tempprc = sklearn.metrics.average_precision_score(
        label.numpy(), output.numpy(), average="samples"
    )
    roc = sklearn.metrics.roc_auc_score(
        label.numpy(), output.numpy(), average="samples"
    )
    return (
        tempprc,
        roc,
        output,
        label,
    )


def train(e):
    model.train()
    tr_loss = 0
    temp_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    cnt = 0
    for step, batch in enumerate(trainload):
        cnt += 1
        age_ids, input_ids, posi_ids, segment_ids, attMask, targets, _ = batch

        dates_ids, age_ids, input_ids, posi_ids, segment_ids, attMask, targets = batch

        # dates_ids = dates_ids.to(global_params['device'])
        input_ids = input_ids.to(global_params["device"])
        posi_ids = posi_ids.to(global_params["device"])
        segment_ids = segment_ids.to(global_params["device"])
        attMask = attMask.to(global_params["device"])
        targets = targets.to(global_params["device"])

        loss, logits = model(
            input_ids,
            age_ids,
            segment_ids,
            posi_ids,
            attention_mask=attMask,
            labels=targets,
        )

        if global_params["gradient_accumulation_steps"] > 1:
            loss = loss / global_params["gradient_accumulation_steps"]
        loss.backward()

        temp_loss += loss.item()
        tr_loss += loss.item()
        nb_tr_examples += input_ids.size(0)
        nb_tr_steps += 1

        if step % 500 == 0:
            prec, a, b = precision(logits, targets)
            print(
                "epoch: {}\t| Cnt: {}\t| Loss: {}\t| precision: {}".format(
                    e, cnt, temp_loss / 500, prec
                )
            )
            temp_loss = 0

        if (step + 1) % global_params["gradient_accumulation_steps"] == 0:
            optim.step()
            optim.zero_grad()


def evaluation():
    model.eval()
    y = []
    y_label = []
    tr_loss = 0
    for step, batch in enumerate(testload):
        model.eval()
        dates_ids, age_ids, input_ids, posi_ids, segment_ids, attMask, targets = batch

        # dates_ids = dates_ids.to(global_params['device'])
        age_ids = age_ids.to(global_params["device"])
        input_ids = input_ids.to(global_params["device"])
        posi_ids = posi_ids.to(global_params["device"])
        segment_ids = segment_ids.to(global_params["device"])
        attMask = attMask.to(global_params["device"])
        targets = targets.to(global_params["device"])

        with torch.no_grad():
            loss, logits = model(
                input_ids,
                age_ids,
                segment_ids,
                posi_ids,
                attention_mask=attMask,
                labels=targets,
            )
        logits = logits.cpu()
        targets = targets.cpu()

        tr_loss += loss.item()

        y_label.append(targets)
        y.append(logits)

    y_label = torch.cat(y_label, dim=0)
    y = torch.cat(y, dim=0)

    aps, roc, output, label = precision_test(y, y_label)
    return aps, roc, tr_loss


best_pre = 0.0
for e in range(50):
    train(e)
    aps, roc, test_loss = evaluation()
    if aps > best_pre:
        # Save a trained model
        print("** ** * Saving fine - tuned model ** ** * ")
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Only save the model it-self
        output_model_file = os.path.join(
            global_params["output_dir"], global_params["best_name"]
        )
        create_folder(global_params["output_dir"])

        torch.save(model_to_save.state_dict(), output_model_file)
        best_pre = aps
    print("aps : {}".format(aps))

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
from dataLoader.dataLoaderMLM import CoercionRiskDataset, process_data_CoercionRisk
from torch.utils.data import DataLoader
import pandas as pd
from model.transModel import BertForMultiLabelPrediction
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
from tqdm import tqdm


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


if Azure:
    file_config = {
        "data_train": "../../EHR_data/data/fine_tune_training_set.json",  # formated data
        "data_val": "../../EHR_data/data/fine_tune_validation_set.json",  # formated data
        "model_path": "MLM/azure_1/",  # where to save model
        "prediction_model_name": "pred",  # Name of the prediction model
        "model_name": "model",  # model name
        "vocab": "vocab.txt",  # vocabulary idx2token, token2idx
        "file_name": "log_pred.txt",  # log path
        "use_cuda": True,
        "device": "cuda:0",
    }
else:
    file_config = {
        "data_train": "/Users/mikkelsinkjaer/data/data.json",
        "data_val": "/Users/mikkelsinkjaer/data/data.json",
        "model_path": "MLM/local_1",  # where to save model
        "model_name": "model",  # model name
        "prediction_model_name": "pred_model",  # Name of the prediction model
        "vocab": "vocab.txt",  # vocabulary idx2token, token2idx
        "file_name": "log.txt",  # log path
        "use_cuda": False,
        "device": "cpu",
    }
create_folder(file_config["model_path"])

global_params = {"max_seq_len": 512, "gradient_accumulation_steps": 1}

optim_param = {"lr": 3e-6, "warmup_proportion": 0.1, "weight_decay": 0.01}

train_params = {
    "batch_size": 32,
    "use_cuda": file_config["use_cuda"],
    "max_len_seq": global_params["max_seq_len"],
    "device": file_config["device"],
}

# Load vocab
vocab_path = os.path.join(file_config["model_path"], file_config["vocab"])
vocab_list, word_to_idx = load_vocab(file_path=vocab_path)

print("size of vocab: ", len(vocab_list))

# %%
# TODO - use new dataloader
# Data loader
with open(file_config["data_train"]) as f:
    data_train_json = json.load(f)
data_train = CoercionRiskDataset(data_train_json, vocab_list, word_to_idx)
sample = next(iter(data_train))
# %%
trainload = DataLoader(
    dataset=data_train,
    batch_size=train_params["batch_size"],
    shuffle=False,
    pin_memory=True,
    num_workers=6,
)

# Data loader for validation set
with open(file_config["data_val"]) as f:
    data_val_json = json.load(f)

data_val = CoercionRiskDataset(data_val_json, vocab_list, word_to_idx)

valload = DataLoader(
    dataset=data_val,
    batch_size=train_params["batch_size"],
    shuffle=False,
    pin_memory=True,
    num_workers=6,
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
feature_dict = {"word": True, "seg": True, "age": True, "position": True}

conf = BertConfig(model_config)
model = BertForMultiLabelPrediction(config=conf, num_labels=1)


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
    label, output = label.cpu(), output.detach().cpu()
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


def train(e, loader):
    tr_loss = 0
    temp_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    start = time.time()

    for step, batch in enumerate(tqdm(loader, desc="training")):
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

        loss, logits = model(  # TODO use prediction model
            input_ids,
            dates_ids=dates_ids,
            age_ids=age_ids,
            seg_ids=segment_ids,
            posi_ids=posi_ids,
            attention_mask=attMask,
            labels=output_labels,
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
                    step + 1,
                    temp_loss / 100,
                    precision(logits, output_labels)[0],
                    time.time() - start,
                )
            )
            temp_loss = 0
            start = time.time()

        if (step + 1) % global_params["gradient_accumulation_steps"] == 0:
            optim.step()
            optim.zero_grad()

        if step == 100:
            break

    print("** ** * Saving fine - tuned model ** ** * ")
    model_to_save = (
        model.module if hasattr(model, "module") else model
    )  # Only save the model it-self
    create_folder(file_config["model_path"])
    output_model_file = os.path.join(
        file_config["model_path"], file_config["prediction_model_name"]
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
        count = 0
        for batch in tqdm(loader, desc="Validation"):
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
            loss, logits = model(
                input_ids,
                dates_ids=dates_ids,
                age_ids=age_ids,
                seg_ids=segment_ids,
                posi_ids=posi_ids,
                attention_mask=attMask,
                labels=output_labels,
            )

            total_acc += precision_test(logits, output_labels)[0]
            total_loss += loss.item()
            total_count += 1
            nb_val_examples += input_ids.size(0)

            count += 1
            if count == 10:
                break

    model.train()  # Set model back to train mode
    return total_loss / nb_val_examples, total_acc / total_count


f = open(os.path.join(file_config["model_path"], file_config["file_name"]), "w")
f.write("{}\t{}\t{}\t{}\t{}\n".format("epoch", "loss", "time", "val_loss", "val_acc"))
for e in range(100):
    loss, time_cost = train(e, trainload)
    loss = loss / 1  # data_len
    val_loss, val_acc = validation(valload)  # Calculate validation loss and accuracy
    print(f"Validation loss at epoch {e} is {val_loss}, accuracy is {val_acc}")
    f.write(
        "{}\t{}\t{}\t{}\t{}\n".format(e, loss, time_cost, val_loss, val_acc)
    )  # Log validation loss and accuracy
f.close()

# %%

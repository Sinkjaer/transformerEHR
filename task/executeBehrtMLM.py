# %%

import sys

sys.path.insert(0, "../")
import lightning.pytorch as pl
from common.common import create_folder
from dataLoader.build_vocab import build_vocab
import pytorch_pretrained_bert as Bert
from dataLoader.dataLoaderMLM import MaskedDataset
from model.behrt import BertModel, BertMLM
from torch.utils.data import DataLoader
import json
import os
from lightning.pytorch.loggers import NeptuneLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.tuner import Tuner

Azure = True

# # Initialize Neptune
name_experiment = "MLM_model"
# neptune_logger = NeptuneLogger(
#     project="sinkjaer/BEHRT",api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIzOWVmOWI3Mi1jNjliLTQ3NmEtODVjMy0wZjkxZTBiMzFiMzEifQ==",
#     log_model_checkpoints=True,
#     name=name_experiment,
# )


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
        self.date_vocab_size = config.get("date_vocab_size")
        self.optim_param = config.get("optim_param")


if Azure:
    # os.environ['NEPTUNE_MODE'] = 'offline'
    file_config = {
        "data_train": "../../EHR_data/data/pre_train_training_set.json",  # formated data
        "data_val": "../../EHR_data/data/pre_train_validation_set.json",  # formated data
        "model_path": "MLM/" + name_experiment,  # where to save model
        "model_name": "behrt",  # model name
        "vocab": "vocab.txt",  # vocabulary idx2token, token2idx
        "file_name": "log.txt",  # log path
    }
else:
    file_config = {
        "data_train": "/Users/mikkelsinkjaer/data/data.json",
        "data_val": "/Users/mikkelsinkjaer/data/data.json",
        "model_path": "MLM/" + name_experiment,  # where to save model
        "model_name": "behrt",  # model name
        "vocab": "vocab.txt",  # vocabulary idx2token, token2idx
        "file_name": "log.txt",  # log path
    }

create_folder(file_config["model_path"])

global_params = {"max_seq_len": 512, "gradient_accumulation_steps": 1}

optim_param = {"lr": 2e-5, "warmup_proportion": 0.1, "weight_decay": 0.01}

train_params = {
    "batch_size": 128,
    "max_len_seq": global_params["max_seq_len"],
}

# load data
with open(file_config["data_train"]) as f:
    data_train_json = json.load(f)
with open(file_config["data_val"]) as f:
    data_val_json = json.load(f)

# Build vocab
vocab_path = os.path.join(file_config["model_path"], file_config["vocab"])
vocab_list, word_to_idx = build_vocab(
    data_train_json,
    save_file=vocab_path,
)

# Data loader
masked_data_train = MaskedDataset(data_train_json, vocab_list, word_to_idx)
trainload = DataLoader(
    dataset=masked_data_train,
    batch_size=train_params["batch_size"],
    shuffle=True,
    pin_memory=True,
    num_workers=6,
)
masked_data_val = MaskedDataset(data_val_json, vocab_list, word_to_idx)
valload = DataLoader(
    dataset=masked_data_val,
    batch_size=train_params["batch_size"],
    shuffle=False,
    pin_memory=True,
    num_workers=6,
)

# Model config
model_config = {
    "vocab_size": len(vocab_list),  # number of disease + symbols for word embedding
    "hidden_size": 288,  # word embedding and seg embedding hidden size
    "seg_vocab_size": 2,  # number of vocab for seg embedding
    "date_vocab_size": int(
        365.25 * 23
    ),  # number of vocab for dates embedding --> days in 23 years
    "age_vocab_size": 144,  # number of vocab for age embedding
    "max_position_embedding": train_params["max_len_seq"],  # maximum number of tokens
    "hidden_dropout_prob": 0.1,  # dropout rate
    "num_hidden_layers": 6,  # number of multi-head attention layers required
    "num_attention_heads": 12,  # number of attention heads
    "attention_probs_dropout_prob": 0.1,  # multi-head attention dropout rate
    "intermediate_size": 512,  # the size of the "intermediate" layer in the transformer encoder
    "hidden_act": "gelu",  # The non-linear activation function in the encoder and the pooler "gelu", 'relu', 'swish' are supported
    "initializer_range": 0.02,  # parameter weight initializer range
    "optim_param": optim_param,  # learning rate
}

# Checkopoint
checkpoint_callback = ModelCheckpoint(
    monitor="metrics/epoch/loss_val",
    dirpath=file_config["model_path"] + "/checkpoints",
    filename="checkpoint-{epoch:02d}",
)

# Define model
# neptune_logger.log_hyperparams(model_config)
model = BertModel(BertConfig(model_config))
task = BertMLM(model, BertConfig(model_config))

# Initialize the Trainer with the callback and Neptune logger
trainer = pl.Trainer(
    # accelerator = 'gpu',
    # logger=neptune_logger,
    max_epochs=10,
    log_every_n_steps=100,
    callbacks=checkpoint_callback,
)


# Train the model as usual
trainer.fit(model=task, train_dataloaders=trainload, val_dataloaders=valload)
# %%
# # load model
# checkpoint = torch.load(
#     "/Users/mikkelsinkjaer/Library/Mobile Documents/com~apple~CloudDocs/transEHR/Code/transformerEHR/task/MLM/local_1/lightning_logs/version_2/checkpoints/epoch=0-step=1.ckpt"
# )
# print(checkpoint["hyper_parameters"])
# %%


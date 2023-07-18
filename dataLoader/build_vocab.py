# Script to build vocabulary from a dataset and save it.

# %%
from torchtext.vocab import vocab
from collections import Counter
import json


# Functions finds unique strings in a list
def unique_strings(string_list):
    unique_list = list(set(string_list))
    return unique_list


def build_vocab(
    data,
    special_tokens=("<UNK>", "<CLS>", "<SEP>", "<PAD>", "<MASK>"),
    save_file="vocab.txt",
):
    events = []
    for key, value in data.items():
        list_of_events = value["events"]
        for n in range(len(list_of_events)):
            events.append(list_of_events[n]["codes"])

    # Build vocabulary
    events = unique_strings(events)
    counter = Counter(events)
    vocab_list = vocab(counter, specials=special_tokens)
    vocab_list.set_default_index(vocab_list[special_tokens[0]])

    # Save vocabulary to a file
    with open(save_file, "w") as file:
        for word in vocab_list.get_itos():
            file.write(word + "\n")

    word_to_idx = vocab_list.get_stoi()

    return vocab_list, word_to_idx


def load_vocab(
    file_path, special_tokens=("<UNK>", "<CLS>", "<SEP>", "<PAD>", "<MASK>")
):
    with open(file_path, "r") as file:
        vocab_words = [line.strip() for line in file]

    vocab_list = vocab(Counter(vocab_words))
    vocab_list.set_default_index(vocab_list[special_tokens[0]])
    word_to_idx = vocab_list.get_stoi()

    return vocab_list, word_to_idx


# # %% Test Tokenizer build
# file_path = "/Users/mikkelsinkjaer/Library/Mobile Documents/com~apple~CloudDocs/transEHR/Code/transformerEHR/data/syntheticData.json"
# with open(file_path) as f:
#     data = json.load(f)

# vocab_list, word_to_idx = build_vocab(
#     data, save_file="/Users/mikkelsinkjaer/Library/Mobile Documents/com~apple~CloudDocs/transEHR/Code/transformerEHR/dataLoader/vocab.txt"
# )

# print("Known token:", vocab_list["Q12"])
# print("Unknown token:", vocab_list["Q1"])

# # %% Test Tokenizer load
# vocab_list, word_to_idx = load_vocab("/Users/mikkelsinkjaer/Library/Mobile Documents/com~apple~CloudDocs/transEHR/Code/transformerEHR/dataLoader/vocab.txt")
# print("Known token:", vocab_list["Q12"])
# print("Unknown token:", vocab_list["Q1"])
# # %%
# len(vocab_list)
# %%

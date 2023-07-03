from torchtext.vocab import vocab
from collections import Counter


# Functions finds unique strings in a list
def unique_strings(string_list):
    unique_list = list(set(string_list))
    return unique_list


# Function should only take data file
def build_vocab(data, special_tokens=("<UNK>", "<CLS>", "<SEP>", "<PAD>", "<MASK>")):
    events = []
    for key, value in data.items():
        list_of_events = value["events"]
        for n in range(len(list_of_events)):
            events.append(list_of_events[n]["codes"])

    # build vocabulary
    events = unique_strings(events)
    counter = Counter(events)
    vocab_list = vocab(counter, specials=special_tokens)
    vocab_list.set_default_index(vocab_list[special_tokens[0]])

    word_to_idx = vocab_list.get_stoi()

    return vocab_list, word_to_idx


# %% Test Tokenizer
# file_path = "H:/Code/transformerEHR/syntheticData.json"
# with open(file_path) as f:
#     data = json.load(f)
# vocab_list, word_to_idx = build_vocab(data)
# print("Known token:", vocab_list["Q12"])
# print("Unknown token:", vocab_list["Q1"])

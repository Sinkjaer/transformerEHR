import random


def random_masking(
    tokens, vocab, token_to_idx, UNKNOWN_TOKEN="<UNK>", MASK_TOKEN="<MASK>"
):
    output_label_w2idx = []
    output_token_w2idx = []
    masked_index = []
    for i, token in enumerate(tokens):
        prob = random.random()
        # mask token with 15% probability
        if prob < 0.15:
            prob /= 0.15

            # 80% randomly change token to mask token
            if prob < 0.8:
                output_token_w2idx.append(vocab[MASK_TOKEN])
                masked_index.append(1)  # 1 indicate masked token

            # 20% randomly change token to random token
            else:
                output_token_w2idx.append(random.choice(list(token_to_idx.values())))
                masked_index.append(2)  # 2 indicate random token

            # append current token to output (we will predict these later
            output_label_w2idx.append(vocab[UNKNOWN_TOKEN])
        else:
            # no masking token (will be ignored by loss function later)
            output_label_w2idx.append(-1)
            output_token_w2idx.append(vocab[token])
            masked_index.append(0)  # 0 indicate no masked token

    return output_token_w2idx, output_label_w2idx, masked_index

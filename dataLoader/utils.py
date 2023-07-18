import random


def random_masking(
    tokens,
    vocab,
    token_to_idx,
    UNKNOWN_TOKEN="<UNK>",
    MASK_TOKEN="<MASK>",
    probability=0.15,
):
    true_tokens = []
    masked_tokens = []
    output_label = []
    for i, token in enumerate(tokens):
        prob = random.random()
        # mask token with 15% probability
        if prob < probability:
            prob /= probability

            # 80% randomly change token to mask token
            if prob < 0.8:
                masked_tokens.append(vocab[MASK_TOKEN])

            # 20% randomly change token to random token
            else:
                masked_tokens.append(random.choice(list(token_to_idx.values())))

            # append current token to output (we will predict these later
            output_label.append(vocab[token])
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)
            masked_tokens.append(vocab[token])

    return true_tokens, masked_tokens, output_label

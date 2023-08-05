from torch import optim, nn, utils, Tensor
import lightning.pytorch as pl


class SequneceEmbeddings(nn.Module):
    """Construct the embeddings from word, dates, segment, age"""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.segment_embeddings = nn.Embedding(
            config.seg_vocab_size, config.hidden_size
        )
        self.age_embeddings = nn.Embedding(config.age_vocab_size, config.hidden_size)
        self.posi_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        ).from_pretrained(
            embeddings=self._init_posi_embedding(
                config.max_position_embeddings, config.hidden_size
            )
        )

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        word_ids,
        dates_ids=None,
        age_ids=None,
        seg_ids=None,
        posi_ids=None,
    ):
        # Only input tokens
        dates_ids = None  # dates_ids is not used

        if seg_ids is None:
            seg_ids = torch.zeros_like(word_ids)
        if age_ids is None:
            age_ids = torch.zeros_like(word_ids)
        if posi_ids is None:
            posi_ids = torch.zeros_like(word_ids)

        word_embed = self.word_embeddings(word_ids)
        segment_embed = self.segment_embeddings(seg_ids)
        age_embed = self.age_embeddings(age_ids.long())
        posi_embeddings = self.posi_embeddings(posi_ids)

        embeddings = word_embed + segment_embed + age_embed + posi_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

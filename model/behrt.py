from torch import nn
import lightning.pytorch as pl
import torch
import numpy as np
import pytorch_pretrained_bert as Bert
from sklearn.metrics import accuracy_score


class SequneceEmbeddings(nn.Module):
    """Construct the embeddings from word, dates, segment, age"""

    def __init__(self, config):
        super(SequneceEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.date_embeddings = nn.Embedding(
            config.date_vocab_size, config.hidden_size
        ).from_pretrained(
            embeddings=self._init_posi_embedding(
                config.date_vocab_size, config.hidden_size
            )
        )
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
        if dates_ids is None:
            dates_ids = torch.zeros_like(word_ids)  # dates_ids is not used
        if seg_ids is None:
            seg_ids = torch.zeros_like(word_ids)
        if age_ids is None:
            age_ids = torch.zeros_like(word_ids)
        if posi_ids is None:
            posi_ids = torch.zeros_like(word_ids)

        word_embed = self.word_embeddings(word_ids)
        date_embed = self.date_embeddings(dates_ids)
        segment_embed = self.segment_embeddings(seg_ids)
        age_embed = self.age_embeddings(age_ids.long())
        posi_embeddings = self.posi_embeddings(posi_ids)

        embeddings = (
            word_embed + date_embed + segment_embed + age_embed + posi_embeddings
        )

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    def _init_posi_embedding(self, max_position_embedding, hidden_size):
        def even_code(pos, idx):
            return np.sin(pos / (10000 ** (2 * idx / hidden_size)))

        def odd_code(pos, idx):
            return np.cos(pos / (10000 ** (2 * idx / hidden_size)))

        # initialize position embedding table
        lookup_table = np.zeros((max_position_embedding, hidden_size), dtype=np.float32)

        # reset table parameters with hard encoding
        # set even dimension
        for pos in range(max_position_embedding):
            for idx in np.arange(0, hidden_size, step=2):
                lookup_table[pos, idx] = even_code(pos, idx)
        # set odd dimension
        for pos in range(max_position_embedding):
            for idx in np.arange(1, hidden_size, step=2):
                lookup_table[pos, idx] = odd_code(pos, idx)

        return torch.tensor(lookup_table)


class BertModel(Bert.modeling.BertPreTrainedModel):
    def __init__(self, config):
        super(BertModel, self).__init__(config)
        self.embeddings = SequneceEmbeddings(config=config)
        self.encoder = Bert.modeling.BertEncoder(config=config)
        self.pooler = Bert.modeling.BertPooler(config)
        self.apply(self.init_bert_weights)

    def forward(
        self,
        input_ids,
        dates_ids=None,
        age_ids=None,
        seg_ids=None,
        posi_ids=None,
        attention_mask=None,
        output_all_encoded_layers=True,
    ):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        if age_ids is None:
            age_ids = torch.zeros_like(input_ids)

        if seg_ids is None:
            seg_ids = torch.zeros_like(input_ids)

        if posi_ids is None:
            posi_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(
            input_ids, dates_ids, age_ids, seg_ids, posi_ids
        )
        encoded_layers = self.encoder(
            embedding_output,
            extended_attention_mask,
            output_all_encoded_layers=output_all_encoded_layers,
        )
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output


class BertMLM(pl.LightningModule):
    def __init__(self, BertModel, config):
        super(BertMLM, self).__init__()
        self.BertModel = BertModel
        self.config = config  # Store the config
        self.cls = Bert.modeling.BertOnlyMLMHead(
            config, BertModel.embeddings.word_embeddings.weight
        )
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-1)

        # Log performance metrics
        self.training_step_outputs = []
        self.validation_step_outputs = []

    def training_step(self, batch, batch_idx):
        (
            dates_ids,
            age_ids,
            input_ids,
            posi_ids,
            segment_ids,
            attMask,
            output_labels,
        ) = batch

        sequence_output, _ = self.BertModel(
            input_ids,
            dates_ids,
            age_ids,
            segment_ids,
            posi_ids,
            attMask,
            output_all_encoded_layers=False,
        )

        prediction_scores = self.cls(sequence_output)

        y = output_labels.view(-1)
        y_hat = prediction_scores.view(-1, self.config.vocab_size)
        masked_lm_loss = self.loss_fct(
            y_hat,
            y,
        ) / len(y_hat)

        y_true = y.cpu().detach().numpy()
        y_pred = y_hat.argmax(axis=1).cpu().detach().numpy()

        # Only make predictions on the masked tokens
        index = y_true != -1
        y_true = y_true[index]
        y_pred = y_pred[index]

        acc = accuracy_score(y_true, y_pred)

        self.log(
            "metrics/batch/loss_train",
            masked_lm_loss,
            prog_bar=False,
        )
        self.log("metrics/batch/acc_train", acc.astype(np.float32))

        output = {"loss": masked_lm_loss, "y_true": y_true, "y_pred": y_pred}
        self.training_step_outputs.append(output)
        return output

    def on_train_epoch_end(self):
        outputs = self.training_step_outputs
        loss = np.array([])
        y_true = np.array([])
        y_pred = np.array([])
        for results_dict in outputs:
            loss = np.append(loss, results_dict["loss"].cpu().detach().numpy())
            y_true = np.append(y_true, results_dict["y_true"])
            y_pred = np.append(y_pred, results_dict["y_pred"])
        acc = accuracy_score(y_true, y_pred)
        self.log("metrics/epoch/loss_train", loss.mean().astype(np.float32))
        self.log("metrics/epoch/acc_train", acc.astype(np.float32))

        self.training_step_outputs.clear()  # free memory

    def validation_step(self, batch, batch_idx):
        (
            dates_ids,
            age_ids,
            input_ids,
            posi_ids,
            segment_ids,
            attMask,
            output_labels,
        ) = batch

        sequence_output, _ = self.BertModel(
            input_ids,
            dates_ids,
            age_ids,
            segment_ids,
            posi_ids,
            attMask,
            output_all_encoded_layers=False,
        )

        prediction_scores = self.cls(sequence_output)

        y = output_labels.view(-1)
        y_hat = prediction_scores.view(-1, self.config.vocab_size)
        masked_lm_loss = self.loss_fct(
            y_hat,
            y,
        ) / len(y_hat)

        y_true = y.cpu().detach().numpy()
        y_pred = y_hat.argmax(axis=1).cpu().detach().numpy()

        # Only make predictions on the masked tokens
        index = y_true != -1
        y_true = y_true[index]
        y_pred = y_pred[index]

        outputs = {"loss": masked_lm_loss, "y_true": y_true, "y_pred": y_pred}
        self.validation_step_outputs.append(outputs)
        return outputs

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        loss = np.array([])
        y_true = np.array([])
        y_pred = np.array([])
        for results_dict in outputs:
            loss = np.append(loss, results_dict["loss"].cpu().detach().numpy())
            y_true = np.append(y_true, results_dict["y_true"])
            y_pred = np.append(y_pred, results_dict["y_pred"])
        acc = accuracy_score(y_true, y_pred)
        self.log("metrics/epoch/loss_val", loss.mean().astype(np.float32))
        self.log("metrics/epoch/acc_val", acc.astype(np.float32))

        self.validation_step_outputs.clear()  # free memory

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.optim_param['lr'],weight_decay=self.config.optim_param['weight_decay'])
        return optimizer


class BertPrediction(pl.LightningModule):
    def __init__(self, BertModel, config, num_labels):
        super(BertPrediction, self).__init__()
        self.config = config  # Store the config
        self.num_labels = num_labels
        self.BertModel = BertModel
        self.BertModel.requires_grad_(False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.cls = nn.Linear(config.hidden_size, num_labels)
        self.loss_fct = nn.MultiLabelSoftMarginLoss()

        # Log performance metrics
        self.training_step_outputs = []
        self.validation_step_outputs = []

    def training_step(self, batch, batch_idx):
        (
            dates_ids,
            age_ids,
            input_ids,
            posi_ids,
            segment_ids,
            attMask,
            output_labels,
        ) = batch

        _, pooled_output = self.BertModel(
            input_ids,
            dates_ids,
            age_ids,
            segment_ids,
            posi_ids,
            attMask,
            output_all_encoded_layers=False,
        )

        pooled_output = self.dropout(pooled_output)
        logits = self.cls(pooled_output)

        y = output_labels.view(-1, self.num_labels)
        y_hat = logits.view(-1, self.num_labels)
        loss = self.loss_fct(
            y_hat,
            y,
        ) / len(y_hat)

        y_true = y.cpu().detach().numpy()
        y_pred = y_hat.argmax(axis=1).cpu().detach().numpy()

        acc = accuracy_score(y_true, y_pred)

        self.log(
            "metrics/batch/loss_train",
            loss,
            prog_bar=False,
        )
        self.log("metrics/batch/acc_train", acc.astype(np.float32))

        output = {"loss": loss, "y_true": y_true, "y_pred": y_pred}
        self.training_step_outputs.append(output)
        return output

    def on_train_epoch_end(self):
        outputs = self.training_step_outputs
        loss = np.array([])
        y_true = np.array([])
        y_pred = np.array([])
        for results_dict in outputs:
            loss = np.append(loss, results_dict["loss"].cpu().detach().numpy())
            y_true = np.append(y_true, results_dict["y_true"])
            y_pred = np.append(y_pred, results_dict["y_pred"])
        acc = accuracy_score(y_true, y_pred)
        self.log("metrics/epoch/loss_train", loss.mean().astype(np.float32))
        self.log("metrics/epoch/acc_train", acc.astype(np.float32))

        self.training_step_outputs.clear()  # free memory

    def validation_step(self, batch, batch_idx):
        (
            dates_ids,
            age_ids,
            input_ids,
            posi_ids,
            segment_ids,
            attMask,
            output_labels,
        ) = batch

        _, pooled_output = self.BertModel(
            input_ids,
            dates_ids,
            age_ids,
            segment_ids,
            posi_ids,
            attMask,
            output_all_encoded_layers=False,
        )

        pooled_output = self.dropout(pooled_output)
        logits = self.cls(pooled_output)

        y = output_labels.view(-1, self.num_labels)
        y_hat = logits.view(-1, self.num_labels)
        loss = self.loss_fct(
            y_hat,
            y,
        ) / len(y_hat)

        y_true = y.cpu().detach().numpy()
        y_pred = y_hat.argmax(axis=1).cpu().detach().numpy()

        output = {"loss": loss, "y_true": y_true, "y_pred": y_pred}
        self.validation_step_outputs.append(output)
        return output

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        loss = np.array([])
        y_true = np.array([])
        y_pred = np.array([])
        for results_dict in outputs:
            loss = np.append(loss, results_dict["loss"].cpu().detach().numpy())
            y_true = np.append(y_true, results_dict["y_true"])
            y_pred = np.append(y_pred, results_dict["y_pred"])
        acc = accuracy_score(y_true, y_pred)
        self.log("metrics/epoch/loss_val", loss.mean().astype(np.float32))
        self.log("metrics/epoch/acc_val", acc.astype(np.float32))

        self.validation_step_outputs.clear()  # free memory

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.optim_param['lr'],weight_decay=self.config.optim_param['weight_decay'])
        return optimizer

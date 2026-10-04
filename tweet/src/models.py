"""RoBERTa hidden-state averaging with start, end, and auxiliary logits."""
from pathlib import Path
import torch
import torch.nn as nn
from transformers import RobertaConfig, RobertaModel
from .config import Config


class TweetSentimentModel(nn.Module):
    def __init__(self, config: Config, encoder_config=None):
        super().__init__()
        self.config = config
        if encoder_config is not None:
            # Checkpoints contain the architecture; scoring never downloads weights.
            roberta_config = RobertaConfig.from_dict(encoder_config)
            self.encoder = RobertaModel(roberta_config)
        elif config.model.pretrained:
            local = Path(config.data.vocab_file).parent
            source = str(local) if (local / 'config.json').exists() else config.model.model_name
            self.encoder = RobertaModel.from_pretrained(source, output_hidden_states=True)
            roberta_config = self.encoder.config
        else:
            roberta_config = RobertaConfig(
                vocab_size=config.model.vocab_size,
                hidden_size=config.model.hidden_size,
                num_hidden_layers=config.model.num_hidden_layers,
                num_attention_heads=config.model.num_attention_heads,
                intermediate_size=config.model.intermediate_size,
                max_position_embeddings=config.data.max_length + 2,
                output_hidden_states=True,
                pad_token_id=1, bos_token_id=0, eos_token_id=2,
            )
            self.encoder = RobertaModel(roberta_config)
        if config.data.max_length + 2 > roberta_config.max_position_embeddings:
            raise ValueError('max_length exceeds the RoBERTa position embedding capacity')
        self.dropout = nn.Dropout(config.model.dropout_rate)
        self.classifier = nn.Linear(roberta_config.hidden_size, 3)
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, tokens, masks, text_mask=None):
        outputs = self.encoder(tokens, attention_mask=masks, output_hidden_states=True, return_dict=True)
        # The full model averages the last four hidden states; tiny models use what is available.
        features = torch.stack(outputs.hidden_states[-4:]).mean(dim=0)
        logits = self.classifier(self.dropout(features))
        start, end, auxiliary = logits.unbind(dim=-1)
        valid = masks.bool() if text_mask is None else text_mask.bool()
        minimum = torch.finfo(start.dtype).min
        return start.masked_fill(~valid, minimum), end.masked_fill(~valid, minimum), auxiliary


class TweetLoss(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()

    def forward(self, start_logits, end_logits, start_idx, end_idx, aux_logits, aux_labels):
        loss = self.ce_loss(start_logits, start_idx) + self.ce_loss(end_logits, end_idx)
        if self.config.model.auxiliary_loss_weight:
            loss = loss + self.config.model.auxiliary_loss_weight * self.dice_loss(aux_logits, aux_labels)
        return loss


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, preds, target):
        preds = torch.sigmoid(preds).reshape(-1)
        target = target.reshape(-1).to(preds.dtype)
        intersection = (preds * target).sum()
        return 1 - (2 * intersection + self.smooth) / (preds.square().sum() + target.square().sum() + self.smooth)

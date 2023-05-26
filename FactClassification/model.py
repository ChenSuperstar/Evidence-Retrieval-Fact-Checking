from torch import nn
from transformers import AutoModel


class FCModel(nn.Module):
    def __init__(self, pre_encoder, dropout_rate=0.5):
        super(FCModel, self).__init__()
        self.encoder = AutoModel.from_pretrained(pre_encoder)
        hidden_size = self.encoder.config.hidden_size

        self.cls = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, 4)
        )

    def forward(self, input_ids, attention_mask):
        texts_emb = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        texts_emb = texts_emb[:, 0, :]
        logits = self.cls(texts_emb)
        return logits
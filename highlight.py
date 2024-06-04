import torch.nn as nn
import torch
from transformers import AutoConfig, AutoModel

class HighlightHead(nn.Module):
    
    def __init__(self, outdim: int, model_name: str = 'bert-base-cased'):
        super().__init__()
        self.outdim = outdim
        self.config = AutoConfig.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(self.config.hidden_size, outdim, bias=False)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.loss_fn = nn.NLLLoss()
        
    def forward(self, input_ids, target, attention_mask):
        encoded = self.model(input_ids, token_type_ids=None, attention_mask=attention_mask)
        encoded = torch.squeeze(encoded['last_hidden_state'])
        d_encoded = self.dropout(encoded)
        outputs = self.linear(d_encoded)
        probs = self.logsoftmax(outputs)
        target_attn = torch.where(attention_mask == 1, target, self.loss_fn.ignore_index).squeeze()
        loss = self.loss_fn(probs, target_attn)
        return probs, loss
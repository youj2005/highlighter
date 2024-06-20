import torch.nn as nn
import torch
from transformers import AutoModel

class HighlightHead(nn.Module):
    
    def __init__(
      self,
      outdim: int,
      hidden_size: int, 
      kernel_size: int,
      model_name: str = 'bert-base-cased'
    ):
        super().__init__()
        self.outdim = outdim
        self.model = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(hidden_size, outdim, bias=False)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.loss_fn = nn.NLLLoss()
        
    def forward(self, input_ids, target, mask):
        encoded = self.model(input_ids, token_type_ids=None)
        encoded = torch.squeeze(encoded['last_hidden_state'])
        d_encoded = self.dropout(encoded)
        outputs = self.linear(d_encoded)
        logprobs = self.logsoftmax(outputs)
        target_attn = torch.where(mask == 1, target, self.loss_fn.ignore_index).squeeze()
        loss = self.loss_fn(logprobs, target_attn)
        return logprobs, loss
    
    def save(self, path):
        torch.save(self.state_dict(), path)
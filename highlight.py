import torch.nn as nn
from transformers import T5Config

class HighlightHead(nn.Module):
    def __init__(self, outdim: int, config: T5Config = T5Config.from_pretrained('t5-small')):
        super().__init__()
        self.linear = nn.Linear(config.d_model, outdim, bias=True)
        self.softmax = nn.Softmax(dim=0)
        
    def forward(self, input_ids):
        output = self.linear(input_ids)
        logits = self.softmax(output)
        return logits
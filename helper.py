from transformers import AutoConfig, AutoTokenizer
import torch

class HighlightHelper():
    def __init__(self, model_name: str = 'bert-base-cased'):
        self.config = AutoConfig.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def split_mask(self, text, highlight, index, device):
        ids = []
        mod_highlights = []
        mask = []

        for i, word in enumerate(text):
            encoded_input = self.tokenizer(word, add_special_tokens=False)['input_ids']
            ids.extend(encoded_input)
            if (i == index):
                mask.extend([1] * len(encoded_input))
                mod_highlights.extend([highlight] * len(encoded_input))
            else:
                mask.extend([0] * len(encoded_input))
                mod_highlights.extend([0] * len(encoded_input))

        ids = [[101] + ids + [102]]
        mod_highlights = [0] + mod_highlights + [0]
        mask = [0] + mask + [0]

        return (
            torch.tensor(ids, device=device), 
            torch.tensor(mod_highlights, dtype=torch.long, device=device), 
            torch.tensor(mask, device=device)
            )
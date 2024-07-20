from typing import Dict, List

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import RobertaTokenizer, LongformerConfig
l_config = LongformerConfig.from_pretrained("allenai/longformer-base-4096")
l_config.attention_mode = 'sliding_chunks'
class MyDataset(Dataset):
    def __init__(self, data: List[Dict], tokenizer: RobertaTokenizer, config: l_config, text_input: str):
        self.data = data
        self.tokenizer = tokenizer
        self.config = config
        self.text_input = text_input
    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        #text = 'Introduction ' + self.data[index]["introduction"] + ' Methods ' + self.data[index]["methods"]
        #text = 'Title ' + self.data[index]["title"] + ' Abstract ' + self.data[index]["abstract"]
        #text = 'Introduction ' + self.data[index]["introduction"] + ' Results ' + self.data[index]["results"]
        # text = 'Introduction ' + self.data[index]["introduction"] + ' Conclusion ' + self.data[index]["discussion"] + self.data[index]["conclusion"]
        text = ''
        if '_' in self.text_input:
            section = self.text_input.split('_')
            for sec in section:
                text += ' ' + str.upper(sec) + ' ' + self.data[index][sec]
        #text = 'Methods ' + self.data[index]["methods"] + ' Results ' + self.data[index]["results"]
        label = self.data[index]["tns"]
        if label == 1 or label == 2:
            label = 0
        if label == 3:
            label = 1
        if label == 4:
            label = 2
        text_encode = self.tokenizer(text, truncation=True, padding=True)
        return {
            "input_ids": torch.tensor(text_encode['input_ids']).unsqueeze(0),
            "attention_mask": torch.tensor(text_encode['attention_mask']).unsqueeze(0),
            "label": torch.tensor(label)
        }

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        input_ids_list = [instance['input_ids'][0] for instance in batch]
        input_ids_pad = pad_sequence(input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id)

        attention_mask_list = [instance['attention_mask'][0] for instance in batch]
        attention_mask_pad = pad_sequence(attention_mask_list, batch_first=True, padding_value=0)

        label_list = [instance['label'] for instance in batch]

        return {
            "input_ids": torch.tensor(input_ids_pad),
            "attention_mask": torch.tensor(attention_mask_pad),
            "label": torch.tensor(label_list)
        }
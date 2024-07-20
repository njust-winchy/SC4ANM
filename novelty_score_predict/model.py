import torch
import torch.nn as nn
from transformers import LongformerModel
from transformers.modeling_outputs import SequenceClassifierOutput


class MyModel_Pool(nn.Module):
    def __init__(self, pretrained_model_name_or_path, num_classes, config):
        super().__init__()
        self.backbone = LongformerModel.from_pretrained(pretrained_model_name_or_path, config=config)

        self.fc1 = nn.Linear(self.backbone.pooler.dense.out_features, 128)  # [768, 128]
        self.fc2 = nn.Linear(128, num_classes)  # [768, num_classes]

    def forward(self, input_ids, attention_mask):
        out_backbone = self.backbone(input_ids=input_ids, attention_mask=attention_mask).pooler_output  # [batch_size, 768]
        out_fc1 = self.fc1(out_backbone)  # [batch_size, 128]
        out_fc2 = self.fc2(out_fc1)  # [batch_size, num_classes]

        return SequenceClassifierOutput(logits=out_fc2)
import os
import sys
import torch


num_classes = 3  # 

pretrained_model_name_or_path = 'allenai/longformer-base-4096'
text_input = 'introduction_methods'
num_train_epochs = 10
train_batch_size = 1
valid_batch_size = 1
learning_rate = 1e-5
weight_decay = 0
num_warmup_steps = 500
accmulation_steps = 16
device = torch.device("cuda:0")
data_dir = os.path.join(sys.path[0], 'tns_dataset')
weights_dir = os.path.join(sys.path[0], "weights")  
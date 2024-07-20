import json
import os
from tqdm import tqdm
import torch
from utils import remove_special_chars
file_list = os.listdir('fin_data')
prior = 'F:\code\\nov_eval\\fin_data\$insert\$insert.json'

save_list = []

for file_name in tqdm(file_list):
     with open(prior.replace('$insert', file_name)) as f:
         data = json.load(f)
     for cont in data:
         if cont['ens'] == 0:
             continue
         save_list.append(cont)
     f.close()
 with open('ens_data.json', 'w') as fp:
     json.dump(save_list, fp)

train_size = int(0.8 * len(save_list))
test_size = int(0.1 * len(save_list))
va_size = len(save_list) - train_size - test_size
train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(save_list, [train_size, va_size, test_size])
with open('final_data\\train.json', 'w') as fp:
    json.dump(list(train_dataset), fp)
with open('final_data\\valid.json', 'w') as fp:
    json.dump(list(valid_dataset), fp)
with open('final_data\\test.json', 'w') as fp:
    json.dump(list(test_dataset), fp)
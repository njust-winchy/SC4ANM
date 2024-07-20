import json
import ast
from tqdm import tqdm
import nltk
labels = ['introduction', 'methods', 'results', 'discussion', 'conclusion']

# with open('novel_dataset.json') as f:
#     data = json.load(f)
# for i in data:
#     paper = i['paper_content'].split('\n')
#     if paper[2] != 'Abstract':
#         print(False)
#     print()
null =''
true= 'true'
false='false'
data = []
with open('pubmed-dataset\\test.txt', 'r') as f:
    pubmed = f.readlines()
f.close()

for line in pubmed:
    data.append(eval(line))
with open('arxiv-dataset\\test.txt', 'r') as f:
    arxiv = f.readlines()
f.close()

for line in arxiv:
    data.append(eval(line))
save_list = []
for i in tqdm(data):
    com = i['section_names']
    for label in com:
        for l in labels:
            save_dict = {}
            if l.lower() in label.lower():
                pos = com.index(label)
                l_pos = labels.index(l)
                sentences = i['sections'][pos]
                if len(sentences[0]) == 0:
                    continue
                senten = ''
                for m in sentences:
                    senten = senten.join(m)
                save_dict['text'] = sentences
                save_dict['label'] = l_pos
                save_list.append(save_dict)
with open('ref_data/train.json', 'w') as f:
    json.dump(save_list, f)


import json
from tqdm import tqdm
from transformers import pipeline

from sc_llama import llama3, section_classifier, read_peer_review
import os
# https://huggingface.co/howanching-clara/classifier_for_academic_texts
classifier = pipeline("text-classification", model="howanching-clara/classifier_for_academic_texts", tokenizer="howanching-clara/classifier_for_academic_texts")
labels = ['introduction', 'methods', 'results', 'discussion', 'conclusion']

save_path = 'F:\code\\nov_eval\\fin_data\\'

with open('F:\code\\nov_eval\\novel_dataset.json') as f:
    data = json.load(f)
f.close()
for da in tqdm(data):
    save_list = []
    save_dic = {}
    p_c = da['paper_content']
    normal_data = p_c.split('\n')
    ens = read_peer_review(da['peer_review'])
    tns = da['novelty_score']
    title = normal_data[1]
    abstract = normal_data[3]
    start = 0
    end = 0
    check_dirs = save_path + da['id']
    save_file = check_dirs + '\\' + da['id'] + '.json'
    if os.path.exists(save_file):
        continue
    text_list = []
    for n, t in enumerate(normal_data):
        if str.isupper(t):
            if start == 0:
                start = n + 1
            else:
                end = n - 1
                if start == end:
                    text = normal_data[start:end+1]
                    if len(text) > 0:
                        text_list.append(text)
                        text = []
                else:
                    text = normal_data[start:end]
                    if len(text) > 0:
                        text_list.append(text)
                        text = []
                start = n + 1
    introduction = ''
    methods = ''
    results = ''
    discussion = ''
    conclusion = ''
    for li in text_list:
        cfat = ''
        for s in li:
            cfat += s
        if len(cfat) > 512:
            pred_text = cfat[0:512]
        else:
            pred_text = cfat
        pred = classifier(pred_text)[0]['label']
        if pred == 'MAIN TEXT':
            if len(cfat) > 19999:
                pre_t = cfat[0:19998]
            else:
                pre_t = cfat
            plm_p = section_classifier(pre_t)
            if plm_p != -1:
                result = labels[plm_p]
            else:
                result = llama3(pre_t)
            for m, l in enumerate(labels):
                if l in result.lower():
                    if m == 0:
                        introduction += cfat
                    if m == 1:
                        methods += cfat
                    if m == 2:
                        results += cfat
                    if m == 3:
                        discussion += cfat
                    if m == 4:
                        conclusion += cfat
    save_dic['id'] = da['id']
    save_dic['title'] = title
    save_dic['abstract'] = abstract
    save_dic['introduction'] = introduction
    save_dic['methods'] = methods
    save_dic['results'] = results
    save_dic['discussion'] = discussion
    save_dic['conclusion'] = conclusion
    save_dic['novelty_sentence'] = da['novelty_sentence']
    save_dic['ref'] = da['ref']
    save_dic['tns'] = tns
    save_dic['ens'] = ens
    save_dic['Decision'] = da['Decision']
    save_list.append(save_dic)
    if not os.path.exists(check_dirs):
        os.makedirs(check_dirs)
    if os.path.exists(save_file):
        continue
    with open(save_file, 'w') as f:
        json.dump(save_list, f)
    f.close()



import json
from tqdm import tqdm
import tiktoken
from openai import OpenAI
import nltk

import torch

i = 0
dec= {'Accept (Poster)':0,'Reject':0,'Accept: notable-top-25%':0,'Accept (Oral)':0, 'Accept: notable-top-5%':0, 'Accept (Spotlight)':0,'Accept: poster':0}
def count_tokens(text):
    enc = tiktoken.get_encoding("cl100k_base")  # 假设使用GPT-4
    tokens = enc.encode(text)
    return len(tokens)
def truncate_text(text, max_tokens):
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    return enc.decode(tokens)

def preprocess_text(string: str):
    string = string.lower()
    punctuations = '''!()-[]{};:'"\<>/?@#$^&*_~+='''
    string = string.replace('’', "")
    string = string.replace('\n', "")
    for x in string.lower():
        if x in punctuations:
            string = string.replace(x, "")
    return string
def gpt_api(prompt):
    ai = OpenAI(base_url="", api_key="")
    completion = ai.chat.completions.create(model='gpt-4o',
                                            messages=[{"role": "system", "content": "you are a helpful assistant."},
                                                    {"role": "user", "content": prompt}],
                                            max_tokens=50)
    return completion.choices[0].message.content

count_dic = {'first':0, 'second':0, 'third':0}
text_input = 'methods_results_discussion_conclusion'

prompt_template = 'You need to perform a text classification task with three labels [0: basic novelty, 1: moderate novelty, 2: highly novel]. ' \
         'The following is an academic paper text, assign corresponding labels to it:\n $insert'\
         '\n Just return the label number.'

with open('tns_data.json') as f:
    data = json.load(f)
    save_list = []
    for rev in tqdm(data):
        text = ''
        save_dic = {}
        if rev['tns'] == 1 or rev['tns'] == 2:
            if count_dic['first'] == 40:
                continue
            if len(rev['introduction']) > 1 and len(rev['methods']) > 1 and len(rev['results']) > 1:
                if len(rev['discussion']) > 1 or len(rev['conclusion']) > 1:
                    dec[rev['Decision']] += 1
                    i += 1
                    if '_' in text_input:
                        section = text_input.split('_')
                        for sec in section:
                            text += ' ' + str.upper(sec) + ' ' + rev[sec]
                    else:
                        text = rev[text_input]
                    text = preprocess_text(text)
                    token_count = count_tokens(text)
                    if token_count > 10000:
                        text = truncate_text(text, 10000)
                    llm_input = prompt_template.replace('$insert', text)
                    result = gpt_api(llm_input)
                    save_dic['llm_input'] = llm_input
                    save_dic['result'] = result
                    save_dic['label'] = rev['tns']
                    save_list.append(save_dic)
                    count_dic['first']+=1

        if rev['tns'] == 3:
            if count_dic['second'] == 40:
                continue
            if len(rev['introduction']) > 1 and len(rev['methods']) > 1 and len(rev['results']) > 1:
                if len(rev['discussion']) > 1 or len(rev['conclusion']) > 1:
                    dec[rev['Decision']] += 1
                    i += 1
                    if '_' in text_input:
                        section = text_input.split('_')
                        for sec in section:
                            text += ' ' + str.upper(sec) + ' ' + rev[sec]
                    else:
                        text = rev[text_input]
                    text = preprocess_text(text)
                    token_count = count_tokens(text)
                    if token_count > 10000:
                        text = truncate_text(text, 10000)
                    llm_input = prompt_template.replace('$insert', text)
                    result = gpt_api(llm_input)
                    save_dic['llm_input'] = llm_input
                    save_dic['result'] = result
                    save_dic['label'] = rev['tns']
                    save_list.append(save_dic)
                    count_dic['second'] += 1
        if rev['tns'] == 4:
            if count_dic['third'] == 40:
                continue
            if len(rev['introduction']) > 1 and len(rev['methods']) > 1 and len(rev['results']) > 1:
                if len(rev['discussion']) > 1 or len(rev['conclusion']) > 1:
                    dec[rev['Decision']] += 1
                    i += 1
                    if '_' in text_input:
                        section = text_input.split('_')
                        for sec in section:
                            text += ' ' + str.upper(sec) + ' ' + rev[sec]
                    else:
                        text = rev[text_input]
                    text = preprocess_text(text)
                    token_count = count_tokens(text)
                    if token_count > 10000:
                        text = truncate_text(text, 10000)
                    llm_input = prompt_template.replace('$insert', text)
                    result = gpt_api(llm_input)
                    save_dic['llm_input'] = llm_input
                    save_dic['result'] = result
                    save_dic['label'] = rev['tns']
                    save_list.append(save_dic)
                    count_dic['third'] += 1
        if count_dic['first'] + count_dic['second'] + count_dic['third'] == 120:
            break
    with open(text_input + '.json', 'w') as fp:
        json.dump(save_list, fp)







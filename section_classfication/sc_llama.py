import torch
from transformers import pipeline, AutoTokenizer
from gradio_client import Client
from model import MyModel_Pool

def classifier_for_academic_texts(text):
    classifier = pipeline("text-classification", model="howanching-clara/classifier_for_academic_texts", tokenizer="howanching-clara/classifier_for_academic_texts")
    your_text = text
    result = classifier(your_text)
    return result

def llama3(text):
    mes = 'This is an academic text, whether it is an introduction, methods, results, discussion, or conclusion. Just reply whether it is introduction, methods, results, discussion, or conclusion.'
    prompt = text + '\n' + mes
    client = Client("")
    res = client.predict(
        message=prompt,
        request="You are helpful AI.",
        param_3=512,
        param_4=0,
        api_name="/chat"
    )
    client.close()
    return res
def section_classifier(text):
    model = MyModel_Pool()
    model.load_state_dict(torch.load(''))
    tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
    input = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")['input_ids'].to(
        'cuda:0')
    pred = model(input)
    v, out = torch.max(pred, dim=1)
    if v > 0.9:
        return out
    else:
        return -1

def read_peer_review(rev_cont):
    score_list = []
    for rev in rev_cont:
        score = rev['Empirical Novelty And Significance']
        if not str.isdigit(score[0]):
            continue
        score_list.append(int(score[0]))
    if len(score_list) == 0:
        score_list.append(0)
    max_value = max(score_list)
    min_value = min(score_list)
    # 计算最大值与最小值之间的差值
    difference = abs(max_value - min_value)
    if difference > 1:
        return 0
    else:
        return round(sum(score_list) / len(score_list))
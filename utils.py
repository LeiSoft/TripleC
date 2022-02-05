# import spacy
import json
from typing import List


# tagger = spacy.load('en_core_web_lg')

def load_data(path):
    """
    :param path: 数据位置
    :return: x,y
    装载训练数据
    """
    x_data = []
    y_data = []

    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            data = line.strip().split("\t")
            token_list = [word for word in data[1].split(" ") if word]

            # contain unique_id for feature extraction
            x_data.append(token_list)
            y_data.append(str(data[2]))

    return x_data, y_data


def load_non_label_data(path):
    x_data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            data = line.strip().split("\t")
            token_list = [word for word in data[1].split(" ") if word != ""]

            x_data.append((data[0], token_list))
    return x_data


def get_multi_label(path, y, data_type):
    output = []

    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f.readlines()):
            if data_type == 'scicite':
                dic = json.loads(line.strip())
                influence = dic['isKeyCitation']
                section_name = str(int(dic['excerpt_index'])//3)

            if data_type == '3c':
                influence = line.strip().split('\t')[2] == '0'
                section_name = 'Non_Section'

            if influence:
                output.append([y[i], 'I', section_name])
            else:
                output.append([y[i], 'N', section_name])
    return output

def get_scaffolds(path):
    worthiness, sections = [], []
    with open(path+'/scaffolds/cite-worthiness-scaffold-train.jsonl', 'r', encoding='utf-8') as f:
        for i, line in enumerate(f.readlines()):
            dic = json.loads(line.strip())
            worthiness.append(
                               (
                               dic['text'].split(), 
                               abs(-(dic['is_citation']==True))
                               )
                            )
    section2id = {'introduction': 0, 'related work': 1, 'method': 2, 'experiments': 3, 'conclusion': 4}
    with open(path+'/scaffolds/sections-scaffold-train.jsonl', 'r', encoding='utf-8') as f:
        for i, line in enumerate(f.readlines()):
            dic = json.loads(line.strip())
            sections.append((dic['text'].split(), section2id[dic['section_name']]))
    return worthiness, sections


def load_json(path, task_type):
    data, context = [], []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            if task_type == 'acl-arc':
                dic = json.loads(line.strip())
                context.append(dic['cleaned_cite_text'].split())
            if task_type == 'scicite':
                dic = json.loads(line.strip())
                context.append(dic['string'].split())
            if task_type == '3c':
                dic = None
                context.append(line.strip().split()[1].split())
            data.append(dic)
    # try to find some features
    # print(data[0].keys())
    # keys = ['section_number', 'extended_context', 'text']
    # for _ in range(10):
    #     for k in keys:
    #         print(k, data[_][k])
    #     print(data[_]['cite_marker_offset'][0]/len(data[_]['cleaned_cite_text']))
    #     print('############')
    # exit(99)
    return context, data


# 3c-task专用
def add_label(_y, multi_task_path: List):
    # only support numerical label
    for i, e in enumerate(_y):
        _y[i] = [str(e)]

    for path in multi_task_path:
        sub_y = load_data(path)[1]
        for i, e in enumerate(sub_y):
            _y[i].append(str(e))
    return _y

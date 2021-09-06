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


def get_multi_label(path, y):
    output = []

    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f.readlines()):
            # scicite
            influence = json.loads(line.strip())['isKeyCitation']
            # acl-arc
            # influence = json.loads(line.strip())['is_citation']
            # seq_pos = json.loads(line.strip())['excerpt_index']

            if influence:
                output.append([y[i], 'I'])
            else:
                output.append([y[i], 'N'])
    return output


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

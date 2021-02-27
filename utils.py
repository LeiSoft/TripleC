# import spacy


# tagger = spacy.load('en_core_web_lg')

def load_data(path, type_):
    """
    :param path: 数据位置
    :param type_: 任务类型
    :return: x,y
    装载训练数据
    """
    x_data = []
    y_data = []
    type_num = 1
    if type_ == 'influence':
        type_num = 2
    with open(path + "/train.tsv", 'r', encoding='utf-8') as f:
        for line in f.readlines():
            data = line.strip().split("\t")
            token_list = [word for word in data[0].split(" ") if word != ""]

            x_data.append(token_list)
            y_data.append(str(data[type_num]))
    return x_data, y_data

# import spacy


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
            token_list = [word for word in data[0].split(" ") if word != ""]

            x_data.append(token_list)
            y_data.append(str(data[1]))
    return x_data, y_data


def load_test_data(path):
    x_data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            data = line.strip().split("\t")
            token_list = [word for word in data[0].split(" ") if word != ""]

            x_data.append(token_list)
    return x_data

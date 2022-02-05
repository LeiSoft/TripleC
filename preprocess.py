import pandas as pd
import re
import os
import en_core_web_lg
from math import floor, ceil
import json
import pickle
from tqdm import tqdm

tagger = en_core_web_lg.load()


# 生成语料文件用于词向量训练
def generate_corpus(path):
    train = pd.read_csv(path + "SDP_train.csv", sep=',', header=0)
    test = pd.read_csv(path + "SDP_test.csv", sep=',', header=0)
    test_corpora = [
        clean_sentence(test.loc[i, 'citation_context'].replace("#AUTHOR_TAG", " ")
                       + test.loc[i, 'cited_title'] + test.loc[i, 'citing_title'])
        for i in range(len(test))]
    # test_corpora = tagger.pipe([test.loc[i, 'citation_context'] for i in range(len(test))])
    train_corpora = [
        clean_sentence(train.loc[i, 'citation_context'].replace("#AUTHOR_TAG", " ")
                       + train.loc[i, 'cited_title'] + train.loc[i, 'citing_title'])
        for i in range(len(train))]

    corpora = test_corpora + train_corpora

    with open("./datasets/corpora.txt", 'w', encoding='utf-8') as f:
        for line in corpora:
            text = line.strip().split(" ")
            block_num = ceil(len(text) / 500)

            sep_len = floor(len(text) / block_num)
            for i in range(block_num):
                begin = i * sep_len
                end = (i + 1) * sep_len
                if end > len(text):
                    end = len(text)
                f.write(" ".join(text[begin:end]) + "\n")
    # 增加额外语料
    # with open("./datasets/data.tsv", 'r', encoding='utf-8') as f:
    #     for line in f.readlines():
    #         corpora.append(clean_sentence(line.strip().split("\t")[2]))

    with open("./datasets/scicite/train.jsonl", 'r', encoding="utf-8") as f:
        for line in f.readlines():
            corpora.append(clean_sentence(json.loads(line.strip())["string"]))

    with open("./datasets/corpora_add.txt", 'w', encoding='utf-8') as f:
        for line in corpora:
            text = line.strip().split(" ")
            block_num = ceil(len(text) / 500)

            sep_len = floor(len(text) / block_num)
            for i in range(block_num):
                begin = i * sep_len
                end = (i + 1) * sep_len
                if end > len(text):
                    end = len(text)
                f.write(" ".join(text[begin:end]) + "\n")

    for path in ["./datasets/fulltext/train", "./datasets/fulltext/test"]:
        for file in tqdm(os.listdir(path)):
            with open("./datasets/corpora_add.txt", 'a+', encoding='utf-8') as f:
                f.writelines(open(path + "/" + file, 'r', encoding='utf-8').readlines())


def formatted(path, _label):
    """
    :param _label: 分类标签名称
    :param path:数据路径
    生成预训练所需要的数据
    """
    data = pd.read_csv(path + "SDP_train.csv", sep=',', header=0)
    output = [(data.loc[i, 'unique_id'],
               clean_sentence(data.loc[i, 'citation_context'].replace("#AUTHOR_TAG", "TAG")) + ' ' +
               data.loc[i, 'cited_title'] + ' ' + data.loc[i, 'citing_title'],
               str(data.loc[i, _label]))
              for i in range(len(data))]
    # output = []
    # for i, doc in enumerate(tagger.pipe([clean_sentence(data.loc[i, 'citation_context']) for i in range(len(data))])):
    #     seq = []
    #     for token in doc:
    #         if token.text == 'AUTHORTAG':
    #             seq.append("TAG")
    #         else:
    #             seq.append(token.pos_)
    #
    #     output.append((" ".join(seq), str(data.loc[i, label])))

    with open(path + "/train.tsv", 'w', encoding='utf-8') as f:
        for line in output:
            f.write("\t".join(line) + "\n")


def formatted_test(path):
    data = pd.read_csv(path + "SDP_test.csv", sep=',', header=0)

    test = [data.loc[i, 'unique_id'] + "\t" +
            clean_sentence(data.loc[i, 'citation_context'].replace("#AUTHOR_TAG", "TAG"))
            for i in range(len(data))]
    # test = []
    # for i, doc in enumerate(tagger.pipe([clean_sentence(data.loc[i, 'citation_context']) for i in range(len(data))])):
    #     seq = []
    #     for token in doc:
    #         if token.text == 'AUTHORTAG':
    #             seq.append("TAG")
    #         else:
    #             seq.append(token.pos_)
    #
    #     test.append((" ".join(seq)))
    with open(path + "/test.tsv", 'w', encoding='utf-8') as f:
        for line in test:
            f.write(line + "\n")


# support scicite and acl-arc
def process_sci(path, type_, _label, _stat, **params):
    context_label = params['context_label']
    intent_label = params['intent_label']
    data = []
    with open(path + type_ + ".jsonl", 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f.readlines()):
            dic = json.loads(line.strip())
            content = clean_sentence(dic[context_label])
            data.append((idx, content, _label[dic[intent_label]]))
            _stat[dic[intent_label]] = _stat.get(dic[intent_label], 0) + 1
    print(len(data), '\n', _stat)

    with open(path + type_ + ".tsv", 'w', encoding='utf-8') as f:
        for d in data:
            f.write(str(d[0]) + "\t" + d[1] + "\t" + str(d[2]) + "\n")


def build_3c_feature_dic(type_):
    assert type_ in ["train", "test"]
    if type_ == "train":
        paths = ["./datasets/3c-shared/intent/SDP_train.csv", "./datasets/3c-shared/influence/SDP_train.csv"]
    else:
        paths = ["./datasets/3c-shared/intent/SDP_test.csv", "./datasets/3c-shared/influence/SDP_test.csv"]
    title_set = set()
    author_set = set()
    for path in paths:
        data = pd.read_csv(path, sep=",", header=0)
        for i in range(len(data)):
            title_set.add(data.loc[i, "citing_title"])
            # title_set.add(data.loc[i, "cited_title"])
            author_set.add(data.loc[i, "citing_author"])
            # author_set.add(data.loc[i, "cited_author"])
    title_dic = {}
    author_dic = {}
    for i, title in enumerate(title_set):
        title_dic[title] = float(i)
    for i, author in enumerate(author_set):
        author_dic[author] = float(i)
    output = (title_dic, author_dic)
    pickle.dump(output, open("./datasets/3c_feature_dic_" + type_ + ".pkl", 'wb'))


def clean_sentence(s: str):
    s = "".join(re.findall('[a-zA-Z :!?]', s))
    return " ".join([token.text for token in tagger(s)])
    # return "".join(re.findall('[a-zA-Z ]', s))


# 为3c多任务进行预处理
def process_mt_3c():
    paths = ["./datasets/3c-shared/intent/SDP_train.csv", "./datasets/3c-shared/influence/SDP_train.csv"]

    data1 = pd.read_csv(paths[0], sep=",", header=0)
    data2 = pd.read_csv(paths[1], sep=",", header=0)
    datas = []
    stat = [0, 0, 0, 0]
    for i in range(len(data1)):
        dic = {'intent': str(data1['citation_class_label'][i]), 'influence': str(data2['citation_influence_label'][i])}
        with open(f'./datasets/3c-shared/fulltext/train/{data1["core_id"][i]}.txt', 'r', encoding='utf-8') as f:
            fulltext = ''.join([line.replace('\n', '') for line in f.readlines()])
            pos = fulltext.find(data1['citation_context'][i].replace('#AUTHOR_TAG', '').strip(' ')[:10])
            if pos == -1:
                print(fulltext, '\n', data1['citation_context'][i])
                exit(9)
            dic['offset'] = int((pos / len(fulltext))*100 // 33) if pos != -1 else 3
            stat[dic['offset']] += 1
        datas.append(dic)
    print(stat)
    with open('./datasets/3c-shared/multi_labels.jsonl', 'w', encoding='utf-8') as f:
        for data in datas:
            # print(data)
            f.write(json.dumps(data)+'\n')


# 3c多标签生成
process_mt_3c()

# 3c-task语料处理
# formatted("datasets/3c-shared/intent/", 'citation_class_label')
# formatted("datasets/3c-shared/influence/", 'citation_influence_label')
# formatted_test("datasets/3c-shared/intent/")
# formatted_test("datasets/3c-shared/influence/")

# 3c-语料特征文件构建，事先抽取
# generate_corpus("./datasets/3c-shared/intent/")
# for _ in ["test", "train"]:
#     build_3c_feature_dic(_)

# 合并语料
# all_ = []
# for s in ["train", "dev", "test"]:
#     with open("./datasets/scicite-scibert/"+s+".jsonl", 'r', encoding='utf-8') as f:
#         for line in f.readlines():
#             all_.append(line)
# with open("./datasets/scicite-scibert/all.jsonl", 'w', encoding='utf-8') as f:
#     for i, dic in enumerate(all_):
#         f.write(dic)

# 预处理scicite
# label = {'result': '0', 'background': '1', 'method': '2'}
# stat = {}
# for key in label.keys():
#     stat[key] = 0
# for t in ['train', 'dev', 'test']:
#     process_sci("./datasets/scicite/", t, label, stat, context_label='string', intent_label='label')

# 预处理acl-arc
# label = {'Background': '0', 'CompareOrContrast': '1', 'Extends': '2', 'Future': '3', 'Motivation': '4', 'Uses': '5'}
# stat = {}
# for key in label.keys():
#     stat[key] = 0
# process_sci("./datasets/acl-arc/", "all", label, stat, context_label='text', intent_label='intent')

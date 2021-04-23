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
                f.writelines(open(path+"/"+file, 'r', encoding='utf-8').readlines())


def formatted(path, label):
    """
    :param label: 分类标签名称
    :param path:数据路径
    生成预训练所需要的数据
    """
    data = pd.read_csv(path + "SDP_train.csv", sep=',', header=0)

    output = [(data.loc[i, 'unique_id'],
               clean_sentence(data.loc[i, 'citation_context'].replace("#AUTHOR_TAG", "TAG")
                              + data.loc[i, 'cited_title'] + data.loc[i, 'citing_title']),
               str(data.loc[i, label])
               )
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

    with open("/".join(path.split("/")[:2]) + "/train.tsv", 'w', encoding='utf-8') as f:
        for line in output:
            f.write("\t".join(line) + "\n")


def formatted_test(path):
    data = pd.read_csv(path + "SDP_test.csv", sep=',', header=0)

    test = [data.loc[i, 'unique_id'] + "\t" +
            clean_sentence(data.loc[i, 'citation_context'].replace("#AUTHOR_TAG", "TAG")
                           + data.loc[i, 'cited_title'] + data.loc[i, 'citing_title'])
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
    with open("/".join(path.split("/")[:2]) + "/test.tsv", 'w', encoding='utf-8') as f:
        for line in test:
            f.write(line + "\n")


def process_scicite(path, type_):
    label = {'result': '0', 'background': '1', 'method': '2'}
    stat = {'result': 0, 'background': 0, 'method': 0}
    data = []
    with open(path + type_ + ".jsonl", 'r', encoding='utf-8') as f:
        for line in f.readlines():
            dic = json.loads(line.strip())
            content = clean_sentence(dic["string"])
            data.append((content, label[dic["label"]]))
            stat[dic["label"]] = stat.get(dic["label"], 0) + 1
    print(len(data))
    print(stat)

    with open(path + type_ + ".tsv", 'w', encoding='utf-8') as f:
        for d in data:
            f.write(d[0] + "\t" + str(d[1]) + "\n")


def build_3c_feature_dic(type_):
    assert type_ in ["train", "test"]
    if type_ == "train":
        paths = ["./datasets/intent/SDP_train.csv", "./datasets/influence/SDP_train.csv"]
    else:
        paths = ["./datasets/intent/SDP_test.csv", "./datasets/influence/SDP_test.csv"]
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
    pickle.dump(output, open("./datasets/3c_feature_dic_"+type_+".pkl", 'wb'))


def clean_sentence(s: str):
    s = "".join(re.findall('[a-zA-Z :!?]', s))
    return " ".join([token.text for token in tagger(s)])
    # return "".join(re.findall('[a-zA-Z ]', s))


# formatted("datasets/intent/", 'citation_class_label')
# formatted("datasets/influence/", 'citation_influence_label')
# formatted_test("datasets/intent/")
# formatted_test("datasets/influence/")
# process_scicite("./datasets/scicite/", "train")
# process_scicite("./datasets/scicite/", "dev")
# process_scicite("./datasets/scicite/", "test")
# generate_corpus("datasets/intent/")
for _ in ["test", "train"]:
    build_3c_feature_dic(_)
import pandas as pd
import re
import en_core_web_lg

tagger = en_core_web_lg.load()


# 生成语料文件用于词向量训练
def generate_corpus(path):
    train = pd.read_csv(path + "SDP_train.csv", sep=',', header=0)
    test = pd.read_csv(path + "SDP_test.csv", sep=',', header=0)
    test_corpora = [
        clean_sentence(test.loc[i, 'citation_context'].replace("#AUTHOR_TAG", test.loc[i, 'cited_title'])
                       + test.loc[i, 'citing_title'])
        for i in range(len(test))]
    test_corpora = tagger.pipe([test.loc[i, 'citation_context'] for i in range(len(test))])
    train_corpora = [
        clean_sentence(train.loc[i, 'citation_context'].replace("#AUTHOR_TAG", train.loc[i, 'cited_title'])
                       + train.loc[i, 'citing_title'])
        for i in range(len(train))]
    corpora = test_corpora + train_corpora
    with open("./datasets/corpora.txt", 'w', encoding='utf-8') as f:
        for line in corpora:
            f.write(line + "\n")


def formatted(path, label):
    """
    :param label: 分类标签名称
    :param path:数据路径
    生成预训练所需要的数据
    """
    data = pd.read_csv(path + "SDP_train.csv", sep=',', header=0)

    output = [(
        clean_sentence(data.loc[i, 'citation_context'].replace("#AUTHOR_TAG", data.loc[i, 'cited_title'])
                       + data.loc[i, 'citing_title']),
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

    test = [clean_sentence(data.loc[i, 'citation_context'].replace("#AUTHOR_TAG", data.loc[i, 'cited_title'])
                           + data.loc[i, 'citing_title'])
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


def clean_sentence(s: str):
    return "".join(re.findall('[a-zA-Z ]', s))


formatted("datasets/intent/", 'citation_class_label')
# formatted("datasets/influence/", 'citation_influence_label')
formatted_test("datasets/intent/")
# formatted_test("datasets/influence/")

# generate_corpus("datasets/intent/")
# x = []
# y = []
# with open("./datasets/practice_data/3c_data.tsv", 'r', encoding='utf-8') as f:
#     for line in f.readlines():
#         x.append(line.strip().split("\t")[0])
#         y.append(line.strip().split("\t")[1])
# with open("./datasets/practice_data/3c.txt", 'w', encoding='utf-8') as f:
#     # f.write("label\ttext_a\n")
#     # for i in range(len(x)):
#     #     f.write(y[i]+"\t"+x[i]+"\n")
#     for i in x:
#         f.write(i+"\n")

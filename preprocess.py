import pandas as pd
import re


def formatted(path, label):
    """
    :param label: 分类标签名称
    :param path:数据路径
    生成预训练所需要的数据
    """
    data = pd.read_csv(path, sep=',', header=0)

    output = [(clean_sentence(data.loc[i, 'citation_context'].
                              replace("#AUTHOR_TAG", data.loc[i, 'citing_title'])),
               str(data.loc[i, label]))
              for i in range(len(data))]

    with open("/".join(path.split("/")[:2]) + "/train.tsv", 'w', encoding='utf-8') as f:
        for line in output:
            f.write("\t".join(line) + "\n")


def clean_sentence(s: str):
    return "".join(re.findall('[a-zA-Z ]', s))


formatted("datasets/intent/SDP_train.csv",
          'citation_class_label')
formatted("datasets/influence/SDP_train.csv",
          'citation_influence_label')

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

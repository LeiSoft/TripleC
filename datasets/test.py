list_ = []
with open("data.tsv", 'r', encoding='utf-8') as f:
    for line in f.readlines():
        list_.append(line.strip().split("\t")[2])

with open("corpora_add.txt", 'w', encoding='utf-8') as f:
    for i in list_:
        f.write(i+"\n")
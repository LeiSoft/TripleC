from gensim.models import FastText
from gensim.models import word2vec
import logging
import argparse


def train(args):
    tool = args.model
    assert tool == 'fasttext' or tool == 'word2vec', 'you can choose: [word2vec, fasttext]'
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    sentences = word2vec.LineSentence(u'../../datasets/corpora_add.txt')
    if tool == 'fasttext':
        _model = FastText(sentences, size=args.D, iter=30, min_count=2, word_ngrams=1)
    else:
        _model = word2vec.Word2Vec(sentences, size=args.D, iter=30, min_count=2)
    _model.save("./"+tool+"/output")


if __name__ == '__main__':
    parse = argparse.ArgumentParser(description="choose the way to train word vectors")
    parse.add_argument("--model", required=True, type=str, help="[word2vec, fasttext]")
    parse.add_argument("--dimension", dest='D', default=300)

    _args = parse.parse_args()
    train(_args)

    if _args.model == 'fasttext':
        model = FastText.load("./fasttext/output")
    else:
        model = word2vec.Word2Vec.load("./word2vec/output")
    dic = {}
    for i in model.wv.vocab:
        dic[i] = [str(n) for n in model.wv[i].tolist()]

    with open("./3C.vec", 'w', encoding='utf-8') as f:
        f.write(str(len(dic.keys()))+" "+str(_args.D)+"\n")
        for item in dic.items():
            f.write(item[0]+" "+" ".join(item[1])+"\n")

    with open("./vocab.txt", 'w', encoding='utf-8') as f:
        for k in dic.keys():
            f.write(k+"\n")
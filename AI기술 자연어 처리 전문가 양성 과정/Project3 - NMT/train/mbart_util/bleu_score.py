import os

try:
    import konlpy
except:
    os.system(
        "curl -s https://raw.githubusercontent.com/teddylee777/machine-learning/master/99-Misc/01-Colab/mecab-colab.sh | bash"
    )
    import konlpy

from nltk.translate.bleu_score import (
    sentence_bleu,
    SmoothingFunction,
)

mecab_tagger = konlpy.tag.Mecab()


def bleu_upto(reference, hypothesis, n_gram):
    res = []
    for i in range(1, n_gram + 1):
        res.append(calc_bleu_ngram(reference, hypothesis, i))
    return res


def corpuswise_bleu(predicts, gts, n_gram=4):
    res_predict = []
    res_gt = []

    for predict in predicts:
        res_predict.append([i for i in mecab_tagger.morphs(predict)])

    for gt in gts:
        res_gt.append([i for i in mecab_tagger.morphs(gt)])

    return bleu_upto(res_gt, res_predict, n_gram)


def calc_bleu_ngram(reference, hypothesis, n_gram):
    score = 0.0
    ratio = 1 / n_gram

    cc = SmoothingFunction()

    for refer, hypo in zip(reference, hypothesis):
        # refer.index()
        score += sentence_bleu([refer], hypo, (ratio,) * n_gram, cc.method1)

    return score / len(reference)

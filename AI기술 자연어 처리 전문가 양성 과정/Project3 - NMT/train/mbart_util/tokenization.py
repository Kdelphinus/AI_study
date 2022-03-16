import os

try:
    import konlpy
except:
    os.system(
        "curl -s https://raw.githubusercontent.com/teddylee777/machine-learning/master/99-Misc/01-Colab/mecab-colab.sh | bash"
    )
    import konlpy

from matplotlib import pyplot as plt
from typing import List, Dict
import pandas as pd
import random
from nltk.tokenize import word_tokenize
from collections import Counter
import numpy as np

mecab_tagger = konlpy.tag.Mecab()


def read_csv(fname: str) -> List[Dict[str, str]]:
    """
    학습 데이터를 읽는 함수입니다.
    """
    df = pd.read_csv(fname)
    sid = df["sid"].to_list()
    src = df["en"].to_list()
    if "ko" in df:
        tgt = df["ko"].to_list()
        assert len(sid) == len(src) == len(tgt)
        return {"sid": sid, "src": src, "tgt": tgt}
    else:
        assert len(sid) == len(src)
        return {"sid": sid, "src": src}


def show_random_sample(data, num_examples: int = 1):
    for _ in range(num_examples):
        random_index = random.randint(0, len(data["sid"]))
        print("ID: {}".format(data["sid"][random_index]))
        print("Source(Eng): {}".format(data["src"][random_index]))
        if "tgt" in data:
            print("Target(Kor): {}".format(data["tgt"][random_index]))
        print()


def tokenize(sent: str, is_eng: bool = False) -> List[str]:
    """
    하나의 문장(string)이 들어오면 이를 토크나이징 하는 함수입니다.
    is_eng가 True이면 입력 문장이 영어, False이면 한글로 가정합니다.
    """
    assert isinstance(sent, str)
    if is_eng:
        return word_tokenize(sent)
    else:
        return mecab_tagger.morphs(sent)


def tokenize_entire_sentences(
    sents: List[str], is_eng: bool = False
) -> List[List[str]]:
    assert all([isinstance(sent, str) for sent in sents])
    tokenized_sentences = [tokenize(sent) for sent in sents]
    return tokenized_sentences


def get_average_token_num(tokenized_data: List[List[str]]):
    token_num = []
    for el in tokenized_data:
        token_num.append(len(el))
    return sum(token_num) / len(token_num)


def draw_length_distribution(tokenized_data: List[List[str]], data_name: str):
    token_num = []
    for el in tokenized_data:
        token_num.append(len(el))

    plt.hist(token_num, bins=30)
    plt.title(data_name)
    plt.xlabel("Number of tokens")
    plt.show()


def get_token_count(tokenized_data: List[List[str]]):
    counter = Counter()
    for tokenized_sent in tokenized_data:
        counter.update(tokenized_sent)

    return counter


def draw_token_frequency_distribution(token_counter, title):
    print("=" * 10)
    print("{}에서 가장 많이 사용된 10개의 단어: {}".format(title, token_counter.most_common(10)))
    print("{}의 Unique한 토큰 개수: {}".format(title, len(token_counter)))
    token_frequency = sorted(list(token_counter.values()), reverse=True)
    token_log_frequency = [np.log10(e) for e in token_frequency]
    x_axis = [_ for _ in range(len(token_frequency))]
    plt.plot(x_axis, token_log_frequency)
    plt.title(title)
    plt.xlabel("Rank")
    plt.ylabel("Log10(Frequency)")
    plt.show()

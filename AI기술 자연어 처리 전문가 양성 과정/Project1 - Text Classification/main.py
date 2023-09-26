LEARNING_RATE = 1e-5
BATCH_SIZE = 128
EPOCH = 5
# MODEL = "roberta-large-mnli"
MODEL = "bert-base-uncased"

import pip


def install(package):
    """colab에 없는 패키지를 설치하는 함수"""
    if hasattr(pip, "main"):
        pip.main(["install", package])
    else:
        pip._internal.main(["install", package])


install("wandb")
install("transformers")

import os
import pdb
import wandb
import argparse
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict

import torch
from torch.nn.utils.rnn import pad_sequence

import numpy as np
from tqdm import tqdm, trange

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoConfig,
    AdamW,
)

import pandas as pd


class SentimentDataset(object):
    """train data, dev data 전처리 class"""

    def __init__(self, tokenizer, pos, neg):
        self.tokenizer = tokenizer
        self.data = []
        self.label = []

        # 토큰화 된 문장을 각각의 토큰으로 분리하여 리스트로 저장
        # 같은 index에 label도 저장
        for pos_sent in pos:
            self.data += [self._cast_to_int(pos_sent.strip().split())]
            self.label += [[1]]
        for neg_sent in neg:
            self.data += [self._cast_to_int(neg_sent.strip().split())]
            self.label += [[0]]

    def _cast_to_int(self, sample):
        return [int(word_id) for word_id in sample]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        return np.array(sample), np.array(self.label[index])


class SentimentTestDataset(object):
    """test data 전처리 class"""

    def __init__(self, tokenizer, test):
        self.tokenizer = tokenizer
        self.data = []

        for sent in test:
            self.data += [self._cast_to_int(sent.strip().split())]

    def _cast_to_int(self, sample):
        return [int(word_id) for word_id in sample]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        return np.array(sample)


def collate_fn_sentiment(samples):
    """train data, dev data의 batch 간의 데이터 길이를 맞추는 함수"""
    input_ids, labels = zip(*samples)
    max_len = max(len(input_id) for input_id in input_ids)
    sorted_indices = np.argsort([len(input_id) for input_id in input_ids])[::-1]

    # 실제 문장의 토큰은 1, 패딩된 토큰은 0
    attention_mask = torch.tensor(
        [
            [1] * len(input_ids[index]) + [0] * (max_len - len(input_ids[index]))
            for index in sorted_indices
        ]
    )
    # 가장 긴 문장 길이에 맞춰 패딩
    input_ids = pad_sequence(
        [torch.tensor(input_ids[index]) for index in sorted_indices], batch_first=True
    )
    # 문장 정보 저장(앞에 문장 0, 뒤에 문장 1)
    token_type_ids = torch.tensor(
        [[0] * len(input_ids[index]) for index in sorted_indices]
    )
    # position_id 생성
    position_ids = torch.tensor(
        [list(range(len(input_ids[index]))) for index in sorted_indices]
    )
    # 문장과 위치가 같도록 label 정렬
    labels = torch.tensor(np.stack(labels, axis=0)[sorted_indices])

    return input_ids, attention_mask, token_type_ids, position_ids, labels


def compute_acc(predictions, target_labels):
    """accuracy를 계산 하는 함수"""
    return (np.array(predictions) == np.array(target_labels)).mean()


def collate_fn_sentiment_test(samples):
    """test data의 batch 간의 데이터 길이를 맞추는 함수"""
    input_ids = samples
    max_len = max(len(input_id) for input_id in input_ids)
    attention_mask = torch.tensor(
        [
            [1] * len(input_id) + [0] * (max_len - len(input_id))
            for input_id in input_ids
        ]
    )
    input_ids = pad_sequence(
        [torch.tensor(input_id) for input_id in input_ids], batch_first=True
    )
    token_type_ids = torch.tensor([[0] * len(input_id) for input_id in input_ids])
    position_ids = torch.tensor([list(range(len(input_id))) for input_id in input_ids])

    return input_ids, attention_mask, token_type_ids, position_ids


def make_id_file(task, tokenizer):
    """train data, dev data를 전처리하는 함수"""

    def make_data_strings(file_name):
        """문장을 토큰화시키는 함수"""
        data_strings = []
        with open(os.path.join("data", file_name), "r", encoding="utf-8") as f:
            id_file_data = [tokenizer.encode(line.lower()) for line in f.readlines()]
        for item in id_file_data:
            data_strings.append(" ".join([str(k) for k in item]))
        return data_strings

    print("it will take some times...")
    train_pos = make_data_strings("sentiment.train.1")
    train_neg = make_data_strings("sentiment.train.0")
    dev_pos = make_data_strings("sentiment.dev.1")
    dev_neg = make_data_strings("sentiment.dev.0")

    print("make id file finished!")
    return train_pos, train_neg, dev_pos, dev_neg


def make_id_file_test(tokenizer, test_dataset):
    """

    Args:
        tokenizer:
        test_dataset:

    Returns:

    """
    data_strings = []
    id_file_data = [tokenizer.encode(sent.lower()) for sent in test_dataset]
    for item in id_file_data:
        data_strings.append(" ".join([str(k) for k in item]))
    return data_strings


def train(train_loader, dev_loader, model, optimizer, epoch, device):
    """ model을 train하는 함수

    Args:
        model: 학습할 모델
        optimizer: 사용할 optimizer
        epoch: 현재 epoch(출력용)
        device: 사용할 device

    """
    model.train()
    with tqdm(train_loader, unit="batch") as tepoch:
        for (
            iteration,
            (input_ids, attention_mask, token_type_ids, position_ids, labels),
        ) in enumerate(tepoch):
            tepoch.set_description(f"Epoch {epoch}")
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            position_ids = position_ids.to(device)
            labels = labels.to(device, dtype=torch.long)

            optimizer.zero_grad()

            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                labels=labels,
            )

            loss = output.loss
            loss.backward()

            optimizer.step()

            tepoch.set_postfix(loss=loss.item())

            # epoch 중간 지점에서 evaluation 진행
            if iteration != 0 and iteration % int(len(train_loader) / 2) == 0:
                dev(dev_loader, model, device)


def dev(dev_loader, model, device):
    """model을 evaluation하는 함수"""
    lowest_valid_loss = 9999.0

    with torch.no_grad():
        model.eval()
        valid_losses = []
        predictions = []
        target_labels = []
        for (input_ids, attention_mask, token_type_ids, position_ids, labels) in tqdm(
            dev_loader, desc="Eval", position=1, leave=None
        ):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            position_ids = position_ids.to(device)
            labels = labels.to(device, dtype=torch.long)

            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                labels=labels,
            )

            logits = output.logits
            loss = output.loss
            valid_losses.append(loss.item())

            batch_predictions = [
                0 if example[0] > example[1] else 1 for example in logits
            ]
            batch_labels = [int(example) for example in labels]

            predictions += batch_predictions
            target_labels += batch_labels

    acc = compute_acc(predictions, target_labels)
    valid_loss = sum(valid_losses) / len(valid_losses)
    wandb.log(
        {"dev Accuracy": 100.0 * acc, "dev Loss": valid_loss,}
    )

    if lowest_valid_loss > valid_loss:
        print("Acc for model which have lower valid loss: ", acc)


def test(test_loader, model, device):
    """ 주어진 데이터의 답을 예측하는 함수

    Args:
        model: 사용할 model
        device: 사용할 device

    Returns:
        predictions: 예측한 값

    """
    with torch.no_grad():
        model.eval()
        predictions = []
        for input_ids, attention_mask, token_type_ids, position_ids in tqdm(
            test_loader, desc="Test", position=1, leave=None
        ):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            position_ids = position_ids.to(device)

            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
            )

            logits = output.logits
            batch_predictions = [
                0 if example[0] > example[1] else 1 for example in logits
            ]
            predictions += batch_predictions

    return predictions


def main():
    # os.system("wandb login")
    os.system("wandb login --relogin")  # 처음 이후는 재로그인

    wandb.init(project="Project 1")
    wandb.run.name = (
        f"Model: {MODEL}, Batch size: {BATCH_SIZE}, Learning rate: {LEARNING_RATE}"
    )
    wandb.run.save()

    # pre-processing
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    train_pos, train_neg, dev_pos, dev_neg = make_id_file("yelp", tokenizer)

    train_dataset = SentimentDataset(tokenizer, train_pos, train_neg)
    dev_dataset = SentimentDataset(tokenizer, dev_pos, dev_neg)

    train_batch_size = BATCH_SIZE
    eval_batch_size = BATCH_SIZE

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        collate_fn=collate_fn_sentiment,
        pin_memory=True,
        num_workers=2,
    )
    dev_loader = torch.utils.data.DataLoader(
        dev_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn_sentiment,
        num_workers=2,
    )

    # random seed
    random_seed = 42
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(torch.cuda.current_device()))
    # else: # gpu(colab)에 연결 되지 않았을 때, 중단
    #     print("not connected")
    #     return

    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    model.to(device)

    model.train()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda epoch: 0.95 ** epoch
    )

    wandb.watch(model)

    # training
    train_epoch = EPOCH
    for epoch in range(1, train_epoch + 1):
        train(train_loader, dev_loader, model, optimizer, epoch, device)
        scheduler.step()

    # test
    test_df = pd.read_csv("data/test_no_label.csv")
    test_dataset = test_df["Id"]
    test_data = make_id_file_test(tokenizer, test_dataset)
    test_dataset = SentimentTestDataset(tokenizer, test_data)

    test_batch_size = BATCH_SIZE
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        collate_fn=collate_fn_sentiment_test,
        num_workers=2,
    )

    predictions = test(test_loader, model, device)
    test_df["Category"] = predictions
    test_df.to_csv(
        f"submission_roberta(lr:{LEARNING_RATE}, epoch:{EPOCH}, batch:{BATCH_SIZE}).csv",
        index=False,
    )


if __name__ == "__main__":
    main()

# Hyperparameter
MODEL = "monologg/koelectra-base-v3-finetuned-korquad"
MAX_LENGTH = 512
EPOCH = 10
LEARNING_RATE = 5e-5
ACCUMULATION = 4
BATCH_SIZE = 32
FLAG = False  # overfitting이 일어났는지 확인하는 변수


# colab에 없는 패키지 설치
import os

os.system("pip install transformers")
os.system("pip install wandb")

import operator
from typing import List, Tuple, Dict, Any, Sequence
import csv
import json
import wandb
import random
from collections import Counter
from itertools import chain
from tqdm import tqdm
from statistics import mean
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    get_linear_schedule_with_warmup,
)


class KoMRC:
    """Json 파일의 데이터를 가공하는 class"""

    def __init__(self, data, indices: List[Tuple[int, int, int]]):
        self._data = data
        self._indices = indices

    # Json을 불러오는 메소드
    @classmethod
    def load(cls, file_path: str):
        with open(file_path, "r", encoding="utf-8") as fd:
            data = json.load(fd)

        indices = []
        for d_id, document in enumerate(data["data"]):
            for p_id, paragraph in enumerate(document["paragraphs"]):
                for q_id, _ in enumerate(paragraph["qas"]):
                    indices.append((d_id, p_id, q_id))

        return cls(data, indices)

    # 데이터 셋을 잘라내는 메소드
    @classmethod
    def split(cls, dataset, eval_ratio: float = 0.1, seed=42):
        indices = list(dataset._indices)
        random.seed(seed)
        random.shuffle(indices)
        train_indices = indices[int(len(indices) * eval_ratio) :]
        eval_indices = indices[: int(len(indices) * eval_ratio)]

        return cls(dataset._data, train_indices), cls(dataset._data, eval_indices)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        d_id, p_id, q_id = self._indices[index]
        paragraph = self._data["data"][d_id]["paragraphs"][p_id]

        context = paragraph["context"]
        qa = paragraph["qas"][q_id]

        guid = qa["guid"]
        question = qa["question"]
        answers = qa["answers"]

        # 가장 짧은 답 하나만 사용
        if answers is not None:
            # 들어가야 할 세 개의 토큰과 질문을 제외한 길이
            min_len = MAX_LENGTH - 3 - len(question)
            min_idx = 0
            for idx, answer in enumerate(answers):
                if len(answer["text"]) < min_len:
                    min_idx = idx

            answer = answers[min_idx]
            answers = [answer]

        return {
            "guid": guid,
            "context": context,
            "question": question,
            "answers": answers,
        }

    def __len__(self) -> int:
        return len(self._indices)


class TokenizedKoMRC(KoMRC):
    """데이터를 전처리하는 class"""

    def __init__(self, data, indices: List[Tuple[int, int, int]]) -> None:
        super().__init__(data, indices)
        self._tokenizer = AutoTokenizer.from_pretrained(MODEL)

    # 데이터를 토큰화하고 토큰이 context에 위치한 자리 저장
    def _tokenize_with_position(
        self, sentence: str
    ) -> List[Tuple[str, Tuple[int, int]]]:
        position = 0
        tokens = []
        for idx, token in enumerate(self._tokenizer.tokenize(sentence)):
            # context에서 위치를 구할 때, 토큰 앞에 ##은 제거
            morph = token[2:] if "##" == token[:2] else token

            # [UNK]토큰이 나왔을 때
            if morph == "[UNK]":  # [UNK]의 앞쪽 위치 저장
                if sentence[position] == " ":
                    position += 1
                tokens.append((token, (position, position + 1)))
                position += 1
            else:
                position = sentence.find(morph, position)
                tokens.append((token, (position, position + len(morph))))
                position += len(morph)

            # [UNK]의 뒤쪽 위치 저장
            if idx > 0 and tokens[idx - 1][0] == "[UNK]":
                tokens[idx - 1] = list(tokens[idx - 1])
                if idx == 1:  # 첫번째 토큰이 [UNK]일 때
                    if tokens[idx][0] == "[UNK]":
                        tokens[idx - 1][1] = (0, position - 1)
                    else:
                        tokens[idx - 1][1] = (0, position - len(morph))
                else:
                    # 두 번 연속 [UNK]가 나왔을 때
                    if tokens[idx][0] == "[UNK]":
                        tokens[idx - 1][1] = (tokens[idx - 1][1][0], position - 1)
                    else:
                        # 현재 토큰과 [UNK] 사이에 공백이 있을 때
                        if sentence[tokens[idx][1][0] - 1] == " ":
                            tokens[idx - 1][1] = (
                                tokens[idx - 1][1][0],
                                tokens[idx][1][0] - 1,
                            )

                            tokens[idx - 1][0] = sentence[
                                tokens[idx - 1][1][0] : tokens[idx - 1][1][1]
                            ]
                        # 현재 토큰과 [UNK] 사이에 공백이 없을 때
                        else:
                            tokens[idx - 1][1] = (
                                tokens[idx - 1][1][0],
                                tokens[idx][1][0],
                            )

                            tokens[idx - 1][0] = (
                                "##"
                                + sentence[
                                    tokens[idx - 1][1][0] : tokens[idx - 1][1][1]
                                ]
                            )
                tokens[idx - 1] = tuple(tokens[idx - 1])

        # [UNK]가 마지막에 있을 때
        if tokens[-1][0] == "[UNK]":
            tokens[-1] = list(tokens[-1])
            tokens[-1][1] = (tokens[-1][1][0], len(sentence))
            tokens[-1][0] = sentence[tokens[-1][1][0] :]
            tokens[-1] = tuple(tokens[-1])

        return tokens

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = super().__getitem__(index)

        context, position = zip(*self._tokenize_with_position(sample["context"]))
        context, position = list(context), list(position)
        question = self._tokenizer.tokenize(sample["question"])

        if sample["answers"] is not None:
            answers = []
            for idx, answer in enumerate(sample["answers"]):
                for start, (position_start, position_end) in enumerate(position):
                    if position_start <= answer["answer_start"] < position_end:
                        break
                else:
                    raise "start error"

                target = "".join(answer["text"].split(" "))
                source = ""
                for end, morph in enumerate(context[start:], start):
                    # [UNK]가 나오면 end를 start로 저장
                    if morph == "[UNK]":
                        break
                    morph = morph[2:] if "##" == morph[:2] else morph
                    source += morph
                    if target in source:
                        break
                else:
                    raise "end error"

                answers.append({"start": start, "end": end})
        else:
            answers = None

        return {
            "guid": sample["guid"],
            "context_original": sample["context"],
            "context_position": position,
            "question_original": sample["question"],
            "context": context,
            "question": question,
            "answers": answers,
        }


class Indexer:
    """모델에 입력할 input 값들을 구성하는 class"""

    def __init__(
        self,
        id2token: List[str],
        max_length: int = MAX_LENGTH,  # 현재 모델의 embedding 크기
        pad: str = "[PAD]",
        unk: str = "[UNK]",
        cls: str = "[CLS]",
        sep: str = "[SEP]",
        mask: str = "[MASK]",
    ):
        self.pad = pad
        self.unk = unk
        self.cls = cls
        self.sep = sep
        self.mask = mask
        self.max_length = max_length

        self.id2token = id2token
        self.token2id = {
            token: token_id for token_id, token in enumerate(self.id2token)
        }

    @property
    def vocab_size(self):
        return len(self.id2token)

    @property
    def pad_id(self):
        return self.token2id[self.pad]

    @property
    def unk_id(self):
        return self.token2id[self.unk]

    @property
    def cls_id(self):
        return self.token2id[self.cls]

    @property
    def sep_id(self):
        return self.token2id[self.sep]

    @property
    def mask_id(self):
        return self.token2id[self.mask]

    @classmethod
    def build_vocab(cls, dataset: TokenizedKoMRC, min_freq: int = 3):
        counter = Counter(
            chain.from_iterable(
                sample["context"] + sample["question"]
                for sample in tqdm(dataset, desc="Counting Vocab")
            )
        )

        # tokenizer에 저장된 vocab을 불러옴
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        vocab = sorted(tokenizer.get_vocab().items(), key=operator.itemgetter(1))
        base_vocab = [v[0] for v in vocab][:34000]

        # tokenizer에 없는 토큰을 추가
        for word, count in counter.items():
            if count >= min_freq and word not in base_vocab:
                base_vocab.append(word)

        return cls(base_vocab)

    # token_id를 단어로 변환하는 함수
    def decode(self, token_ids: Sequence[int]):
        return [
            self.id2token[token_id][2:]
            if "##" == self.id2token[token_id][:2]
            else self.id2token[token_id]
            for token_id in token_ids
        ]
        # answer = ""
        # for token_id in token_ids:
        #     token = [self.id2token[token_id]]
        #     answer += token[2:] if token[:2] == "##" else token
        # return answer

    def sample2ids(self, sample: Dict[str, Any],) -> Dict[str, Any]:
        context = [self.token2id.get(token, self.unk_id) for token in sample["context"]]
        question = [
            self.token2id.get(token, self.unk_id) for token in sample["question"]
        ]

        # 답이 주어진 경우
        if sample["answers"] is not None:
            max_seq_len = MAX_LENGTH - len(question) - 3
            answer = sample["answers"][0]

            # context가 최대 길이를 넘어가면 잘라준다.
            if len(context) > max_seq_len:
                # 답의 위치를 중간에 두고 자름
                start_token = (
                    0
                    if answer["start"] < max_seq_len // 2
                    else answer["start"] - max_seq_len // 2
                )
                end_token = (
                    answer["start"] + max_seq_len // 2
                    if answer["start"] + max_seq_len // 2 < len(context)
                    else len(context)
                )
                new_context = context[start_token:end_token]

                # 답의 위치도 잘린 context에 맞춰서 변경
                start = answer["start"] - start_token + len(question) + 2
                end = answer["end"] - start_token + len(question) + 2

                input_ids = (
                    [self.cls_id]
                    + question
                    + [self.sep_id]
                    + new_context
                    + [self.sep_id]
                )
                # electra 모델은 중간에 있는 [sep] 토큰까지 0으로 취급
                token_type_ids = [0] * (len(question) + 2) + [1] * (
                    len(new_context) + 1
                )

            # context가 최대 길이를 넘지 않으면 그대로 사용한다.
            else:
                # 앞에 들어갈 질문의 길이와 두 개의 토큰
                start = answer["start"] + len(question) + 2
                end = answer["end"] + len(question) + 2

                input_ids = (
                    [self.cls_id] + question + [self.sep_id] + context + [self.sep_id]
                )
                # electra 모델은 중간에 있는 [sep] 토큰까지 0으로 취급
                token_type_ids = [0] * (len(question) + 2) + [1] * (len(context) + 1)
        # 답이 없는 경우(test)
        else:
            # 답을 모르므로 앞에서 부터 자른다.
            context = context[: self.max_length - len(question) - 3]

            # electra 모델은 중간에 있는 [sep] 토큰까지 0으로 취급
            token_type_ids = [0] * (len(question) + 2) + [1] * (len(context) + 1)
            input_ids = (
                [self.cls_id] + question + [self.sep_id] + context + [self.sep_id]
            )

            # 답이 없음
            start = None
            end = None

        return {
            "guid": sample["guid"],
            "context": sample["context_original"],
            "question": sample["question_original"],
            "position": sample["context_position"],
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "start": start,
            "end": end,
        }


class IndexerWrappedDataset(Indexer):
    """attention mask를 추가하는 class"""

    def __init__(self, dataset: TokenizedKoMRC, indexer: Indexer) -> None:
        self._dataset = dataset
        self._indexer = indexer

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self._indexer.sample2ids(self._dataset[index])
        sample["attention_mask"] = [1] * len(sample["input_ids"])

        return sample


class Collator:
    """학습할 수 있도록 데이터를 전처리하는 함수"""

    def __init__(self, indexer: Indexer) -> None:
        self._indexer = indexer

    def __call__(self, samples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        samples = {key: [sample[key] for sample in samples] for key in samples[0]}

        for key in "start", "end":
            # 답이 없는 경우(test)
            if samples[key] is None:
                samples[key] = None
            # 답이 주어진 경우
            else:
                samples[key] = torch.tensor(samples[key], dtype=torch.long)

        # 각 데이터들에 padding
        for key in "input_ids", "attention_mask", "token_type_ids":
            samples[key] = pad_sequence(
                [torch.tensor(sample, dtype=torch.long) for sample in samples[key]],
                batch_first=True,
                padding_value=self._indexer.pad_id,
            )

        return samples


def train(
    model,
    train_loader,
    dev_loader,
    optimizer,
    scheduler,
    device,
    train_epoch,
    accumulation,
):
    """모델을 학습시키는 함수"""
    step = 0  # accumulation에 사용할 변수
    min_dev_loss = 99  # overfitting 방지를 위해 사용할 변수
    for epoch in range(1, train_epoch + 1):
        print(f"------Train {epoch}------")
        model.train()
        running_loss = 0.0
        losses = []
        progress_bar = tqdm(train_loader, desc="Train")

        # overfitting이 발생해 train이 멈추면 직전 모델을 사용하기 위해 저장
        model.save_pretrained(f"models/model_last")
        for batch in progress_bar:
            del batch["guid"], batch["context"], batch["question"], batch["position"]
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            start_positions = batch["start"].to(device)
            end_positions = batch["end"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                start_positions=start_positions,
                end_positions=end_positions,
            )

            loss = outputs.loss
            (loss / accumulation).backward()
            running_loss += loss.item()
            del batch, start_positions, end_positions, outputs, loss

            step += 1
            if step % accumulation:
                continue

            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            losses.append(running_loss / accumulation)
            running_loss = 0.0
            progress_bar.set_postfix(loss=losses[-1])
        train_loss = mean(losses)
        print(f"train loss: {train_loss:.3f}")

        dev_loss = dev(model, dev_loader, device)
        print(f"Evaluation score: {dev_loss:.3f}")
        wandb.log({"dev Loss": dev_loss, "train loss": train_loss})

        # overfitting이 생기면 바로 종료
        if min_dev_loss < dev_loss:
            FLAG = True
            print("overfitting!")
            return [epoch - 1, min_dev_loss]
        min_dev_loss = dev_loss
        model.save_pretrained(f"models/model_current")

    return [epoch, min_dev_loss]


def dev(model, dev_loader, device):
    """모델 학습을 평가하는 함수"""
    model.eval()
    losses = []
    for batch in tqdm(dev_loader, desc="Evaluation", unit="batch"):
        del batch["guid"], batch["context"], batch["question"], batch["position"]
        batch = {key: value.to(device) for key, value in batch.items()}

        with torch.no_grad():
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch["token_type_ids"],
                start_positions=batch["start"],
                end_positions=batch["end"],
            )

        loss = outputs.loss

        losses.append(loss.item())
        del batch, outputs, loss

    return mean(losses)


def test(model, test_dataset, device):
    """모델을 이용하여 답을 예측하는 함수"""
    model.eval()
    os.makedirs("out", exist_ok=True)
    with torch.no_grad(), open("out/baseline.csv", "w") as fd:
        writer = csv.writer(fd)
        writer.writerow(["Id", "Predicted"])

        rows = []
        for sample in tqdm(test_dataset, "Testing"):
            input_ids, token_type_ids = [
                torch.tensor(sample[key], dtype=torch.long, device=device)
                for key in ("input_ids", "token_type_ids")
            ]

            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids[None, :], token_type_ids=token_type_ids[None, :]
                )
            start_logits = outputs["start_logits"]
            end_logits = outputs["end_logits"]

            start_logits.squeeze_(0), end_logits.squeeze_(0)

            # context에서 제외할 토큰은 맨 뒤에 있는 sep 토큰 하나
            start_prob = start_logits[token_type_ids.bool()][:-1].softmax(-1)
            end_prob = end_logits[token_type_ids.bool()][:-1].softmax(-1)
            probability = torch.triu(start_prob[:, None] @ end_prob[None, :])
            index = torch.argmax(probability).item()

            start = index // len(end_prob)
            end = index % len(end_prob)

            start = sample["position"][start][0]
            end = sample["position"][end][1]
            if end < start or len(sample["context"][start:end]) > 10:
                answer = ""
            else:
                answer = sample["context"][start:end]

            rows.append([sample["guid"], answer])

        writer.writerows(rows)


def semi_test(model, indexed_train_dataset, device):
    """모델이 잘 학습됐는지 train 데이터로 답을 예측하는 함수"""
    model.eval()
    for idx, sample in zip(range(1, 4), indexed_train_dataset):
        print(f"------postprocessing {idx}------")
        print("Question:", sample["question"])

        input_ids, token_type_ids = [
            torch.tensor(sample[key], dtype=torch.long, device=device)
            for key in ("input_ids", "token_type_ids")
        ]

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids[None, :], token_type_ids=token_type_ids[None, :]
            )

        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

        start_logits.squeeze_(0), end_logits.squeeze_(0)

        start_prob = start_logits[token_type_ids.bool()][:-1].softmax(-1)
        end_prob = end_logits[token_type_ids.bool()][:-1].softmax(-1)
        probability = torch.triu(start_prob[:, None] @ end_prob[None, :])
        index = torch.argmax(probability).item()

        start = index // len(end_prob)
        end = index % len(end_prob)

        start = sample["position"][start][0]
        end = sample["position"][end][1]

        print("Answer:", sample["context"][start:end])


def main():
    # 전처리
    dataset = TokenizedKoMRC.load("data/train.json")
    train_dataset, dev_dataset = TokenizedKoMRC.split(dataset)

    indexer = Indexer.build_vocab(dataset)
    indexed_train_dataset = IndexerWrappedDataset(train_dataset, indexer)
    indexed_dev_dataset = IndexerWrappedDataset(dev_dataset, indexer)

    collator = Collator(indexer)
    train_loader = DataLoader(
        indexed_train_dataset,
        batch_size=BATCH_SIZE // ACCUMULATION,
        shuffle=True,
        collate_fn=collator,
        num_workers=2,
    )
    dev_loader = DataLoader(
        indexed_dev_dataset,
        batch_size=BATCH_SIZE // ACCUMULATION,
        shuffle=False,
        collate_fn=collator,
        num_workers=2,
    )

    # 학습 및 예측
    grid_search = []
    os.system("wandb login")
    wandb.init(project="Project 2")
    wandb.run.name = f"lr:{LEARNING_RATE}, epoch:{EPOCH}"
    wandb.run.save()
    torch.manual_seed(42)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), LEARNING_RATE)
    total_steps = len(train_loader) * EPOCH
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps,
    )
    wandb.watch(model)

    result = train(
        model,
        train_loader,
        dev_loader,
        optimizer,
        scheduler,
        device,
        EPOCH,
        ACCUMULATION,
    )

    # semi_test(model, indexed_train_dataset, device)
    #
    # if FLAG: # overfitting
    # model = AutoModelForQuestionAnswering.from_pretrained("models/model_last")
    # else: # overfitting 없음
    # model = AutoModelForQuestionAnswering.from_pretrained("models/model_current")
    # model.to(device)
    # test_dataset = TokenizedKoMRC.load("data/test.json")
    # test_dataset = IndexerWrappedDataset(test_dataset, indexer)
    # test(model, test_dataset, device)
    # print("Done!")
    #


if __name__ == "__main__":
    main()

import dataclasses
from dataclasses import dataclass
import json
import os
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from typing import Optional, List
import pandas as pd

data_path = "data/"
pad_index = 1
max_length = 1024
DATAMAP = {
    "train": "train_added.csv",
    "dev": "dev_added.csv",
    "test1": "test.csv",
    "test2": "test2.csv",
}

# DATAMAP = {
#     "train": "train.csv",
#     "dev": "dev.csv",
#     "test1": "test.csv",
#     "test2": "test2.csv",
# }


@dataclass
class NMTExample(object):
    guid: str
    input_ids: List[int]
    trg_ids: Optional[List[int]] = None

    def to_dict(self):
        return dataclasses.asdict(self)

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2) + "\n"


def load_dataset(data_dir, source_tokenizer, target_tokenizer):
    cache_path = os.path.join(data_dir, "cache")

    if not os.path.isdir(cache_path):
        os.makedirs(cache_path)

        train_pairs = get_pairs_from_dataset(data_dir, "train")
        train_examples = convert_data_to_examples(
            source_tokenizer, target_tokenizer, train_pairs
        )
        pd.to_pickle(train_examples, os.path.join(cache_path, "train.pkl"))

        dev_pairs = get_pairs_from_dataset(data_dir, "dev")
        dev_examples = convert_data_to_examples(
            source_tokenizer, target_tokenizer, dev_pairs
        )
        pd.to_pickle(dev_examples, os.path.join(cache_path, "dev.pkl"))

        test_pairs = get_pairs_from_dataset(data_dir, "test1", test=True)
        test_examples = convert_data_to_examples(
            source_tokenizer, target_tokenizer, test_pairs, test=True
        )
        pd.to_pickle(test_examples, os.path.join(cache_path, "test.pkl"))

        test2_pairs = get_pairs_from_dataset(data_dir, "test2", test=True)
        test2_examples = convert_data_to_examples(
            source_tokenizer, target_tokenizer, test2_pairs, test=True
        )
        pd.to_pickle(test2_examples, os.path.join(cache_path, "test2.pkl"))

    else:
        train_examples = pd.read_pickle(os.path.join(cache_path, "train.pkl"))
        dev_examples = pd.read_pickle(os.path.join(cache_path, "dev.pkl"))
        test_examples = pd.read_pickle(os.path.join(cache_path, "test.pkl"))
        test2_examples = pd.read_pickle(os.path.join(cache_path, "test2.pkl"))

    return train_examples, dev_examples, test_examples, test2_examples


def get_pairs_from_dataset(data_dir, data_type, test=False):
    df = pd.read_csv(os.path.join(data_dir, DATAMAP[data_type]))
    sid = df["sid"].to_list()
    src_language = df["en"].to_list()
    if test:
        return {"sid": sid, "src": src_language}
    else:
        trg_language = df["ko"].to_list()
        return {"sid": sid, "src": src_language, "trg": trg_language}


def convert_data_to_examples(source_tokenizer, target_tokenizer, dataset, test=False):
    examples = []

    if test:
        for idx, (sid, src) in tqdm(enumerate(zip(dataset["sid"], dataset["src"]))):
            src_ids = source_tokenizer.encode(src.strip())
            examples.append(NMTExample(guid=f"{sid}", input_ids=src_ids, trg_ids=None))
    else:
        for idx, (sid, src, trg) in tqdm(
            enumerate(zip(dataset["sid"], dataset["src"], dataset["trg"]))
        ):
            src_ids = source_tokenizer.encode(src.strip())
            trg_ids = target_tokenizer.encode(trg.strip())
            examples.append(
                NMTExample(guid=f"{sid}", input_ids=src_ids, trg_ids=trg_ids)
            )

    return examples


class NMTDataset(Dataset):
    def __init__(
        self,
        examples: List[NMTExample],
        max_len=max_length,
        pad_idx=pad_index,
        is_test: bool = False,
    ):
        """Reads source and target sequences from txt files."""
        self.max_len = max_len
        self.pad_idx = pad_index
        self.is_test = is_test
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        data = {}
        data["sid"] = self.examples[index].guid
        data["src_ids"] = torch.tensor(self.examples[index].input_ids)
        data["tgt_ids"] = torch.tensor(self.examples[index].trg_ids)
        data["tgt_label_ids"] = torch.tensor(self.examples[index].trg_ids[1:] + [2])
        return data


class NMTDatasetforEval(Dataset):
    def __init__(
        self,
        examples: List[NMTExample],
        max_len=max_length,
        pad_idx=pad_index,
        is_test: bool = True,
    ):
        """Reads source and target sequences from txt files."""
        self.max_len = max_len
        self.pad_idx = pad_index
        self.is_test = is_test
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        data = {}
        data["sid"] = self.examples[index].guid
        data["src_ids"] = torch.tensor(self.examples[index].input_ids)
        return data


def collate_fn(samples: List[NMTExample]):
    encoder_inputs = [sample["src_ids"] for sample in samples]
    decoder_inputs = [sample["tgt_ids"] for sample in samples]
    decoder_outputs = [sample["tgt_label_ids"] for sample in samples]
    padded_encoder_inputs = pad_sequence(
        encoder_inputs, batch_first=True, padding_value=pad_index
    )
    padded_decoder_inputs = pad_sequence(
        decoder_inputs, batch_first=True, padding_value=pad_index
    )
    padded_decoder_targets = pad_sequence(
        decoder_outputs, batch_first=True, padding_value=1
    )

    encoder_attention_mask = (padded_encoder_inputs != pad_index).to(torch.long)
    decoder_attention_mask = (padded_decoder_inputs != pad_index).to(torch.long)
    return {
        "src_ids": padded_encoder_inputs,
        "tgt_ids": padded_decoder_inputs,
        "tgt_label_ids": padded_decoder_targets,
        "src_mask": encoder_attention_mask,
        "tgt_mask": decoder_attention_mask,
    }

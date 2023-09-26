import os

try:
    from transformers import MBartForConditionalGeneration, MBart50Tokenizer
except:
    os.system("pip install transformers")
    os.system("pip install sentencepiece")
    from transformers import MBartForConditionalGeneration, MBart50Tokenizer

import json
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Optional, List
import dataclasses
from dataclasses import dataclass
from tqdm import tqdm
import numpy as np
import random


def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


@dataclass
class NMTExample(object):
    guid: str
    input_ids: List[int]
    ko_text: Optional[List[int]] = None

    def to_dict(self):
        return dataclasses.asdict(self)

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2) + "\n"


def get_pairs_from_dataset(data_dir, data_name, test=False):
    df = pd.read_csv(os.path.join(data_dir, data_name + ".csv"), encoding="utf-8")
    sid = df["sid"].to_list()
    src_language = df["ko"].to_list()
    return {"sid": sid, "src": src_language}


def convert_data_to_examples(tokenizer, dataset):
    examples = []

    for idx, (sid, src) in tqdm(enumerate(zip(dataset["sid"], dataset["src"]))):
        src_ids = tokenizer.encode(src.strip())
        examples.append(NMTExample(guid=f"{sid}", input_ids=src_ids, ko_text=f"{src}"))

    return examples


def load_dataset(data_dir, data_name, tokenizer):
    additional_pairs = get_pairs_from_dataset(data_dir, data_name)
    additional_examples = convert_data_to_examples(tokenizer, additional_pairs)

    return additional_examples


class NMTDatasetforAdditional(Dataset):
    def __init__(self, examples, max_len=1024, pad_idx=1):
        self.max_len = max_len
        self.pad_idx = pad_idx
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        data = {}
        data["sid"] = self.examples[index].guid
        data["src_ids"] = torch.tensor(self.examples[index].input_ids)
        data["ko_text"] = self.examples[index].ko_text
        return data


def additional_en(model, dataset, tokenizer, fname, eval_dir, device):
    tokenizer.bos_token = tokenizer.cls_token
    decoded_list = []
    sid_list = []
    ko_list = []
    progress_bar = tqdm(dataset, desc=fname)
    for input in progress_bar:
        encoder_input_ids = input["src_ids"]
        sid = input["sid"]
        ko_text = input["ko_text"]

        with torch.no_grad():
            generated = model.generate(
                encoder_input_ids.unsqueeze(0).to(device),
                do_sample=False,
                num_beams=1,
                num_return_sequences=1,
                decoder_start_token_id=tokenizer.bos_token_id,
            )[0]

        decoded = tokenizer.decode(generated, skip_special_tokens=True)
        decoded_list.append(decoded)
        sid_list.append(sid)
        ko_list.append(ko_text)
        progress_bar.set_postfix()

    results = pd.DataFrame({"sid": sid_list, "en": decoded_list, "ko": ko_list})

    results.to_csv(os.path.join(eval_dir, fname), header=True, index=False)


def main():
    set_seed(42)

    model = MBartForConditionalGeneration.from_pretrained("../ko_en_model")
    tokenizer = MBart50Tokenizer.from_pretrained(
        "facebook/mbart-large-50-many-to-one-mmt"
    )
    tokenizer.src_lang = "ko_KR"
    device = "cuda"

    model.to(device)

    # website_data = load_dataset("data/", "website", tokenizer)
    # bylaw_data = load_dataset("data/", "bylaw", tokenizer)
    # culture_data = load_dataset("data/", "culture", tokenizer)
    news1_data = load_dataset("data/", "news(1)", tokenizer)
    # news2_data = load_dataset("data/", "news(2)", tokenizer)
    # news3_data = load_dataset("data/", "news(3)", tokenizer)
    # news4_data = load_dataset("data/", "news(4)", tokenizer)
    # talk1_data = load_dataset("data/", "talk(1)", tokenizer)
    # talk2_data = load_dataset("data/", "talk(2)", tokenizer)
    # talking_data = load_dataset("data/", "talking", tokenizer)

    news1_data = random.sample(news1_data, int(len(news1_data) * 0.35))

    eval_dir = "translation"
    os.makedirs(eval_dir, exist_ok=True)

    # website_dataset = NMTDatasetforAdditional(website_data)
    # bylaw_dataset = NMTDatasetforAdditional(bylaw_data)
    # culture_dataset = NMTDatasetforAdditional(culture_data)
    news1_dataset = NMTDatasetforAdditional(news1_data)
    # news2_dataset = NMTDatasetforAdditional(news2_data)
    # news3_dataset = NMTDatasetforAdditional(news3_data)
    # news4_dataset = NMTDatasetforAdditional(news4_data)
    # talk1_dataset = NMTDatasetforAdditional(talk1_data)
    # talk2_dataset = NMTDatasetforAdditional(talk2_data)
    # talking_dataset = NMTDatasetforAdditional(talking_data)

    # additional_en(
    #     model, website_dataset, tokenizer, "website_en_ko.csv", eval_dir, device
    # )
    # additional_en(model, bylaw_dataset, tokenizer, "bylaw_en_ko.csv", eval_dir, device)
    # additional_en(
    #     model, culture_dataset, tokenizer, "culture_en_ko.csv", eval_dir, device
    # )
    additional_en(model, news1_dataset, tokenizer, "news1_en_ko.csv", eval_dir, device)
    # additional_en(model, news2_dataset, tokenizer, "news2_en_ko.csv", eval_dir, device)
    # additional_en(model, news3_dataset, tokenizer, "news3_en_ko.csv", eval_dir, device)
    # additional_en(model, news4_dataset, tokenizer, "news4_en_ko.csv", eval_dir, device)
    # additional_en(
    #     model, talk1_dataset, tokenizer, "talk1_en_ko.csv", eval_dir, device
    # )
    # additional_en(
    #     model, talk2_dataset, tokenizer, "talk2_en_ko.csv", eval_dir, device
    # )
    # additional_en(
    #     model, talking_dataset, tokenizer, "talking_en_ko.csv", eval_dir, device
    # )

    print("done")


if __name__ == "__main__":
    main()

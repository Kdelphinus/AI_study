MODEL = "facebook/mbart-large-50-many-to-one-mmt"

import os

try:
    from transformers import (
        AdamW,
        get_linear_schedule_with_warmup,
        MBartForConditionalGeneration,
        MBart50Tokenizer,
    )
except:
    os.system("pip install transformers")
    os.system("pip install sentencepiece")

    from transformers import (
        AdamW,
        get_linear_schedule_with_warmup,
        MBartForConditionalGeneration,
        MBart50Tokenizer,
    )

from statistics import mean
from torch import nn
from torch.utils.data import DataLoader
from mbart_util.config import Config, set_seed
from mbart_util.preprocessing import *


def get_model_and_tokenizer():
    tokenizer = MBart50Tokenizer.from_pretrained(MODEL)
    model = MBartForConditionalGeneration.from_pretrained(MODEL)

    return model, tokenizer


def train(
    model,
    config,
    train_dataloader,
    dev_dataloader,
    optimizer,
    device,
    scheduler,
):
    step = 0
    min_dev_loss = 99
    accumulation = config.gradient_accumulation
    for epoch in range(config.epoch):
        progress_bar = tqdm(train_dataloader, desc=f"Train {epoch}")
        running_loss = 0.0
        train_losses = []
        model.train()

        for batch in progress_bar:
            batch = {k: v.to(device) for k, v in batch.items()}
            output = model(
                input_ids=batch["src_ids"],
                attention_mask=batch["src_mask"],
                decoder_input_ids=batch["tgt_ids"],
                decoder_attention_mask=batch["tgt_mask"],
                labels=batch["tgt_label_ids"],
            )
            (output.loss / accumulation).backward()
            running_loss += output.loss.item()

            del batch
            step += 1
            if step % accumulation:
                continue

            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            train_losses.append(running_loss / accumulation)
            running_loss = 0.0

            progress_bar.set_postfix(loss=output.loss.item())

        train_loss = mean(train_losses)
        print(f"train loss: {train_loss:.3f}")

        model.eval()
        loss_list = []
        for batch in tqdm(dev_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                output = model(
                    input_ids=batch["src_ids"],
                    attention_mask=batch["src_mask"],
                    decoder_input_ids=batch["tgt_ids"],
                    decoder_attention_mask=batch["tgt_mask"],
                    labels=batch["tgt_label_ids"],
                )
            loss_list.append(output.loss.item())
        print("Dev loss after {} epoch: {}".format(epoch + 1, mean(loss_list)))
        if min_dev_loss < mean(loss_list):
            print("overfitting")
            return
        min_dev_loss = mean(loss_list)


def main():
    config = Config()
    device = config.device

    set_seed(config.seed)

    # 모델을 정의합니다.
    model, tokenizer = get_model_and_tokenizer()
    model.to(device)

    train_data, dev_data = load_dataset("data/", tokenizer, tokenizer)

    # 데이터셋을 준비합니다.
    train_dataset, dev_dataset = NMTDataset(train_data), NMTDataset(dev_data)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
    )
    dev_dataloader = DataLoader(
        dev_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
    )

    save_dir = "ko_en_NMT"
    os.makedirs(save_dir, exist_ok=True)

    optimizer = AdamW(
        model.parameters(),
        config.learning_rate,
        weight_decay=config.optimizer_weight_decay,
    )
    total_steps = len(train_dataloader) * config.epoch
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps,
    )

    train(
        model,
        config,
        train_dataloader,
        dev_dataloader,
        optimizer,
        device,
        scheduler,
    )

    model.save_pretrained(save_dir + "/ko_en_model")

    print("done")


if __name__ == "__main__":
    main()

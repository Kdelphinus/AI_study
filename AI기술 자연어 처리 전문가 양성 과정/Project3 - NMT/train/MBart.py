import torch.cuda

MODEL = "facebook/mbart-large-50"

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

try:
    import wandb
except:
    os.system("pip install wandb")
    import wandb

import gc
from statistics import mean
from torch import nn
from torch.utils.data import DataLoader

# from mbart_util.tokenization import *
from mbart_util.config import Config, set_seed
from mbart_util.preprocessing import *
from mbart_util.bleu_score import *


def get_model_and_tokenizer():
    tokenizer = MBart50Tokenizer.from_pretrained(
        MODEL, src_lang="en_XX", tgt_lang="ko_KR"
    )
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
    save_path,
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
            gc.collect()
            torch.cuda.empty_cache()
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
            del batch
            gc.collect()
            torch.cuda.empty_cache()
        print("Dev loss after {} epoch: {}".format(epoch + 1, mean(loss_list)))
        if min_dev_loss < mean(loss_list):
            print("overfitting")
            return
        min_dev_loss = mean(loss_list)

        wandb.log({"dev Loss": min_dev_loss, "train loss": train_loss})
        model.save_pretrained(save_path + "/model.epoch{}".format(epoch))


def evaluate(model, dataset, tokenizer, device):
    tokenizer.bos_token = tokenizer.cls_token
    predictions, answers = [], []
    with tqdm(dataset, desc="Predict") as progress_bar:
        for item in progress_bar:
            encoder_input_ids = item["src_ids"]
            answer_ids = item["tgt_label_ids"]
            sid = item["sid"]
            with torch.no_grad():
                generated = model.generate(
                    input_ids=encoder_input_ids.unsqueeze(0).to(device),
                    do_sample=False,
                    num_beams=5,
                    num_return_sequences=1,
                    decoder_start_token_id=tokenizer.bos_token_id,
                    forced_bos_token_id=tokenizer.lang_code_to_id["ko_KR"],
                )[0]
                decoded = tokenizer.decode(generated, skip_special_tokens=True)
            answer = tokenizer.decode(answer_ids, skip_special_tokens=True)
            predictions.append(decoded)
            answers.append(answer)
            progress_bar.set_postfix()
    score = corpuswise_bleu(predictions, answers)
    return score


def prepare_submission(model, dataset, tokenizer, fname, eval_dir, device):
    tokenizer.bos_token = tokenizer.cls_token
    decoded_list = []
    sid_list = []
    progress_bar = tqdm(dataset, desc=fname)
    for input in progress_bar:
        encoder_input_ids = input["src_ids"]
        sid = input["sid"]
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
        progress_bar.set_postfix()

    results = pd.DataFrame({"sid": sid_list, "predicts": decoded_list})

    results.to_csv(os.path.join(eval_dir, fname), header=True, index=False)


def main():
    config = Config()
    device = config.device

    set_seed(config.seed)

    # 모델을 정의합니다.
    model, tokenizer = get_model_and_tokenizer()
    model.to(device)

    train_data, dev_data, test1_data, test2_data = load_dataset(
        "data/", tokenizer, tokenizer
    )

    # 데이터셋을 준비합니다.
    train_dataset, dev_dataset = NMTDataset(train_data), NMTDataset(dev_data)

    print(
        "batch_size:", config.batch_size, "accumulation:", config.gradient_accumulation
    )

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

    os.makedirs(config.model_save_dir, exist_ok=True)

    optimizer = AdamW(
        model.parameters(),
        config.learning_rate,
        weight_decay=config.optimizer_weight_decay,
    )
    total_steps = len(train_dataloader) * config.epoch
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps,
    )

    os.system("wandb login")
    wandb.init(project="Project 3-NMT")
    wandb.run.name = f"epoch:1~"
    wandb.run.save()
    wandb.watch(model)

    train(
        model,
        config,
        train_dataloader,
        dev_dataloader,
        optimizer,
        device,
        scheduler,
        config.model_save_dir,
    )

    model.save_pretrained(config.save_path + "/model_last")

    # 제출용 csv 생성
    eval_dir = "eval"
    os.makedirs(eval_dir, exist_ok=True)

    test1_dataset, test2_dataset = (
        NMTDatasetforEval(test1_data),
        NMTDatasetforEval(test2_data),
    )

    prepare_submission(
        model, test1_dataset, tokenizer, "test1_submission.csv", eval_dir, device
    )
    prepare_submission(
        model, test2_dataset, tokenizer, "test2_submission.csv", eval_dir, device
    )

    # BLEU score
    result = evaluate(model, dev_dataset, tokenizer, device)
    print(result)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
# @Time    : 2023/2/11 10:24
# @Author  : zxf
import os
import random

import torch
import numpy as np
from rouge import Rouge
from transformers import AutoTokenizer
from transformers import BertTokenizer
from torch.nn.utils.rnn import pad_sequence
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from config.getConfig import get_config
from config.global_conf import PROJECT_DIR

# get config params
Config = get_config(os.path.join(PROJECT_DIR, "config/config.ini"))

if 'bert' in Config["pretrain_model_type"]:
    tokenizer = BertTokenizer.from_pretrained(Config["pretrain_model_path"])
else:
    tokenizer = AutoTokenizer.from_pretrained(Config["pretrain_model_path"])
pad_token_id = tokenizer.pad_token_id


def set_global_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def collate_fn(batch_data):
    input_ids = [torch.tensor(item["input_ids"][:-1], dtype=torch.long) for item in batch_data]
    label = [torch.tensor(item["label"], dtype=torch.long) for item in batch_data]
    attention_mask = [torch.tensor(item["attention_mask"], dtype=torch.long) for item in batch_data]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    label = pad_sequence(label, batch_first=True, padding_value=-100)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    return {"input_ids": input_ids, "label": label, "attention_mask": attention_mask}


def model_rouge_bleu_evaluate(pred_result, true_result):
    metrics = Rouge()
    smooth = SmoothingFunction().method1
    rouge1, rouge2, rougel, bleu = 0, 0, 0, 0
    pred_nums = 0
    for i in range(len(pred_result)):
        try:
            # pred = ' '.join(pred_result[i])
            # ans = ' '.join(true_result[i])
            pred = pred_result[i]
            ans = true_result[i]
            rouge_result = metrics.get_scores(hyps=pred, refs=ans)
            rouge1 += rouge_result[0]['rouge-1']['f']
            rouge2 += rouge_result[0]['rouge-2']['f']
            rougel += rouge_result[0]['rouge-l']['f']
            bleu += sentence_bleu(
                references=[ans.split(' ')], hypothesis=pred.split(' '), smoothing_function=smooth)
            pred_nums += 1
        except Exception as e:
            rouge1 += 0
            rouge2 += 0
            rougel += 0
            bleu += 0
            pred_nums += 1

    if pred_nums > 0:
        rouge1_score = rouge1 / pred_nums
        rouge2_score = rouge2 / pred_nums
        rougel_score = rougel / pred_nums
        bleu_score = bleu / pred_nums
    else:
        rouge1_score, rouge2_score, rougel_score, bleu_score = 0, 0, 0, 0
    # logger.info("模型生成评价结果: epoch:{}, rouge1_score:{}, rouge2_score:{}, rougel_score:{}, bleu_score:{}".format(
    #     epoch, rouge1_score, rouge2_score, rougel_score, bleu_score
    # ))
    gene_metric = {"rouge1": rouge1_score, "rouge2": rouge2_score, "rougel": rougel_score, "bleu": bleu_score}
    return gene_metric


def model_evaluate(model, dev_dataloader, tokenizer, device):
    """模型评价"""
    model.eval()
    pred_result = []
    true_result = []
    dev_loss = 0.0
    model_eval = model.module if hasattr(model, 'module') else model
    with torch.no_grad():
        for step, batch_data in enumerate(dev_dataloader):
            input_ids = batch_data["input_ids"].to(device)
            label = batch_data["label"].to(device)
            attention_mask = batch_data["attention_mask"].to(device)
            logits, loss, gene_acc_num, gene_total_num = model_eval(input_ids=input_ids, label=label)

            dev_loss += loss.item()

            pred = torch.argmax(torch.softmax(logits, dim=-1), dim=-1)
            pred = pred * attention_mask
            pred = pred.data.cpu().numpy().tolist()
            pred_tokens = [tokenizer.decode(item, skip_special_tokens=True,
                                            clean_up_tokenization_spaces=True) for item in pred]

            label = label.data.cpu().numpy().tolist()
            true_tokens = [tokenizer.decode(item, skip_special_tokens=True,
                                            clean_up_tokenization_spaces=True) for item in label]

            pred_result.extend(pred_tokens)
            true_result.extend(true_tokens)

    gene_metric = model_rouge_bleu_evaluate(pred_result, true_result)
    dev_loss = float(dev_loss / len(dev_dataloader))
    return dev_loss, gene_metric

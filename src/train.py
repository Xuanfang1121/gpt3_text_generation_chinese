# -*- coding: utf-8 -*-
# @Time    : 2023/2/11 14:25
# @Author  : zxf
import os
import json
import math

import torch
import numpy as np
from torch.optim import AdamW
import torch.distributed as dist
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel.distributed import DistributedDataParallel

from util.util import Config
from util.util import tokenizer
from util.util import collate_fn
from util.logger import get_logger
from util.util import model_evaluate
from util.util import set_global_random_seed
from util.data_processor import DataProcessor
from model.GPT3TextGenarateModel import GPT3TextGenerateModel


def main():
    os.environ["CUDA_VISIBLE_DEVICE"] = Config["gpu_ids"]
    rank_num = len(Config["gpu_ids"].split(','))
    if rank_num > 1:
        dist.init_process_group(backend='nccl')
        local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = "cpu" if Config["gpu_ids"] == '-1' else "cuda"
        local_rank = -1
    logger = get_logger(__name__, log_file=Config["log_file"], log_level='DEBUG')
    # set randomseed
    set_global_random_seed(Config["seed"])
    # check path
    if not os.path.exists(Config["output_path"]):
        if rank_num > 1:
            if local_rank in [-1, 0]:
                os.mkdir(Config["output_path"])
        else:
            os.mkdir(Config["output_path"])

    train_dataset = DataProcessor(Config["train_data_file"], Config["max_length"], tokenizer)
    dev_dataset = DataProcessor(Config["test_data_file"], Config["max_length"], tokenizer)
    logger.info("train data size:{}".format(len(train_dataset)))
    logger.info("dev data size:{}".format(len(train_dataset)))

    # train dataloader
    if rank_num > 1:
        train_sampler = DistributedSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, batch_size=Config["batch_size"],
                                      collate_fn=collate_fn, sampler=train_sampler)
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=Config["batch_size"],
                                      collate_fn=collate_fn, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=Config["dev_batch_size"],
                                collate_fn=collate_fn, shuffle=True)
    logger.info("pre epoch having {} training step".format(len(train_dataloader)))
    logger.info("pre epoch having {} dev step".format(len(dev_dataloader)))

    model = GPT3TextGenerateModel(Config["pretrain_model_path"])
    if Config["model_name"] is not None:
        model.load_state_dict(torch.load(os.path.join(Config["output_path"], Config["model_name"]),
                                         map_location=lambda storage, loc: storage), strict=True)
        logger.info("加载模型继续训练")
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('训练的参数量: %.2fM' % float(total_params / 1000000))
    if rank_num > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True
        )
    no_decay = ['bias', 'norm']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': Config["weight_decay"]
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=Config["learning_rate"],
                      betas=(Config["beta1"], Config["beta2"]))

    total_step = len(train_dataloader) * Config["epochs"]
    warmup_steps = math.ceil(total_step * Config["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_step
    )

    best_ppl = np.inf
    best_rougel = 0.0
    best_bleu = 0.0
    best_gene_acc = 0.0
    global_step = 0
    last_imporve = 0

    model.train()
    for epoch in range(Config['epochs']):
        for step, batch_data in enumerate(train_dataloader):
            input_ids = batch_data["input_ids"].to(device)
            label = batch_data["label"].to(device)
            # attention = batch_data["attention_mask"].to(device)
            _, loss, gene_acc_num, gene_total_num = model(input_ids, label=label)

            global_step += 1
            loss.backward()

            if global_step % Config["gradient_accumulation_steps"] == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            print("epoch:{}/{}, step:{}/{}, loss:{}, batch gene acc:{}".format(
                epoch + 1, Config["epochs"], step + 1, len(train_dataloader), loss, float(gene_acc_num / gene_total_num)
            ))

        if (epoch + 1) >= Config["eval_min_start_epoch"]:
            if rank_num > 1:
                if local_rank not in [-1, 0]:
                    dist.barrier()

            if local_rank in [-1, 0]:
                dev_loss, gene_metric = model_evaluate(model, dev_dataloader, tokenizer, device)
                # logger.info("epoch:{}, model evaluate: {}".format(epoch + 1, gene_metric))
                logger.info("epoch:{}, model evaluate, rouge1:{}, rouge2:{}, rougel:{}, bleu:{}".format(
                   epoch + 1, gene_metric['rouge1'], gene_metric['rouge2'], gene_metric['rougel'], gene_metric['bleu']
                ))
                if gene_metric['rougel'] >= best_rougel:
                    best_rougel = gene_metric['rougel']
                    last_imporve = epoch + 1
                    model2save = model.module if hasattr(model, 'module') else model
                    if rank_num > 1:
                        if local_rank in [-1, 0]:
                            torch.save(model2save.state_dict(), os.path.join(Config["output_path"], "model_rougel.pt"))
                    else:
                        torch.save(model2save.state_dict(), os.path.join(Config["output_path"], "model_rougel.pt"))

                if gene_metric['bleu'] >= best_bleu:
                    best_bleu = gene_metric['bleu']

            if rank_num > 1:
                if local_rank == 0:
                    dist.barrier()

            if local_rank in [-1, 0] and Config["require_improvement"] > 0 and (epoch + 1 - last_imporve) >= Config[
                "require_improvement"]:
                logger.info("模型训练没有提升中断训练")
                break
            model.train()

        model2save = model.module if hasattr(model, 'module') else model
        if rank_num > 1:
            if local_rank in [-1, 0]:
                torch.save(model2save.state_dict(), os.path.join(Config["output_path"], "model_{}.pt".format(str(epoch))))
        else:
            torch.save(model2save.state_dict(), os.path.join(Config["output_path"], "model_{}.pt".format(str(epoch))))

    logger.info('best rougel: {}, best bleu:{}'.format(best_rougel, best_bleu))


if __name__ == "__main__":
    main()

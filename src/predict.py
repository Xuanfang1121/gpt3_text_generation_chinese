# -*- coding: utf-8 -*-
# @Time    : 2023/2/13 21:49
# @Author  : zxf
import os
import json

import torch
from transformers import BertTokenizer

from model.GPT3TextGenarateModel import GPT3TextGenerateModel


def predict(pretrain_model_path, model_file, max_length, device):
    tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)

    model = GPT3TextGenerateModel(pretrain_model_path)
    model.load_state_dict(torch.load(model_file, map_location=lambda storage, loc: storage),
                          strict=True)
    model.to(device)
    model.eval()
    while True:
        with torch.no_grad():
            text = input("输入文本：")
            feature = tokenizer(text, max_length=max_length, padding=True, truncation=True)
            input_ids = torch.tensor([feature["input_ids"]], dtype=torch.long).to(device)
            output = model.generate(input_ids, max_length, top_k=3)
            pred = tokenizer.decode(output[0], skip_special_tokens=True).replace(' ', '')
            print(json.dumps({"gene_text": pred}, ensure_ascii=False))


if __name__ == "__main__":
    gpu_ids = "-1"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    device = "cpu" if gpu_ids == "-1" else "cuda"

    pretrain_model_path = "D:/Spyder/pretrain_model/modelscope/nlp_gpt3_text-generation_chinese-base/"
    model_file = "./output/model_rougel.pt"
    max_length = 64

    predict(pretrain_model_path, model_file, max_length, device)
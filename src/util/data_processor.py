# -*- coding: utf-8 -*-
# @Time    : 2023/2/11 10:24
# @Author  : zxf
import os
import json


class DataProcessor(object):
    def __init__(self, data_file, max_length, tokenizer):
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.cls_token = self.tokenizer.cls_token
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token = self.tokenizer.sep_token
        self.sep_token_id = self.tokenizer.sep_token_id
        self.data = self.read_data(data_file)

    def read_data(self, data_file):
        data = []
        with open(data_file, "r", encoding="utf-8") as f:
            for line in f.readlines():
                line = line.strip()
                data.append(line)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data[index]
        features = self.tokenizer(text, max_length=self.max_length, truncation=True,
                                  padding=True)
        input_ids = features["input_ids"]
        label = input_ids[1:]
        attention_mask = [1] * len(label)
        return {"input_ids": input_ids, "label": label, "attention_mask": attention_mask}
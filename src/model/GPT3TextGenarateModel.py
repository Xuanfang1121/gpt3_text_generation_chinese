# -*- coding: utf-8 -*-
# @Time    : 2023/2/12 17:40
# @Author  : zxf
import torch
import torch.nn as nn

from model.modeling_gpt3 import GPT3Model


class GPT3TextGenerateModel(nn.Module):
    def __init__(self, pretrain_model_path):
        super(GPT3TextGenerateModel, self).__init__()

        self.base_model = GPT3Model.from_pretrained(pretrain_model_path)

    def forward(self, input_ids, label=None):
        outputs = self.base_model(input_ids, labels=label)
        output = (outputs['logits'], )
        if label is not None:
            output = output + (outputs['loss'], )
            gene_logits = outputs['logits'][:, 1:, :]  # 去掉首位开始解码符
            labels = label[:, :-1]
            pred = torch.argmax(torch.softmax(gene_logits, dim=-1), dim=-1)
            batch_result = torch.eq(torch.masked_select(pred, labels.gt(-100)),
                                    torch.masked_select(labels, labels.gt(-100)))
            gene_acc_num = torch.sum(batch_result)
            gene_total_num = len(batch_result.data.cpu().numpy())
            output = output + (gene_acc_num, gene_total_num)
        return output

    def generate(self, input_ids, max_length, top_k=10):
        input_ids = input_ids[:, :-1]
        output = self.base_model.generate(input_ids, max_length=max_length, do_sample=True,
                                 top_k=top_k, top_p=None)
        return output
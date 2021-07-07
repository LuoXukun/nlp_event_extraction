"""
Author: Xinrui Ma
Date: 2021-06-26 16:00:50
LastEditTime: 2021-06-29 19:32:47
Description: processing original data into model dataset
"""


import config
import utils

import pandas as pd
import torch

from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained(config.BERT_PATH)


class ArgumentDataset:
    """ 将 MRC 格式的数据处理为模型输入格式
    """
    def __init__(self, role, context, has_answer, answer, context_id):
        self.role = role
        self.context = context
        self.has_answer = has_answer
        self.answer = answer
        self.context_id = context_id
        self.max_len = config.MAX_LEN
        self.tokenizer = tokenizer
       
    def __len__(self):
        return len(self.role)
    
    def __getitem__(self, item):
        role = self.role[item]
        context = self.context[item][:self.max_len-20]  # truncate long sentence
        has_answer = self.has_answer[item]
        answer = self.answer[item]
        context_id = self.context_id[item]

        # question
        q_tokens = tokenizer.tokenize(role)
        q_token_ids = tokenizer.convert_tokens_to_ids(q_tokens)
        # context
        c_tokens = tokenizer.tokenize(context)
        c_token_ids = tokenizer.convert_tokens_to_ids(c_tokens)

        # [CLS] question [SEP] context [SEP]
        input_ids = [101] + q_token_ids + [102] + c_token_ids + [102]
        mask = [1] * len(input_ids)
        token_type_ids = [0] * (len(q_token_ids) + 2) + [1] * (len(c_token_ids) + 1)

        # 第一种标注方式
        # 给答案所在位置标1,其他位置标0,无答案则所有位置均为0
        # 第二种标注方式
        # 给答案开始/结束位置标1,其他位置标0,无答案则所有位置均为0
        targets = [0] * len(input_ids)
        start_positions = [0] * len(input_ids)
        end_positions = [0] * len(input_ids)
        if has_answer:
            qlen = len(q_token_ids) + 2
            for ans in eval(answer):  # if multiple span
                ans_tokens = tokenizer.tokenize(ans)
                ans_token_ids = tokenizer.convert_tokens_to_ids(ans_tokens)
                start_index = utils.search(ans_token_ids, c_token_ids)
                if start_index != -1:
                    targets[start_index + qlen: start_index + len(ans_token_ids) + qlen] = [1] * len(ans_token_ids)
                    start_positions[start_index + qlen] = 1
                    end_positions[start_index + len(ans_token_ids) - 1 + qlen] = 1

        padding_len = self.max_len - len(input_ids)
        if padding_len > 0:
            input_ids = input_ids + [0] * padding_len
            mask = mask + [0] * padding_len
            token_type_ids = token_type_ids + [0] * padding_len
            targets = targets + [0] * padding_len
            start_positions = start_positions + [0] * padding_len
            end_positions = end_positions + [0] * padding_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "targets": torch.tensor(targets, dtype=torch.long),
            "start_positions": torch.tensor(start_positions, dtype=torch.float),
            "end_positions": torch.tensor(end_positions, dtype=torch.float),
            "labels": torch.tensor(has_answer, dtype=torch.long),
            "context_id": context_id,
            "role": role
        }


if __name__ == "__main__":
    df = pd.read_csv(config.TRAIN_MRC_FILE)
    dset = ArgumentDataset(
        role = df.role.values,
        context=df.context.values,
        has_answer=df.has_answer.values,
        answer=df.answer.values,
        context_id=df.context_id.values
    )
    print(dset[1], dset[10], dset[100])
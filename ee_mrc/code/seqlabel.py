"""
Author: Xinrui Ma
Date: 2021-06-15 15:29:47
LastEditTime: 2021-06-30 15:54:35
Description: seqence labeling to extract events
FilePath: \mrc\src\seqlabel.py
"""

import utils
import config

import torch
import torch.nn as nn
import tokenizers
import transformers
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

import os
from tqdm import tqdm
import joblib
import json
import pandas as pd
import csv
import re
import string


BERT_PATH = "../input/bert-base-chinese/"
TOKENIZER = tokenizers.BertWordPieceTokenizer(
    os.path.join(BERT_PATH, "vocab.txt"),
    lowercase=True
)


class NERDataset:
    def __init__(self, text, tag, text_id, argum):
        self.text = text
        self.tag = tag
        self.text_id = text_id
        self.argum = argum
        self.tokenizer = TOKENIZER

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        text = self.text[item]
        tag = self.tag[item]
        text_id = self.text_id[item]
        argum = self.argum[item]

        tok_text_ids = []
        for i, ch in enumerate(text):
            o = self.tokenizer.encode(ch).ids[1:-1]
            if not o:
                tok_text_ids.extend([100])  # [100] is for [UNK]
            else:
                tok_text_ids.extend(o)
        
        assert len(text) == len(tok_text_ids)
        tag = eval(tag)
        assert len(tag) == len(text)

        tok_text_ids = tok_text_ids[:config.MAX_LEN - 2]
        tag = tag[:config.MAX_LEN - 2]

        tok_text_ids = [101] + tok_text_ids + [102]
        tag = [0] + tag + [0]

        mask = [1] * len(tok_text_ids)
        token_type_ids = [0] * len(tok_text_ids)

        padding_len = config.MAX_LEN - len(tok_text_ids)
        if padding_len > 0:
            tok_text_ids = tok_text_ids + ([0] * padding_len)
            mask = mask + ([0] * padding_len)
            token_type_ids = token_type_ids + ([0] * padding_len)
            tag = tag + ([0] * padding_len)

        return {
            "text_id": text_id,
            "orig_text": text,
            "tok_text_ids": torch.tensor(tok_text_ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "tag": torch.tensor(tag, dtype=torch.long),
            "orig_argum": argum
        }


class NERModel(nn.Module):
    def __init__(self, num_tag):
        super(NERModel, self).__init__()
        self.num_tag = num_tag
        self.bert = transformers.BertModel.from_pretrained(config.BERT_PATH, return_dict=False)
        self.bert_drop = nn.Dropout(0.3)
        self.out_tag = nn.Linear(768, self.num_tag)
    
    def forward(self, tok_text_ids, mask, token_type_ids):
        o1, _ = self.bert(tok_text_ids, attention_mask=mask, token_type_ids=token_type_ids)
        bo_tag = self.bert_drop(o1)
        tag = self.out_tag(bo_tag)
        return tag

def loss_fn(output, target, mask, num_labels):
    lfn = nn.CrossEntropyLoss()
    active_loss = mask.view(-1) == 1
    active_logits = output.view(-1, num_labels)
    active_labels = torch.where(
        active_loss,
        target.view(-1),
        torch.tensor(lfn.ignore_index).type_as(target)
    )
    loss = lfn(active_logits, active_labels)
    return loss


def train_fn(data_loader, model, optimizer, device, scheduler):
    model.train()
    role_map = joblib.load(config.ROLE_MAP_FILE)
    id2role = role_map["id2role"]; num_tag = len(id2role) * 2 + 1
    losses = utils.AverageMeter()  # moniter loss
    tk0 = tqdm(data_loader, total=len(data_loader), desc="Training")
    for bi, d in enumerate(tk0):
        text_id = d["text_id"]
        orig_text = d["orig_text"]
        tok_text_ids = d["tok_text_ids"].to(device, dtype=torch.long)
        mask = d["mask"].to(device, dtype=torch.long)
        token_type_ids = d["token_type_ids"].to(device, dtype=torch.long)
        tag = d["tag"].to(device, dtype=torch.long)

        optimizer.zero_grad()
        output = model(tok_text_ids=tok_text_ids, mask=mask, token_type_ids=token_type_ids)
        loss = loss_fn(output, tag, mask, num_tag)
        loss.backward()
        optimizer.step()
        scheduler.step()

        losses.update(loss.item(), tok_text_ids.size(0))  # batch size
        tk0.set_postfix(loss=losses.avg)
    return losses.avg


def eval_fn(data_loader, model, device, is_print=False):
    model.eval()
    role_map = joblib.load(config.ROLE_MAP_FILE)
    id2role = role_map["id2role"]
    sa, sb, sc = 0, 0, 0
    
    if is_print:
        fw = open(config.RESULT, "w", encoding="utf-8")
    
    with torch.no_grad():
        tk0 = tqdm(data_loader, total=len(data_loader), desc="Testing")
        for bi, d in enumerate(tk0):
            text_id = d["text_id"]
            orig_text = d["orig_text"]
            tok_text_ids = d["tok_text_ids"].to(device, dtype=torch.long)
            mask = d["mask"].to(device, dtype=torch.long)
            token_type_ids = d["token_type_ids"].to(device, dtype=torch.long)
            tag = d["tag"].to(device, dtype=torch.long)
            orig_argum = d["orig_argum"]
            
            output = model(tok_text_ids=tok_text_ids, mask=mask, token_type_ids=token_type_ids)
            
            output = torch.argmax(output, dim=-1).cpu().detach().numpy()

            bsz = tok_text_ids.size()[0]
            for px in range(bsz):
                _mask = sum(mask[px].tolist())
                _tok_text_ids = tok_text_ids[px][:_mask].cpu().numpy()
                _tag = tag[px][:_mask]
                _pred = output[px][:_mask]
                _orig_argum = eval(orig_argum[px])

                a, b, c, _pred_argum = utils.calculte_ner(_tok_text_ids, _mask, _pred, _orig_argum, id2role)
                sa, sb, sc = sa + a, sb + b, sc + c
                
                if is_print:
                    _orig_text = orig_text[px]
                    data_str = json.dumps({
                        "metrics": (a, b, c), 
                        "text": _orig_text, 
                        "predict": _pred_argum, 
                        "gold": _orig_argum},
                        ensure_ascii=False)
                    fw.write(data_str + "\n")
                
    if is_print: 
        fw.close()
    print("correct {} prediction {} golden {}".format(sa, sb, sc))
    if sb < 1:
        return 0, 0, 0
    else:
        p, r, f1 = 1.0*sa/sb, 1.0*sa/sc, 2.0*sa/(sb+sc)
        return p, r, f1


def process_data_NER_format(in_path, out_path):
    """ process data into NER format, 
        used for directly tagging each argument
    """
    role_map = joblib.load(config.ROLE_MAP_FILE)
    role2id = role_map["role2id"]; id2role = role_map["id2role"]; num_tag = len(id2role) * 2 + 1
    print("Number of role tags: ", num_tag, "\n")

    csv_columns = ["text", "tag", "text_id", "argum"]
    lines = utils.read_by_lines(in_path)
    data = []
    for line in lines:
        line = json.loads(line.strip())
        text = "".join(str(line["text"]).split()).lower()  # 大写全部转为小写
        # text = re.sub('[{}]'.format(string.punctuation),"",text) # 去除英文标点
        # text = re.sub('[{}]'.format(punctuation),"",text) # 去除中文标点
        tag = len(text) * [0] 
        text_id = line["id"]
        argu_dict = {}

        for event in line["event_list"]:
            event_type = event["event_type"]

            for argum in event["arguments"]:
                role = argum["role"]
                argu_tag = event_type + "-" + role
                argu_text = "".join(str(argum["argument"]).split()).lower()
                # argu_text = re.sub('[{}]'.format(string.punctuation),"",argu_text)
                # argu_text = re.sub('[{}]'.format(punctuation),"",argu_text)
                argu_dict[argu_text] = argu_tag
        
        for argu_text in argu_dict:
            st_index = utils.search(argu_text, text)
            if st_index != -1:
                ttidx = role2id[argu_dict[argu_text]]
                tag[st_index: st_index+len(argu_text)] = [ttidx * 2] * len(argu_text) # BIO tag, this is I
                tag[st_index] = (ttidx * 2) - 1  # this is B
        
        if argu_dict:
            data.append({"text": text, "tag": tag, "text_id": text_id, "argum": argu_dict})

    with open(out_path, "w", encoding="utf-8", newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for d in data:
            writer.writerow(d)
    
    print("From {}, to {}, NER tagged data saved.\n".format(in_path, out_path))


def run():
    print("Training for NER tagging...\n")
    df_train = pd.read_csv(config.TRAINING_FILE_NER).dropna().reset_index(drop=True)
    df_test = pd.read_csv(config.TESTING_FILE_NER).dropna().reset_index(drop=True)

    train_dataset = NERDataset(
        text=df_train.text.values, 
        tag=df_train.tag.values, 
        text_id=df_train.text_id.values,
        argum = df_train.argum.values
    )
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.TRAIN_BATCH_SIZE, shuffle=True, num_workers=8
    )
    test_dataset = NERDataset(
        text=df_test.text.values, 
        tag=df_test.tag.values, 
        text_id=df_test.text_id.values,
        argum = df_test.argum.values
    )
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config.VALID_BATCH_SIZE, shuffle=False, num_workers=4
    )

    role_map = joblib.load(config.ROLE_MAP_FILE)
    id2role = role_map["id2role"]; num_tag = len(id2role) * 2 + 1

    device = torch.device(config.DEVICE)
    model = NERModel(num_tag=num_tag)
    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],"weight_decay": 0.0}
    ]

    num_train_steps = int(len(df_train) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
    optimizer = AdamW(optimizer_parameters, lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )
    
    precision, recall, f = [], [], []; train_loss = []
    best_f1 = 0
    for epoch in range(config.EPOCHS):
        tloss = train_fn(train_data_loader, model, optimizer, device, scheduler)
        print("\nEpoch {} train loss {:.4f}".format(epoch, tloss)); train_loss.append(tloss)
        
        p, r, f1 = eval_fn(test_data_loader, model, device)
        print("test p = {:.4f} r = {:.4f} f1 = {:.4f}".format(p, r, f1))
        precision.append(p); recall.append(r); f.append(f1)
        if f1 > best_f1:
            torch.save(model.state_dict(), config.MODEL_PATH)
            best_f1 = f1
        
    print("\nTraining is finished.\nTrain loss - {}".format(train_loss))
    print("Metrics of each epoch") 
    print('precision - {}'.format(precision))
    print('recall - {}'.format(recall))
    print('f1 - {}'.format(f))


def predict():
    print("\nPredict ...\n")
    df_test = pd.read_csv(config.TESTING_FILE_NER).dropna().reset_index(drop=True)

    test_dataset = NERDataset(
            text=df_test.text.values, 
            tag=df_test.tag.values, 
            text_id=df_test.text_id.values,
            argum = df_test.argum.values
        )
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config.VALID_BATCH_SIZE, shuffle=False, num_workers=4
    )

    role_map = joblib.load(config.ROLE_MAP_FILE)
    id2role = role_map["id2role"]; num_tag = len(id2role) * 2 + 1

    device = torch.device(config.DEVICE)
    model = NERModel(num_tag=num_tag)
    model.to(device)
    model.load_state_dict(torch.load(config.MODEL_PATH))

    p, r, f1 = eval_fn(test_data_loader, model, device, is_print=True)
    print("test p={:.4f} r={:.4f} f1={:.4f}".format(p, r, f1))
    print("Output is done.\n")


def main():
    process_data_NER_format(config.TRAINING_FILE, config.TRAINING_FILE_NER)
    process_data_NER_format(config.TESTING_FILE, config.TESTING_FILE_NER)
    run()
    predict()


if __name__ == "__main__":
    main()
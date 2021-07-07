"""
Author: Xinrui Ma
Date: 2021-06-12 10:49:12
LastEditTime: 2021-06-30 17:32:46
Description: Predict event type for EE work
FilePath: \mrc\src\event_type.py
""" 


import pandas as pd
import numpy as np
import json
import os
import joblib
from tqdm import tqdm

import torch
import torch.nn as nn
from sklearn.model_selection import KFold
import tokenizers
import transformers
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

import config
import utils
from utils import load_data_event_type, read_by_lines


BERT_PATH = "../input/bert-base-chinese/"
TOKENIZER = tokenizers.BertWordPieceTokenizer(
    os.path.join(BERT_PATH, "vocab.txt"),
    lowercase=True
)


class EntityDataset:
    """ used for predicting event type given a sentence
    """
    def __init__(self, texts, tags, text_ids):
        # texts: [[雅高控股(03313)执行董事梁迦杰辞职], [balabala.....]]
        # tags: [[0,0,0,...], [....].....]]
        self.texts = texts
        self.tags = tags
        self.text_ids = text_ids
        self.tokenizer = config.TOKENIZER

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = self.texts[item][:config.MAX_LEN - 5]
        tag = self.tags[item][:config.MAX_LEN - 5]
        text_id = self.text_ids[item]

        ids = []
        for i, ch in enumerate(text):
            o = self.tokenizer.encode(ch).ids[1:-1]
            if not o:
                ids.extend([100]) # # [100] is for [UNK]
            else:
                ids.extend(o)
                       
        try:
            assert len(ids) == len(text)
        except:
            print(ids)
            print(text)

        ids = [101] + ids + [102]  # Add [CLS] and [SEP] 
        tag = [0] + tag + [0]
        lens = len(ids)
        mask = [1] * len(ids)
        token_type_ids = [0] * len(ids)

        padding_len = config.MAX_LEN - len(ids)
        ids = ids + ([0] * padding_len)
        mask = mask + ([0] * padding_len)
        token_type_ids = token_type_ids + ([0] * padding_len)
        tag = tag + ([0] * padding_len)

        return {
            "text_id": text_id,
            "lens": torch.tensor(lens, dtype=torch.long),
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "target_tag": torch.tensor(tag, dtype=torch.long),
        }


class EntityModel(nn.Module):
    """ used to predict event type
    """
    def __init__(self, num_tag):
        super(EntityModel, self).__init__()
        self.num_tag = num_tag
        self.bert = transformers.BertModel.from_pretrained(config.BERT_PATH,return_dict=False)
        self.bert_drop = nn.Dropout(0.3)
        self.out_tag = nn.Linear(768, self.num_tag)
    
    def forward(self, lens, ids, mask, token_type_ids, target_tag):
        o1, _ = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        bo_tag = self.bert_drop(o1)
        pred_tag = self.out_tag(bo_tag)

        loss = loss_fn(pred_tag, target_tag, mask, self.num_tag)

        return lens, ids, pred_tag, target_tag, loss
        

def loss_fn(output, target, mask, num_labels=2):
    lfn = nn.CrossEntropyLoss()
    active_loss = mask.view(-1) == 1  # 不计算padding的loss
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
    losses = utils.AverageMeter()  # moniter loss
    tk0 = tqdm(data_loader, total=len(data_loader), desc="Training")
    for data in tk0:
        for k, v in data.items():
            data[k] = v.to(device)
        optimizer.zero_grad()
        _, _, _, _, _, loss = model(**data)
        loss.backward()
        optimizer.step()
        scheduler.step()
       
        losses.update(loss.item())
        tk0.set_postfix(loss=losses.avg)
    return losses.avg



def eval_fn(data_loader, model, device, is_print=False, out_path=None):
    evt_map = joblib.load(config.EVENT_TYPE_FILE)
    id2evt = evt_map["id2evt"]
    model.eval()
    sa, sb, sc = 0, 0, 0
    if is_print:
        fw = open(out_path, "w", encoding="utf-8")
    with torch.no_grad():
        for data in tqdm(data_loader, total=len(data_loader), desc="Testing"):
            text_id = data["text_id"]
            lens = data["lens"].to(device, dtype=torch.long)
            ids = data["ids"].to(device, dtype=torch.long)
            mask = data["mask"].to(device, dtype=torch.long)
            token_type_ids = data["token_type_ids"].to(device, dtype=torch.long)
            target_tag = data["target_tag"].to(device, dtype=torch.long)

            lens, ids, pred_tag, target_tag, _ = model(lens, ids, mask, token_type_ids, target_tag)
            
            bsz = ids.size()[0]
            for i in range(bsz):
                text_id_ = text_id[i]
                len_ = lens[i].tolist()
                ids_ = ids[i].cpu().numpy().reshape(-1)[:len_]
                pred_tag_ = pred_tag[i].argmax(-1).cpu().numpy().reshape(-1)[:len_]
                target_tag_ = target_tag[i].cpu().numpy().reshape(-1)[:len_]
                
                pred, targ = set(), set()
                for tt in pred_tag_:
                    if tt:
                        pred.add(tt)
                for tt in target_tag_:
                    if tt:
                        targ.add(tt)

                a, b, c = len(pred & targ), len(pred), len(targ)
                sa, sb, sc = sa + a, sb + b, sc + c
                
                if is_print:
                    data_str = utils.print_event_type(text_id_, ids_, pred, targ, id2evt)
                    fw.write(data_str + "\n")
    if is_print:               
        fw.close()                
    print("correct {} prediction {} golden {}".format(sa, sb, sc))
    if sb < 1:
        return 0, 0, 0
    else:
        p, r, f1 = 1.0*sa/sb, 1.0*sa/sc, 2.0*sa/(sb+sc)
        return p, r, f1


def run():
    """ train to predict event type of each sentence,
        using 5-fold cross validation
    """
    evt_map = joblib.load(config.EVENT_TYPE_FILE)
    evt2id = evt_map["evt2id"]
    id2evt = evt_map["id2evt"]
    num_tag = len(evt2id)
    print("Number of event types: ", num_tag, "\n")

    test_sentences, test_tag, text_id = load_data_event_type(config.TEST_FILE_EVENT_TYPE)
    test_dataset = EntityDataset(
            texts=test_sentences, tags=test_tag, text_ids=text_id
        )
    test_data_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=config.VALID_BATCH_SIZE, num_workers=2
        )

    
    # K_FOLD
    precision, recall, f1 = [], [], []
    kf = KFold(n_splits=config.K_FOLD, shuffle=True, random_state=2021)
    orig_train_text, orig_train_tag, orig_train_text_id = load_data_event_type(config.TRAIN_FILE_EVENT_TYPE)

    total_train_len = len(orig_train_tag)
    
    k = 1
    for train_index, valid_index in kf.split(np.array(list(range(total_train_len)))):
        print("K-fold {}".format(k))
        train_sentences, train_tag, train_text_id = [], [], []
        valid_sentences, valid_tag, valid_text_id = [], [], []
        
        for ti in train_index:
            train_sentences.append(orig_train_text[ti])
            train_tag.append(orig_train_tag[ti])
            train_text_id.append(orig_train_text_id[ti])
        for ti in valid_index:
            valid_sentences.append(orig_train_text[ti])
            valid_tag.append(orig_train_tag[ti])
            valid_text_id.append(orig_train_text_id[ti])
        
        train_dataset = EntityDataset(
            texts=train_sentences, tags=train_tag, text_ids=train_text_id
        )
        train_data_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=config.TRAIN_BATCH_SIZE, num_workers=2
        )
        valid_dataset = EntityDataset(
            texts=valid_sentences, tags=valid_tag, text_ids=valid_text_id
        )
        valid_data_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=config.VALID_BATCH_SIZE, num_workers=2
        )
        
        device = torch.device(config.DEVICE)
        model = EntityModel(num_tag=num_tag)
        model.to(device)

        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.001,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        num_train_steps = int(len(train_sentences) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
        optimizer = AdamW(optimizer_parameters, lr=3e-5)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
        )

        best_f1, p_, r_ = 0, 0, 0
        for epoch in range(config.EPOCHS):            
            train_loss = train_fn(train_data_loader, model, optimizer, device, scheduler)
            print("Epoch {} train loss {}".format(epoch, train_loss))
            
            p, r, f = eval_fn(valid_data_loader, model, device)
            if f > best_f1:
                torch.save(model.state_dict(), config.MODEL_PATH)
                best_f1, p_, r_ = f, p, r
            print("valid p={:.4f} r={:.4f} f1={:.4f}".format(p, r, f))

            # 打印输出测试集上的指标
            p, r, f = eval_fn(test_data_loader, model, device)
            print("test p={:.4f} r={:.4f} f1={:.4f}\n".format(p, r, f))

        precision.append(p_); recall.append(r_); f1.append(best_f1)
        k += 1

    print("Metrics of each fold") 
    print('precision - {}'.format(precision))
    print('recall - {}'.format(recall))
    print('f1 - {}'.format(f1))

    print('Avg metrcis: p={} r={} f1={}\n'.format(
        sum(precision)/config.K_FOLD,
        sum(recall)/config.K_FOLD,
        sum(f1)/config.K_FOLD))


def predict():
    """ predict event type of each sentence in test data,
        using all training data
    """
    evt_map = joblib.load(config.EVENT_TYPE_FILE)
    evt2id = evt_map["evt2id"]
    id2evt = evt_map["id2evt"]
    num_tag = len(evt2id)
    print("Number of event types: ", num_tag, "\n")

    test_sentences, test_tag, test_text_id = load_data_event_type(config.TEST_FILE_EVENT_TYPE)
    test_dataset = EntityDataset(
            texts=test_sentences, tags=test_tag, text_ids=test_text_id
        )
    test_data_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=config.VALID_BATCH_SIZE, num_workers=2
        )
    
    train_sentences, train_tag, train_text_id = load_data_event_type(config.TRAIN_FILE_EVENT_TYPE)
    train_dataset = EntityDataset(
            texts=train_sentences, tags=train_tag, text_ids=train_text_id
        )
    train_data_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=config.TRAIN_BATCH_SIZE, shuffle=True, num_workers=4
        )

    device = torch.device(config.DEVICE)
    model = EntityModel(num_tag=num_tag)
    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    num_train_steps = int(len(train_sentences) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
    optimizer = AdamW(optimizer_parameters, lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )
    
    # record each epoch
    precision, recall, f1 = [], [], []
    best_f1 = 0
    for epoch in range(config.EPOCHS):            
        train_loss = train_fn(train_data_loader, model, optimizer, device, scheduler)
        print("\nEpoch {} train loss {}".format(epoch, train_loss))

        p, r, f = eval_fn(test_data_loader, model, device)
        if f > best_f1:
            torch.save(model.state_dict(), config.MODEL_PATH)
            best_f1 = f
        print("test p={:.4f} r={:.4f} f1={:.4f}".format(p, r, f))
        precision.append(p); recall.append(r); f1.append(f)

    print("\nMetrics of each epoch") 
    print('precision - {}'.format(precision))
    print('recall - {}'.format(recall))
    print('f1 - {}'.format(f1))
    
    print("\nPredcit and output the result of event type.\n")
    model = EntityModel(num_tag=num_tag)
    model.load_state_dict(torch.load(config.MODEL_PATH))
    model.to(device)
    p, r, f = eval_fn(test_data_loader, model, device, is_print=True)
    print("test p={:.4f} r={:.4f} f1={:.4f}".format(p, r, f))
    print("Output is done.\n")


def process_data_predict_event_label(in_path, out_path):
    """process data into NER format, for predicting event type
    """
    evt_map = joblib.load(config.EVENT_TYPE_FILE)
    evt2id = evt_map["evt2id"]
    id2evt = evt_map["id2evt"]
    # print(evt2id, id2evt)

    lines = utils.read_by_lines(in_path)
    print("From dir {}, total {} lines have been read.".format(in_path, len(lines)))

    fw = open(out_path, "w", encoding="utf-8")
    for line in lines:
        data = json.loads(line.strip())
        if not len(data["text"]): continue
        tag = [0] * len(data["text"])  # 0 represents tag "O"
        
        for event in data["event_list"]:
            tri_start = event["trigger_start_index"]
            tri_end = tri_start + len(event["trigger"])
            for i in range(tri_start, tri_end):
                tag[i] = evt2id[event["event_type"]]  # just use I/O tag, not BIO
                # if i == tri_start:
                #     tags[i] = "B-{}".format(event["event_type"])
                # else:
                #     tags[i] = "I-{}".format(event["event_type"])
        assert len(data["text"]) == len(tag)
        data_str = json.dumps({"text": data["text"], "tag": tag, "text_id": data["id"]}, ensure_ascii=False)
        fw.write(data_str + "\n")
    print("To dir {}, trigger tagging is finished, for predicting event type.".format(out_path))
    fw.close()


def predict_run(in_path, out_path):
    """ predict event type of each sentence in all data,
        using trained model
    """
    print("\nPredcit and output the result of event type.\n")

    evt_map = joblib.load(config.EVENT_TYPE_FILE)
    evt2id = evt_map["evt2id"]; num_tag = len(evt2id)

    sentences, tag, text_id = load_data_event_type(in_path)
    datasets = EntityDataset(
            texts=sentences, tags=tag, text_ids=text_id
        )
    data_loader = torch.utils.data.DataLoader(
            datasets, batch_size=config.TRAIN_BATCH_SIZE, num_workers=4
        )

    device = torch.device(config.DEVICE)
    model = EntityModel(num_tag=num_tag)
    model.load_state_dict(torch.load(config.MODEL_PATH))
    model.to(device)
    p, r, f = eval_fn(data_loader, model, device, is_print=True, out_path=out_path)
    print("test p={:.4f} r={:.4f} f1={:.4f}".format(p, r, f))
    print("Output is done, file saved in {}.\n".format(out_path))
    

def process_data_stage1(in_path, match_path, out_path):
    """ process data, 
        append predicted event type to original data,
        match by text_id
    """
    lines = read_by_lines(match_path)
    match_data = {}
    for line in lines:
        line = json.loads(line.strip())
        match_data[line["text_id"]] = line["predict"]

    with open(out_path, "w", encoding="utf-8") as fw:
        lines = read_by_lines(in_path)
        for line in lines:
            line = json.loads(line.strip())
            line["predict_event_type"] = match_data[line["id"]]
            data_str = json.dumps(line, ensure_ascii=False)
            fw.write(data_str + "\n")
    print("Data has been append predicted event type, from {}, to {}".format(in_path, out_path))


def main():
    print("Train for predicting event type.\n")

    process_data_predict_event_label(config.TRAINING_FILE, config.TRAIN_FILE_EVENT_TYPE)
    process_data_predict_event_label(config.TESTING_FILE, config.TEST_FILE_EVENT_TYPE)

    run()   # 5-fold cross validation

    print("Predicting event type on test data.\n")
    predict()  # all training data

    predict_run(config.TEST_FILE_EVENT_TYPE, config.TEST_WITH_EVENT)
    process_data_stage1(config.TESTING_FILE, config.TEST_WITH_EVENT, config.TEST_STAGE1)  # 将第一阶段预测的事件类型拼接到原数据中，作为之后论元抽取的依据


if __name__ == "__main__":
    main()
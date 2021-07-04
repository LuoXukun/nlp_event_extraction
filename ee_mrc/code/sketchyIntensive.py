'''
Author: Xinrui Ma
Date: 2021-06-26 19:05:12
LastEditTime: 2021-07-01 11:29:42
Description: using mrc to extract events 
            --- pipeline stage2 
             (1) sketchy reader model
             (2) intensive reader model
            --- train / evaluate functions
'''

import config
import utils


import csv
from tqdm import tqdm

import torch
import torch.nn as nn

from transformers import BertConfig, BertModel, BertPreTrainedModel

model_config = BertConfig.from_pretrained(config.BERT_PATH)


def loss_fn(logits, labels, mask, num_labels):
    lfn = nn.CrossEntropyLoss()
    active_loss = mask.view(-1) == 1  # 不计算padding的loss
    active_logits = logits.view(-1, num_labels)
    active_labels = torch.where(
        active_loss,
        labels.view(-1),
        torch.tensor(lfn.ignore_index).type_as(labels)
    )
    loss = lfn(active_logits, active_labels)
    return loss


class SketchyReader(BertPreTrainedModel):
    """ classify --- external front verify,
        whether a question has an answer or not,
        input:
            question + context
        return:
            label: 1-has-answer, 0-not
    """

    def __init__(self):
        super(SketchyReader, self).__init__(model_config)
        self.num_labels = 2
        self.bert = BertModel.from_pretrained(config.BERT_PATH, config=model_config)
        self.drop_out = nn.Dropout(config.dropout_prob)
        self.classifier = nn.Linear(768, self.num_labels)

        nn.init.normal_(self.classifier.weight, std=0.02)

    def forward(self, input_ids, mask, labels):
        outputs = self.bert(
            input_ids,
            attention_mask=mask
        )
        out = outputs.pooler_output
        out = self.drop_out(out)
        logits = self.classifier(out)  # [bsz, 2]

        assert labels is not None

        lfn = nn.CrossEntropyLoss()
        loss = lfn(logits.view(-1, self.num_labels), labels.view(-1))

        return loss, logits


class IntensiveReader(BertPreTrainedModel):
    """ sequence tag --- predict answer,
        classify --- internal front verify,
        input: 
            question + context
        return: 
            answer span, start/end position
            label: 1-has-answer, 0-not
    """

    def __init__(self):
        super(IntensiveReader, self).__init__(model_config)
        self.num_labels = 2
        self.bert = BertModel.from_pretrained(config.BERT_PATH, config=model_config)
        self.drop_out = nn.Dropout(config.dropout_prob)

        # for question answering, predict start/end span
        self.qa_start = nn.Linear(768, self.num_labels)
        self.qa_end = nn.Linear(768, self.num_labels)

        # for classification, whether has answere or not
        self.has_ans = nn.Linear(768, self.num_labels)

        nn.init.normal_(self.qa_start.weight, std=0.02)
        nn.init.normal_(self.qa_end.weight, std=0.02)
        nn.init.normal_(self.has_ans.weight, std=0.02)

    def forward(self, input_ids, mask, token_type_ids, 
                start_positions, end_positions, labels):

        outputs = self.bert(
            input_ids,
            attention_mask=mask,
            token_type_ids=token_type_ids
        )

        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        sequence_output = self.drop_out(sequence_output)

        # 1 --- For start and end logits (span)
        start_logits = self.qa_start(sequence_output)  # [batch_size, seq_len, 2]
        end_logits = self.qa_end(sequence_output)  # [batch_size, seq_len, 2]

        # 2 --- For answerable classification
        first_word = sequence_output[:, 0, :]   # first_word [batch_size, hidden_size]
        cls_logits = self.has_ans(first_word)   # [batch_size, 2]

        # calculate loss
        start_loss = loss_fn(logits=start_logits, labels=start_positions, mask=mask, num_labels=self.num_labels)
        end_loss = loss_fn(logits=end_logits, labels=end_positions, mask=mask, num_labels=self.num_labels)

        lfn = nn.CrossEntropyLoss()
        cls_loss = lfn(cls_logits.view(-1, self.num_labels), labels.view(-1))

        loss = start_loss + end_loss + cls_loss

        return loss, start_logits, end_logits, cls_logits


class RearVerify():
    """ combine external and internal front verify,
        predict answer
        input:
            scores in SketchyReader and IntensiveReader
        return:
            if has answer: answer
            if not: null
    """
    pass


def train_sketchy_reader(data_loader, model, optimizer, device, scheduler):
    model.train()
    losses = utils.AverageMeter()  # moniter loss
    tk0 = tqdm(data_loader, total=len(data_loader), desc="Training")
    for _, d in enumerate(tk0):
        input_ids = d["input_ids"].to(device, dtype=torch.long)
        mask = d["mask"].to(device, dtype=torch.long)
        labels = d["label"].to(device, dtype=torch.long)
        optimizer.zero_grad()
        loss, _ = model(input_ids=input_ids, mask=mask, labels=labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.update(loss.item(), input_ids.size(0))  # bsz
        tk0.set_postfix(loss=losses.avg)
    return losses.avg


def evaluate_sketchy_reader(data_loader, model, device, is_test=False, fold=None):
    model.eval()
    losses = utils.AverageMeter()
    accs = utils.AverageMeter()
    results = []
    with torch.no_grad():
        tk0 = tqdm(data_loader, total=len(data_loader), desc="Validating")
        for _, d in enumerate(tk0):
            input_ids = d["input_ids"].to(device, dtype=torch.long)
            mask = d["mask"].to(device, dtype=torch.long)
            labels = d["label"].to(device, dtype=torch.long)
            context_id = d["context_id"]
            role = d["role"]

            loss, logits = model(input_ids=input_ids, mask=mask, labels=labels)  # logits [bsz, 2] 

            bsz = input_ids.size(0)
            preds = torch.argmax(logits, dim=1).squeeze()  # [bsz, 1] --> [bsz]
            acc = sum(preds == labels).float() / bsz   # 一个batch的平均准确率

            losses.update(loss.item(), bsz)
            accs.update(acc.item(), bsz)
            tk0.set_postfix(loss=losses.avg)

            if is_test:
                logits = torch.softmax(logits, dim=1).cpu().detach().numpy()
                for px, logit in enumerate(logits):
                    result = {
                        "context_id": context_id[px],
                        "role": role[px],
                        "score1": logit[1] - logit[0],
                        "predict": preds[px].item(),
                        "label": labels[px].item()
                    }
                    results.append(result)

    if is_test:
        assert fold is not None
        save_result_path = "../output/" + "fold_" + str(fold) + "_" + config.SKETCHY
        csv_columns = ["context_id", "role", "score1", "predict", "label"]
        with open(save_result_path, "w", encoding="utf-8", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_columns)
            writer.writeheader()
            for d in results:
                writer.writerow(d)
        print("Test result saved in {}".format(save_result_path))

    return losses.avg, accs.avg


def train_intensive_reader(data_loader, model, optimizer, device, scheduler):
    model.train()
    losses = utils.AverageMeter()
    tk0 = tqdm(data_loader, total=len(data_loader), desc="Training")
    for _, d in enumerate(tk0):
        input_ids = d["input_ids"].to(device, dtype=torch.long)
        mask = d["mask"].to(device, dtype=torch.long)
        token_type_ids = d["token_type_ids"].to(device, dtype=torch.long)
        start_positions = d["start_positions"].to(device, dtype=torch.long)
        end_positions = d["end_positions"].to(device, dtype=torch.long)
        labels = d["label"].to(device, dtype=torch.long)

        optimizer.zero_grad()
        loss, _, _, _ = model(
            input_ids=input_ids, mask=mask,
            token_type_ids=token_type_ids,
            start_positions=start_positions,
            end_positions=end_positions,
            labels=labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.update(loss.item(), input_ids.size(0))
        tk0.set_postfix(loss=losses.avg)
    return losses.avg


def evaluate_intensive_reader(data_loader, model, device, is_test=False, fold=None):
    model.eval()
    losses = utils.AverageMeter()
    accs = utils.AverageMeter()
    results = []
    sc, sp, sg = 0, 0, 0
    with torch.no_grad():
        tk0 = tqdm(data_loader, total=len(data_loader), desc="Validating")
        for _, d in enumerate(tk0):
            input_ids = d["input_ids"].to(device, dtype=torch.long)
            mask = d["mask"].to(device, dtype=torch.long)
            token_type_ids = d["token_type_ids"].to(device, dtype=torch.long)
            start_positions = d["start_positions"].to(device, dtype=torch.long)
            end_positions = d["end_positions"].to(device, dtype=torch.long)
            labels = d["label"].to(device, dtype=torch.long)
            context_id = d["context_id"]
            role = d["role"]

            loss, start_logits, end_logits, cls_logits = model(
                input_ids=input_ids, mask=mask,
                token_type_ids=token_type_ids,
                start_positions=start_positions,
                end_positions=end_positions,
                labels=labels)

            bsz = input_ids.size(0)
            losses.update(loss.item(), bsz)
            tk0.set_postfix(loss=losses.avg)

            # classification
            cls_preds = torch.argmax(cls_logits, dim=1).squeeze()  # [bsz, 1] --> [bsz]
            acc = sum(cls_preds == labels).float() / bsz   # 一个batch对有无答案的预测平均准确率
            accs.update(acc.item(), bsz)

            # answer span
            start_preds = torch.argmax(start_logits, dim=-1).squeeze()  # [bsz, seq_len, 2] --> [bsz, seq_len]
            end_preds = torch.argmax(end_logits, dim=-1).squeeze()

            start_logits = torch.softmax(start_logits, dim=-1).cpu().detach().numpy()  # [bsz, seq_len, 2]
            end_logits = torch.softmax(end_logits, dim=-1).cpu().detach().numpy()

            for px in range(bsz):
                mask_len = sum(mask[px]).item()  # remove padding positions
                input_postitions = [start_preds[px][:mask_len], end_preds[px][:mask_len],
                                    start_positions[px][:mask_len], end_positions[px][:mask_len]]
                num_correct, num_predict, num_gold, pred_pos, gold_pos = (
                                        utils.calculate_cpg(input=input_postitions))
                sc, sp, sg = sc + num_correct, sp + num_predict, sg + num_gold

                if is_test:

                    first_word_start = start_logits[px, 0, 1]
                    first_word_end = end_logits[px, 0, 1]
                    score_dict = {}

                    if pred_pos:
                        for (s, e) in pred_pos:
                            ans_start = start_logits[px, s, 1]
                            ans_end = end_logits[px, e, 1]
                            score = ans_start + ans_end - (first_word_start + first_word_end)
                            score_dict[(s, e)] = score

                    result = {
                        "context_id": context_id[px],
                        "role": role[px],
                        "cls_predict": cls_preds[px].item(),
                        "label": labels[px].item(),
                        "pred_position": pred_pos,
                        "gold_position": gold_pos,
                        "score2": score_dict
                    }
                    results.append(result)

    f1 = 2.0 * sc / (sp + sg)
    p = 1.0 * sc / sp if sp > 1 else 0
    r = 1.0 * sc / sg
    print("[number correct={} predict={} golden={}]".format(sc, sp, sg))

    if is_test:
        assert fold is not None
        save_result_path = "../output/" + "fold_" + str(fold) + "_" + config.INTENSIVE
        csv_columns = ["context_id", "role", "cls_predict", "label", 
                       "pred_position", "gold_position", "score2"]
        with open(save_result_path, "w", encoding="utf-8", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_columns)
            writer.writeheader()
            for d in results:
                writer.writerow(d)
        print("Test result saved in {}".format(save_result_path))

    return losses.avg, accs.avg, f1, p, r
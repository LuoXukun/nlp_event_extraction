'''
Author: Xinrui Ma
Date: 2021-06-25 00:29:56
LastEditTime: 2021-07-01 10:24:28
Description: using mrc to extract events 
            --- pipeline stage2 
             (1) sketchy reader model
             (2) intensive reader model
            --- train / evaluate functions
'''

import config
import utils
from dataset import ArgumentDataset

import csv
from tqdm import tqdm
import pandas as pd
import datetime
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertConfig, BertModel, BertPreTrainedModel
from transformers import AdamW, get_linear_schedule_with_warmup

model_config = BertConfig.from_pretrained(config.BERT_PATH)


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
        self.num_labels = 1
        self.bert = BertModel.from_pretrained(config.BERT_PATH, config=model_config)
        self.drop_out = nn.Dropout(config.dropout_prob)
        self.classifier = nn.Linear(768, self.num_labels)

        nn.init.normal_(self.classifier.weight, std=0.02)

    def forward(self, input_ids, mask, token_type_ids, labels):
        outputs = self.bert(
            input_ids,
            attention_mask=mask,
            token_type_ids=token_type_ids
        )
        out = outputs.pooler_output
        out = self.drop_out(out)
        logits = self.classifier(out).squeeze()  # [bsz]
        logits = nn.Sigmoid()(logits)
        assert labels is not None
        lfn = nn.BCELoss()
        loss = lfn(logits, labels)
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
        self.num_labels = 1
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
        start_logits = self.qa_start(sequence_output).squeeze()  # [batch_size, seq_len]
        end_logits = self.qa_end(sequence_output).squeeze()  # [batch_size, seq_len]

        # 2 --- For answerable classification
        first_word = sequence_output[:, 0, :]   # first_word [batch_size, hidden_size]
        cls_logits = self.has_ans(first_word).squeeze()   # [batch_size]

        bsz, seq_len = mask.size(0), mask.size(1)
        start_logits = nn.Sigmoid()(start_logits).view(bsz, seq_len) # [batch_size, seq_len]
        end_logits = nn.Sigmoid()(end_logits).view(bsz, seq_len)
        cls_logits = nn.Sigmoid()(cls_logits).view(bsz)  # [batch_size]

        # calculate loss
        loss_fct = nn.BCELoss()
        start_loss = loss_fct(start_logits*mask.float(), start_positions)
        end_loss = loss_fct(end_logits*mask.float(), end_positions)
        cls_loss = loss_fct(cls_logits, labels)
        loss = (start_loss + end_loss) / 2 + cls_loss

        return loss, start_logits, end_logits, cls_logits


def train_sketchy_reader(data_loader, model, optimizer, device, scheduler):
    model.train()
    losses = utils.AverageMeter()  # moniter loss
    tk0 = tqdm(data_loader, total=len(data_loader), desc="Training")
    for _, d in enumerate(tk0):
        input_ids = d["input_ids"].to(device, dtype=torch.long)
        mask = d["mask"].to(device, dtype=torch.long)
        token_type_ids = d["token_type_ids"].to(device, dtype=torch.long)
        labels = d["labels"].to(device, dtype=torch.float)
        optimizer.zero_grad()
        loss, _ = model(input_ids=input_ids, mask=mask, labels=labels, token_type_ids=token_type_ids)
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.update(loss.item(), input_ids.size(0))  # bsz
        tk0.set_postfix(loss=losses.avg)
    return losses.avg


def train_intensive_reader(data_loader, model, optimizer, device, scheduler):
    model.train()
    losses = utils.AverageMeter()
    tk0 = tqdm(data_loader, total=len(data_loader), desc="Training")
    for _, d in enumerate(tk0):
        input_ids = d["input_ids"].to(device, dtype=torch.long)
        mask = d["mask"].to(device, dtype=torch.long)
        token_type_ids = d["token_type_ids"].to(device, dtype=torch.long)
        start_positions = d["start_positions"].to(device, dtype=torch.float)
        end_positions = d["end_positions"].to(device, dtype=torch.float)
        labels = d["labels"].to(device, dtype=torch.float)

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



def evaluate_sketchy_reader(data_loader, model, device, is_test=False):
    model.eval()
    losses = utils.AverageMeter()
    accs = utils.AverageMeter()
    results = []
    with torch.no_grad():
        tk0 = tqdm(data_loader, total=len(data_loader), desc="Validating")
        for _, d in enumerate(tk0):
            input_ids = d["input_ids"].to(device, dtype=torch.long)
            mask = d["mask"].to(device, dtype=torch.long)
            labels = d["labels"].to(device, dtype=torch.float)
            context_id = d["context_id"]
            role = d["role"]

            loss, logits = model(input_ids=input_ids, mask=mask, labels=labels)  # logits [bsz] 

            bsz = input_ids.size(0)
            preds = torch.where(logits > 0.5, 1, 0)
            acc = sum(preds == labels).float() / bsz   # 一个batch的平均准确率

            losses.update(loss.item(), bsz)
            accs.update(acc.item(), bsz)
            tk0.set_postfix(loss=losses.avg)

            if is_test:
                for px in range(bsz):
                    result = {
                        "context_id": context_id[px],
                        "role": role[px],
                        "predict": preds[px].item(),
                        "label": int(labels[px].item()),
                        "score1": round(logits[px].item(), 4)
                        }
                    results.append(result)

    if is_test:
        save_result_path = config.SKETCHY
        csv_columns = ["context_id", "role", "predict", "label", "score1"]
        with open(save_result_path, "w", encoding="utf-8", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_columns)
            writer.writeheader()
            for d in results:
                writer.writerow(d)
        print("Test result saved in {}".format(save_result_path))

    return losses.avg, accs.avg



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
            start_positions = d["start_positions"].to(device, dtype=torch.float)
            end_positions = d["end_positions"].to(device, dtype=torch.float)
            labels = d["labels"].to(device, dtype=torch.float)
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
            cls_preds = torch.where(cls_logits > 0.5, 1, 0)
            acc = sum(cls_preds == labels).float() / bsz   # 一个batch的平均准确率
            accs.update(acc.item(), bsz)

            # answer span
            start_preds = torch.where(start_logits > 0.5, 1, 0)
            end_preds = torch.where(end_logits > 0.5, 1, 0)

            for px in range(bsz):
                mask_len = sum(mask[px]).item()  # remove padding positions
                input_postitions = [start_preds[px][:mask_len], end_preds[px][:mask_len],
                                    start_positions[px][:mask_len], end_positions[px][:mask_len]]
                num_correct, num_predict, num_gold, pred_pos, gold_pos = (
                                        utils.calculate_cpg(input=input_postitions))
                sc, sp, sg = sc + num_correct, sp + num_predict, sg + num_gold

                if is_test:
                    first_word_start = start_logits[px, 0]
                    first_word_end = end_logits[px, 0]
                    score_dict = {}

                    if pred_pos:
                        for (s, e) in pred_pos:
                            ans_start = start_logits[px, s]
                            ans_end = end_logits[px, e]
                            score = ans_start + ans_end - (first_word_start + first_word_end)
                            score_dict[(s, e)] = round(score.item(), 4)

                    result = {
                        "context_id": context_id[px],
                        "role": role[px],
                        "cls_predict": cls_preds[px].item(),
                        "label": int(labels[px].item()),
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
        save_result_path = config.INTENSIVE
        csv_columns = ["context_id", "role", "cls_predict", "label", 
                       "pred_position", "gold_position", "score2"]
        with open(save_result_path, "w", encoding="utf-8", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_columns)
            writer.writeheader()
            for d in results:
                writer.writerow(d)
        print("Test result saved in {}".format(save_result_path))

    return losses.avg, accs.avg, p, r, f1


def run(choice):
    print("MRC readers, using all training data.")
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    df_train = pd.read_csv(config.TRAIN_MRC_FILE).reset_index(drop=True)
#     df_train = df_train.sample(frac=0.3, random_state=config.SEED)
    df_test = pd.read_csv(config.TEST_MRC_FILE).reset_index(drop=True)
    print("train_size={}, test_size={}".format(len(df_train), len(df_test)))

    train_dataset = ArgumentDataset(
        role=df_train.role.values,
        context=df_train.context.values,
        has_answer=df_train.has_answer.values,
        answer=df_train.answer.values,
        context_id=df_train.context_id.values
    )
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=4
    )
    test_dataset = ArgumentDataset(
        role=df_test.role.values,
        context=df_test.context.values,
        has_answer=df_test.has_answer.values,
        answer=df_test.answer.values,
        context_id=df_test.context_id.values
    )
    test_data_loader = DataLoader(
        test_dataset,
        batch_size=config.VALID_BATCH_SIZE,
        num_workers=4
    )

    ModelDict = {
    "sketchy": (SketchyReader(), train_sketchy_reader, evaluate_sketchy_reader),
    "intensive": (IntensiveReader(), train_intensive_reader, evaluate_intensive_reader)
    }

    if choice == "sketchy":
        model, train_fn, eval_fn = ModelDict[choice]
        print("Choose sketchy reader.")
        model_path = config.SKETCHY_MODEL

    elif choice == "intensive":
        model, train_fn, eval_fn = ModelDict[choice]
        print("Choose intensive reader.")
        model_path = config.INTENSIVE_MODEL

    else:
        print("No model chosen.")

    device = torch.device(config.DEVICE)
    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.001},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],"weight_decay": 0.0}
    ]

    num_train_steps = int(len(df_train) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
    optimizer = AdamW(optimizer_parameters, lr=5e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )

    best_f1, best_acc = 0, 0
    r_t0 = time.time()
    for epoch in range(config.EPOCHS):
        print("\n=== Start training epoch {} ===".format(epoch))
        tloss = train_fn(train_data_loader, model, optimizer, device, scheduler) 
        out = eval_fn(test_data_loader, model, device)

        if len(out) == 2:
            l, acc = out
            print("train loss={:.4f} test loss={:.4f} acc={:.4f}".format(tloss, l, acc))
            if acc > best_acc:
                torch.save(model.state_dict(), model_path)
                best_acc = acc

        else:
            l, acc, p, r, f1 = out
            print("train loss={:.4f} test loss={:.4f} acc={:.4f} p={:.4f} r={:.4f} f1={:.4f}".format(tloss, l, acc, p, r, f1))
            if f1 > best_f1:
                torch.save(model.state_dict(), model_path)
                best_f1 = f1

    print("\nTotal took {:} (h:mm:ss)".format(utils.format_time(time.time()-r_t0)))
    print("\nFinished! {}".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    
    print("Final best results---")
    model.load_state_dict(torch.load(model_path))
    out = eval_fn(test_data_loader, model, device, is_test=True)
    if len(out) == 2:
        print("acc={:.4f}".format(out[1]))
    else:
        _, acc, p, r, f1 = out
        print("acc={:.4f} p={:.4f} r={:.4f} f1={:.4f}".format(acc, p, r, f1))


if __name__ == "__main__":
    run(choice = "sketchy")
    run(choice = "intensive")
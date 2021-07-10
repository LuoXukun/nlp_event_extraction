'''
Author: Xinrui Ma
Date: 2021-06-15 20:19:03
LastEditTime: 2021-07-03 16:12:27
Description: Using start/end probability to extract answer span,
             NLP EE work
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


class SEMODEL(BertPreTrainedModel):
    """ sequence tag --- predict answer,
        classify --- internal front verify,
        input: 
            question + context
        return: 
            answer span, start/end position
    """

    def __init__(self):
        super(SEMODEL, self).__init__(model_config)
        self.num_labels = 1  # sigmoid
        self.bert = BertModel.from_pretrained(config.BERT_PATH, config=model_config)
        self.drop_out = nn.Dropout(config.dropout_prob)

        # for question answering, predict start/end span
        self.qa_start = nn.Linear(768, self.num_labels)
        self.qa_end = nn.Linear(768, self.num_labels)

        nn.init.normal_(self.qa_start.weight, std=0.02)
        nn.init.normal_(self.qa_end.weight, std=0.02)

    def forward(self, input_ids, mask, token_type_ids, 
                start_positions, end_positions):

        outputs = self.bert(
            input_ids,
            attention_mask=mask,
            token_type_ids=token_type_ids
        )

        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        sequence_output = self.drop_out(sequence_output)

        # For start and end logits (span)
        start_logits = self.qa_start(sequence_output).squeeze()  # [batch_size, seq_len]
        end_logits = self.qa_end(sequence_output).squeeze()
        
        bsz, seq_len = mask.size(0), mask.size(1)
        start_logits = nn.Sigmoid()(start_logits).view(bsz, seq_len) # [batch_size, seq_len]
        
        end_logits = nn.Sigmoid()(end_logits).view(bsz, seq_len)
        
#         print("start_logits", start_logits)

        # calculate loss
        loss_fct = nn.BCELoss()
        start_loss = loss_fct(start_logits*mask.float(), start_positions)
        end_loss = loss_fct(end_logits*mask.float(), end_positions)
        loss = start_loss + end_loss
        
        return loss, start_logits, end_logits


def train_mrc_se(data_loader, model, optimizer, device, scheduler):
    model.train()
    losses = utils.AverageMeter()
    tk0 = tqdm(data_loader, total=len(data_loader), desc="Training")
    for _, d in enumerate(tk0):
        input_ids = d["input_ids"].to(device, dtype=torch.long)
        mask = d["mask"].to(device, dtype=torch.long)
        token_type_ids = d["token_type_ids"].to(device, dtype=torch.long)
        start_positions = d["start_positions"].to(device, dtype=torch.float)
        end_positions = d["end_positions"].to(device, dtype=torch.float)

        optimizer.zero_grad()
        loss, _, _ = model(
            input_ids=input_ids, mask=mask,
            token_type_ids=token_type_ids,
            start_positions=start_positions,
            end_positions=end_positions,
            )
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.update(loss.item(), input_ids.size(0))
        tk0.set_postfix(loss=losses.avg)
    return losses.avg


def evaluate_mrc_se(data_loader, model, device, is_test=False):
    model.eval()
    losses = utils.AverageMeter()
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
            labels = d["labels"].to(device, dtype=torch.long)
            context_id = d["context_id"]
            role = d["role"]

            loss, start_logits, end_logits = model(
                input_ids=input_ids, mask=mask,
                token_type_ids=token_type_ids,
                start_positions=start_positions,
                end_positions=end_positions)

            bsz = input_ids.size(0)
            losses.update(loss.item(), bsz)
            tk0.set_postfix(loss=losses.avg)

            # answer span
            start_preds = torch.where(start_logits > 0.5, 1, 0)
            end_preds = torch.where(end_logits > 0.5, 1, 0)

            for px in range(bsz):
                mask_len = sum(mask[px]).item()  # remove padding positions
                input_postitions = [start_preds[px][:mask_len], end_preds[px][:mask_len],
                                    start_positions[px][:mask_len], end_positions[px][:mask_len]]
                num_correct, num_predict, num_gold, pred_pos, gold_pos = (
                                        utils.calculate_cpg(input=input_postitions))
#                 print("input_postitions", input_postitions)
                sc, sp, sg = sc + num_correct, sp + num_predict, sg + num_gold

                if is_test:
                    result = {
                        "context_id": context_id[px],
                        "role": role[px],
                        "label": labels[px].item(),
                        "pred_position": pred_pos,
                        "gold_position": gold_pos,
                    }
#                     print("result", result)
                    results.append(result)

    f1 = 2.0 * sc / (sp + sg)
    p = 1.0 * sc / sp if sp > 1 else 0
    r = 1.0 * sc / sg
    print("[number correct={} predict={} golden={}]".format(sc, sp, sg))

    if is_test:
        save_result_path = config.MRC_SE
        csv_columns = ["context_id", "role", "label", 
                       "pred_position", "gold_position"]
        with open(save_result_path, "w", encoding="utf-8", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_columns)
            writer.writeheader()
            for d in results:
                writer.writerow(d)
        print("Test result saved in {}".format(save_result_path))

    return losses.avg, p, r, f1


def run():
    print("MRC SE, using all training data.")
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    df_train = pd.read_csv(config.TRAIN_MRC_FILE).reset_index(drop=True)
#     df_train = df_train.sample(frac=0.1, random_state=config.SEED)
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

    device = torch.device(config.DEVICE)
    model = SEMODEL()
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

    best_f1 = 0
    r_t0 = time.time()
    for epoch in range(config.EPOCHS):
        print("\n=== Start training epoch {} ===".format(epoch))
        tloss = train_mrc_se(train_data_loader, model, optimizer, device, scheduler) 
        l, p, r, f1 = evaluate_mrc_se(test_data_loader, model, device)
        print("train loss={:.4f} test loss={:.4f} p={:.4f} r={:.4f} f1={:.4f}".format(tloss, l, p, r, f1))

        if f1 > best_f1:
            torch.save(model.state_dict(), config.SEMODEL_PATH)
            best_f1 = f1

    print("\nTotal took {:} (h:mm:ss)".format(utils.format_time(time.time()-r_t0)))
    print("\nFinished! {}".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    
    print("Final best results---")
    model.load_state_dict(torch.load(config.SEMODEL_PATH))
    l, p, r, f1 = evaluate_mrc_se(test_data_loader, model, device, is_test=True)
    print("p={:.4f} r={:.4f} f1={:.4f}".format(p, r, f1))


if __name__ == "__main__":
    run()
'''
Author: Xinrui Ma
Date: 2021-06-27 13:50:37
LastEditTime: 2021-07-03 16:11:09
Description: train stage2, sketchy & intensive reader
'''

import config
from utils import calculate_token_cpg, make_text_map, format_time
from sketchyIntensive import SketchyReader, train_sketchy_reader, evaluate_sketchy_reader
from sketchyIntensive import IntensiveReader, train_intensive_reader, evaluate_intensive_reader
from dataset import ArgumentDataset

import os
import pandas as pd
import time
import datetime
import csv
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained(config.BERT_PATH)


ModelDict = {
    "sketchy": (SketchyReader(), train_sketchy_reader, evaluate_sketchy_reader),
    "intensive": (IntensiveReader(), train_intensive_reader, evaluate_intensive_reader)
    }



# Prepare dataset
def k_fold_split():
    df = pd.read_csv(config.TRAIN_MRC_FILE)
    df["kfold"] = -1
    df = df.sample(frac=config.FRACTION,
                   random_state=config.SEED).reset_index(drop=True)
    y = df.class_type.values
    kf = StratifiedKFold(n_splits=config.K_FOLD,
                         shuffle=True, random_state=config.SEED)
    for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_, 'kfold'] = f
    df.to_csv(config.TRAIN_MRC_FOLDS, index=False)
    print("Training k-folds data saved in {}, total size {}.".format(config.TRAIN_MRC_FOLDS, len(df)))


def build_model(choice):
    if choice in ModelDict:
        return ModelDict[choice]
    else:
        print("no model choose")
        return


def build_optimizer(model, train_size):
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],"weight_decay": 0.0}
    ]
    optimizer = AdamW(
        optimizer_parameters,
        lr=3e-5
    )
    num_train_steps = int(train_size / config.TRAIN_BATCH_SIZE * config.EPOCHS)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=100,
        num_training_steps=num_train_steps
    )
    return optimizer, scheduler


def get_data(dfx, fold):
    df_train = dfx[dfx.kfold != fold].reset_index(drop=True)
    df_valid = dfx[dfx.kfold == fold].reset_index(drop=True)
    print("train_size={}, valid_size={}\n".format(len(df_train), len(df_valid)))

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
    valid_dataset = ArgumentDataset(
        role=df_valid.role.values,
        context=df_valid.context.values,
        has_answer=df_valid.has_answer.values,
        answer=df_valid.answer.values,
        context_id=df_valid.context_id.values
    )
    valid_data_loader = DataLoader(
        valid_dataset,
        batch_size=config.VALID_BATCH_SIZE,
        num_workers=4
    )
    train_size = len(df_train)
    return train_data_loader, valid_data_loader, train_size


def train_kfold(choice, fold):
    print("\n=== Start training fold {} ===".format(fold))

    dfx = pd.read_csv(config.TRAIN_MRC_FOLDS)
    train_data_loader, valid_data_loader, train_size = get_data(dfx=dfx, fold=fold)

    device = config.DEVICE
    model, train_fn, eval_fn = build_model(choice=choice)
    model = model.to(device)
    optimizer, scheduler = build_optimizer(model, train_size)

    save_model_path = "fold_" + str(fold) + "_" + config.MODEL_PATH
    best_acc, best_f1 = 0, 0
    total_t0 = time.time()

    for epoch in range(1, config.EPOCHS + 1):
        train_loss = train_fn(train_data_loader, model, optimizer, device, scheduler)
        valid = eval_fn(valid_data_loader, model, device)

        if choice == "sketchy":
            valid_loss, valid_acc = valid
            print("Fold {} Epoch {} Train loss={:.4f} Valid loss={:.4f} Valid acc={:.4f}\n".format(
                fold, epoch, train_loss, valid_loss, valid_acc))
            if valid_acc > best_acc:
                torch.save(model.state_dict(), save_model_path)
                best_acc = valid_acc

        elif choice == "intensive":
            valid_loss, acc, f1, p, r = valid
            print("Fold {} Epoch {} Train loss={:.4f} Valid loss={:.4f} acc={:.4f} f1={:.4f} p={:.4f} r={:.4f}\n".format(
                fold, epoch, train_loss, valid_loss, acc, f1, p, r))
            if f1 > best_f1:
                torch.save(model.state_dict(), save_model_path)
                best_f1 = f1

        else:
            print("no choice model, no result")

    print("Fold {} took {:} (h:mm:ss)".format(fold, format_time(time.time()-total_t0)))


def predict_kfold(choice, fold):
    print("\n=== Start predicting fold {} ===".format(fold))

    df_test = pd.read_csv(config.TEST_MRC_FILE).reset_index(drop=True)
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
    device = config.DEVICE
    model, _, eval_fn = build_model(choice=choice)
    model = model.to(device)

    save_model_path = "fold_" + str(fold) + "_" + config.MODEL_PATH
    if not os.path.exists(save_model_path):
        print("There is no trained model, do cheak.")
    model.load_state_dict(torch.load(save_model_path))

    test = eval_fn(test_data_loader, model, device, is_test=True, fold=fold)

    if choice == "sketchy":
        test_loss, test_acc = test
        print("Fold {} Test loss={:.4f} acc={:.4f}\n".format(fold, test_loss, test_acc))

    elif choice == "intensive":
        test_loss, acc, f1, p, r = test
        print("Fold {} Test loss={:.4f} acc={:.4f} f1={:.4f} p={:.4f} r={:.4f}\n".format(
            fold, test_loss, acc, f1, p, r))

    else:
        print("no model, no result.")


def model_ensemble():
    """ using k-fold results, vote answers appearing more than half.
        5折交叉，出现次数>=3的论元作为最后输出
        待处理：sketchy和intensive两个阶段的score
        print exactly match, token level metrics
    """

    print("=== Start model ensemble ===")
    # level 1--context_id, 2--role, 3--pred_position, 4--count
    res_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    # level 1--context_id, 2--role, 3--gold_position
    gold_dict = defaultdict(lambda: defaultdict(list))

    for k in range(config.K_FOLD):
        save_result_path = "../output/" + "fold_" + str(k) + "_" + config.INTENSIVE
        res = pd.read_csv(save_result_path).reset_index(drop=True)

        if k == 0:
            for row in res.itertuples():
                gold_dict[getattr(row, 'context_id')][getattr(row, 'role')] = eval(getattr(row, 'gold_position'))

        for row in res.itertuples():
            l1 = getattr(row, 'context_id')
            l2 = getattr(row, 'role')
            l3 = getattr(row, 'pred_position')
            for argum in eval(l3):
                res_dict[l1][l2][argum] += 1

    correct, predict, golden = 0, 0, 0  # exact match
    tc, tp, tg = 0, 0, 0   # token level
    final_output = []
    for l1 in res_dict:
        for l2 in res_dict[l1]:
            l3_list = []  # may exist multiple answers
            gold = gold_dict[l1][l2]
            for l3 in res_dict[l1][l2]:
                if res_dict[l1][l2][l3] >= 3:
                    l3_list.append(l3)
                    if l3 in gold:
                        correct += 1
            output = {"context_id": l1, "role": l2, "pred": l3_list, "gold": gold}
            predict += len(l3_list)
            golden += len(gold)
            final_output.append(output)
            c, p, g = calculate_token_cpg(l3_list, gold)
            tc += c; tp += p; tg += g

    print("\nFinal result metrics:\n[Arugument number of correct {} predict {} golden {}]".format(
        correct, predict, golden))
    f1 = 2.0 * correct / (predict + golden)
    p = 1.0 * correct / predict if predict > 1 else 0
    r = 1.0 * correct / golden
    print("Exact match: f1={:.4f} p={:.4f} r={:.4f}\n".format(f1, p, r))

    print("[Token number correct {} predict {} golden {}]".format(tc, tp, tg))
    print("Token level: f1={:.4f} p={:.4f} r={:.4f}\n".format(
        2.0 * tc / (tp + tg), 1.0 * tc / tp, 1.0 * tc / tg))

    csv_columns = ["context_id", "role", "pred", "gold"]
    with open(config.ENSEMBLE, "w", encoding="utf-8", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()
        for d in final_output:
            writer.writerow(d)
    print("Final result saved in {}".format(config.ENSEMBLE))


def print_final_output():
    """ print tokens, using start and end positions
    """
    df_ensembel = pd.read_csv(config.ENSEMBLE).reset_index(drop=True)
    final_print = []
    text_map = make_text_map(config.TESTING_FILE)
    for row in df_ensembel.itertuples():
        context_id = getattr(row, 'context_id')
        context = text_map[context_id]
        role = getattr(row, 'role')
        pred = getattr(row, 'pred')
        gold = getattr(row, 'gold')

        # question
        q_tokens = tokenizer.tokenize(role)
        q_token_ids = tokenizer.convert_tokens_to_ids(q_tokens)
        # context
        c_tokens = tokenizer.tokenize(context)
        c_token_ids = tokenizer.convert_tokens_to_ids(c_tokens)

        # [CLS] question [SEP] context [SEP]
        input_ids = [101] + q_token_ids + [102] + c_token_ids + [102]

        predict_list, golden_list = [], []
        if pred:
            for (s, e) in eval(pred):
                ans_ids = input_ids[s: e + 1]
                ans = tokenizer.convert_ids_to_tokens(ans_ids)
                predict_list.append("".join(ans))
        if gold:
            for (s, e) in eval(gold):
                ans_ids = input_ids[s: e + 1]
                ans = tokenizer.convert_ids_to_tokens(ans_ids)
                golden_list.append("".join(ans))

        output = {"context_id": context_id, "role": role, "pred": predict_list, "gold": golden_list}
        final_print.append(output)

    csv_columns = ["context_id", "role", "pred", "gold"]
    with open(config.FINAL_PRINT, "w", encoding="utf-8", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()
        for d in final_print:
            writer.writerow(d)
    print("Final print saved in {}".format(config.FINAL_PRINT))


def run(choice):
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    k_fold_split()
    print("This is k-fold cross validation.\nTo extract argument.\nWe will use the GPU:",
          torch.cuda.get_device_name(0))
    print("\nUsing existed train data, do {} reader.".format(choice))

    r_t0 = time.time()
    for k in range(config.K_FOLD):
        train_kfold(choice=choice, fold=k)     # training and validating on train dataset
        predict_kfold(choice=choice, fold=k)   # testing on test dataset
    print("\nTotal took {:} (h:mm:ss)".format(format_time(time.time()-r_t0)))

    if choice == "intensive":
        model_ensemble()
        print_final_output()
    print("\nFinished! {}".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))


if __name__ == "__main__":
    run(choice="intensive")
'''
Author: Xinrui
Date: 2021-07-03 15:50:56
LastEditTime: 2021-07-03 16:18:39
'''
import pandas as pd
from collections import defaultdict
import utils

save_result_path = "../output/mrc_se.csv"
res = pd.read_csv(save_result_path).reset_index(drop=True)

correct, predict, golden = 0, 0, 0  # exact match
tc, tp, tg = 0, 0, 0   # token level

for row in res.itertuples():
    gold = eval(getattr(row, 'gold_position'))
    pred = eval(getattr(row, 'pred_position'))
    predict += len(pred)
    golden += len(gold)
    c, p, g = utils.calculate_token_cpg(pred, gold)
    tc += c; tp += p; tg += g

    
# print("\nFinal result metrics:\n[Arugument number of correct {} predict {} golden {}]".format(correct, predict, golden))
# f1 = 2.0 * correct / (predict + golden)
# p = 1.0 * correct / predict if predict > 1 else 0
# r = 1.0 * correct / golden
# print("Exact match: f1={:.4f} p={:.4f} r={:.4f}\n".format(f1, p, r))
print("[Token number correct {} predict {} golden {}]".format(tc, tp, tg))
print("Token level: f1={:.4f} p={:.4f} r={:.4f}\n".format(2.0 * tc / (tp + tg), 1.0 * tc / tp, 1.0 * tc / tg))

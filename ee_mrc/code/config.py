'''
Author: Xinrui Ma
Date: 2021-06-26 19:14:12
LastEditTime: 2021-07-03 17:28:52
Description: Configurations for EE work
'''

# {
#     "text": "北京时间昨天，南美足联决定：因为阿根廷球星梅西在今年美洲杯赛期间指责裁判腐败的言论，对他给予禁止随阿根廷队参加国际赛事3个月的处罚。根据这一决定，他将无法参加今年的4场友谊赛。与此同时，南美足联还对梅西罚款5万美元。", 
#     "id": "7a156965ccf1bc073e39399e601cddbd", 
#     "event_list": [
#         {
#             "event_type": "司法行为-罚款", 
#             "trigger": "罚款", 
#             "trigger_start_index": 101, 
#             "arguments": [
#                 {"argument_start_index": 0, "role": "时间", "argument": "北京时间昨天", "alias": []}, 
#                 {"argument_start_index": 7, "role": "执法机构", "argument": "南美足联", "alias": []}, 
#                 {"argument_start_index": 99, "role": "罚款对象", "argument": "梅西", "alias": []}, 
#                 {"argument_start_index": 103, "role": "罚款金额", "argument": "5万美元", "alias": []}], 
#             "class": "司法行为"
#         }, 
#         {
#             "event_type": "竞赛行为-禁赛", 
#             "trigger": "禁止", 
#             "trigger_start_index": 46, 
#             "arguments": [
#                 {"argument_start_index": 21, "role": "被禁赛人员", "argument": "梅西", "alias": []}, 
#                 {"argument_start_index": 7, "role": "禁赛机构", "argument": "南美足联", "alias": []}, 
#                 {"argument_start_index": 59, "role": "禁赛时长", "argument": "3个月", "alias": []}], 
#             "class": "竞赛行为"
#         }
#     ]
# }


# FILE
EVENT_SCHEMA_FILE = "../input/event_schema.json"
EVENT_TYPE_FILE = "../input/event_type_map.bin"
ROLE_MAP_FILE = "../input/role_map.bin"

TRAINING_FILE = "../input/train.json"
TESTING_FILE = "../input/test.json"

# SEQENCE LABEL
TRAINING_FILE_NER = "../output/train_ner.csv"
TESTING_FILE_NER = "../output/test_ner.csv"

# EVENT TYPE
TRAIN_FILE_EVENT_TYPE = "../input/train_event_type.json"
TEST_FILE_EVENT_TYPE = "../input/test_event_type.json"

TRAIN_WITH_EVENT = "../input/train_with_predict_event.json"
TRAIN_STAGE1 = "../input/train_stage1.json"
TEST_WITH_EVENT = "../input/test_with_predict_event.json"
TEST_STAGE1 = "../input/test_stage1.json"

# TRAIN_MRC_FILE = "../input/train_mrc.csv"
# TEST_MRC_FILE = "../input/test_mrc.csv"


TRAIN_MRC_FILE = "/home/featurize/data/train_mrc.csv"
TEST_MRC_FILE = "/home/featurize/data/test_mrc.csv"


# MRC_SE
MRC_SE = "../output/mrc_se.csv"
SEMODEL_PATH = "mrc_se_model.bin"
# MRC_IO
MRC_IO = "../output/mrc_io.csv"
IOMODEL_PATH = "mrc_io_model.bin"

TRAIN_MRC_FOLDS = "../output/train_mrc_folds.csv"

SKETCHY = "sketchy_result.csv"
INTENSIVE = "intensive_result.csv"
SKETCHY_MODEL = "sketchy.bin"
INTENSIVE_MODEL = "intensive.bin"


ENSEMBLE = "../output/ensemble.csv"
FINAL_PRINT = "../output/final.csv"

# PRETRAINED MODEL
MAX_LEN = 150
# BERT_PATH = "../input/chinese-bert-wwm-ext/"
BERT_PATH = "/home/featurize/data/chinese-bert-wwm-ext"

dropout_prob = 0.3

SEED = 123
FRACTION = 1  # 若小于1则选择部分数据训练
K_FOLD = 5

EPOCHS = 10
TRAIN_BATCH_SIZE = 64
VALID_BATCH_SIZE = 64

DEVICE = "cuda"
MODEL_PATH = "model.bin"




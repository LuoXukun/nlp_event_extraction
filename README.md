# nlp_event_extraction



### 实验命令

#### 1. BERT预训练

```
cd /event_extraction_codes

python3 utils.py

cd ../

python3 preprocess.py \
	--corpus_path corpora/event_corpus.txt \
	--vocab_path models/google_zh_vocab.txt \
	--dataset_path corpora/event_dataset.pt --processes_num 8 --target bert

CUDA_VISIBLE_DEVICES=6,7,8,9 python3 pretrain.py \
	--dataset_path corpora/event_dataset.pt --vocab_path models/google_zh_vocab.txt \
	--pretrained_model_path models/bert/google_zh_model.bin \
	--output_model_path models/bert/event_bert_model.bin \
	--world_size 4 --gpu_ranks 0 1 2 3 \
	--total_steps 5000 --save_checkpoint_steps 5000 --batch_size 32 \
	--embedding word_pos_seg --encoder transformer --mask fully_visible --target bert
```

注：预训练完成的bert模型之后提供（服务器崩了连不上），可以自己预训练

若要预训练，需要先下载`google_zh_model.bin`到`models/bert/`,下载链接如下：

```
链接：https://pan.baidu.com/s/1E-cXEaVDPtHtfAdC0S21eA 
提取码：00dd
```

#### 2. 模型训练

- `Baseline (BERT-base + linear)`

  分事件类型进行角色的序列标注

  ```
  CUDA_VISIBLE_DEVICES=6,7,8,9 python3 train.py --learning_rate 5e-5
  
  # baseline_criterion_weigth = [1.0, 0.0, 50.0]; 100 epochs; Optimizer: linear.
  CUDA_VISIBLE_DEVICES=6,7,8,9 nohup python3 -u train.py --learning_rate 5e-5 > ../logs/baseline.out 2>&1 &!
  
  验证
  CUDA_VISIBLE_DEVICES=6,7,8,9 python3 train.py \
  	--middle_model_path ../result_models/baseline/best/model0.bin
  	
  结果
  Event: total_right, total_predict, predict_right: 1657, 1771, 1489
  Event: precision, recall, and f1: 0.841, 0.899, 0.869
  Role: total_right, total_predict, predict_right: 3696, 4005, 2391
  Role: precision, recall, and f1: 0.597, 0.647, 0.621
  
  # baseline_criterion_weigth = [1.0, 0.0, 50.0]; 200 epochs; Optimizer: polynomial.
  CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python3 -u train.py \
  	--learning_rate 5e-5 --scheduler polynomial --epochs_num 200 \
  	--save_dir_name baseline-polynomial \
  	> ../logs/baseline-polynomial.out 2>&1 &!

  CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python3 -u train.py \
  	--learning_rate 5e-5 --scheduler polynomial --epochs_num 200 \
    --middle_model_path ../result_models/baseline-polynomial/best/model0.bin \
  	--save_dir_name baseline-polynomial \
  	> ../logs/baseline-polynomial.out 2>&1 &!

  Event: total_right, total_predict, predict_right: 1657, 1729, 1493
  Event: precision, recall, and f1: 0.864, 0.901, 0.882
  Role: total_right, total_predict, predict_right: 3696, 3865, 2347
  Role: precision, recall, and f1: 0.607, 0.635, 0.621

  # baseline_criterion_weigth = [1.0, 0.0, 50.0]; 400 epochs; Optimizer: polynomial.
  CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python3 -u train.py \
  	--learning_rate 5e-5 --scheduler polynomial --epochs_num 400 \
  	--save_dir_name baseline-polynomial \
  	> ../logs/baseline-polynomial.out 2>&1 &!
  ```

- `Baseline-lstm (BERT-base + LSTM + linear)`

  ```
  # ptimizer: polynomial
  CUDA_VISIBLE_DEVICES=6,7,8,9 python3 train.py \
  	--learning_rate 5e-5 --model_type baseline-lstm

  CUDA_VISIBLE_DEVICES=6,7,8,9 nohup python3 -u train.py \
  	--learning_rate 5e-5 \
  	--model_type baseline-lstm > ../logs/baseline-lstm.out 2>&1 &!
```
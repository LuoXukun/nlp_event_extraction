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



#### 2. 模型训练

- `Baseline (BERT-base + linear)`

  分事件类型进行角色的序列标注

  ```
  CUDA_VISIBLE_DEVICES=6,7,8,9 python3 train.py --learning_rate 5e-5
  
  CUDA_VISIBLE_DEVICES=6,7,8,9 nohup python3 -u train.py --learning_rate 5e-5 > ../logs/baseline.out 2>&1 &!
  
  验证
  CUDA_VISIBLE_DEVICES=6,7,8,9 python3 train.py \
  	--middle_model_path ../result_models/baseline/best/model0.bin
  ```

- `Baseline-lstm (BERT-base + LSTM + linear)`


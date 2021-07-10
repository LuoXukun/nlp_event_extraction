# NLP第二次大作业实验

主要模型代码文件和结果报告文件在文件夹`/ee_mrc/`和`/event_extraction_codes/`中

- `/ee_mrc/`：包括成功运行的模型和日志文件

  - `Seq_label`模型：对应的代码文件为`/ee_mrc/code/seqlabel.py`，日志文件为`/ee_mrc/log/seqlabel.txt`
  - `SPN`模型：直接使用了该模型的[开源代码](https://github.com/DianboWork/SPN4RE)，日志文件为`/ee_mrc/log/SPN.txt`
  - `MRC_se`模型：对应的代码文件为`/ee_mrc/code/mrc_se.py`，日志文件为`/ee_mrc/log/mrc_se.txt`
  - `MRC_io`模型：对应的代码文件为`/ee_mrc/code/mrc_io.py`，日志文件为`/ee_mrc/log/mrc_io.txt`
  - `MRC_readers`模型：对应的代码文件为`/ee_mrc/code/mrc_readers.py`，日志文件为`/ee_mrc/log/mrc_readers.txt`

  需要注意的是，这些代码文件直接运行即可，即`python *.py`。但需要先将事件抽取原始数据集的`train.json`和`test.json`加入到文件夹``/ee_mrc/input/`中。

- `/event_extraction_codes/`：主要是未能成功跑出效果的模型`CasRel`以及一些针对0、1类别不平衡的尝试，模型代码主要在`/event_extraction_codes/models/cascade.py`文件中，模型日志在`/logs/`文件夹中，这部分代码运行请查看`/README_CAS.md`。
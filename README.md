# 合同法三元组抽取代码


## 实验环境
This repo was tested on Python 3.7 and Keras 2.2.4. The main requirements are:
- tqdm
- codecs
- keras-bert
- tensorflow-gpu == 1.13.1

## 各个文件夹说明
data/HTF: 合同法数据集
pretrained_bert_models: 预训练好的中文bert模型
## 用法
1. 参照pretrained_bert_models/README.md文件下载预训练模型
2. 使用python run.py --train=True进行模型训练
3. 完成训练后，使用python run.py --train=False命令进行在测试集上测试
4. 使用python run.py --train=False --extract=True进行三元组的抽取，在data/baike_text/文件夹下放入要抽取的文本(txt文件)
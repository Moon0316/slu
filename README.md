### 创建环境

    conda create -n slu python=3.6
    source activate slu
    pip install torch==1.7.1

### 运行
    
在根目录下运行

    python scripts/slu_baseline.py

### 代码说明

+ `utils/args.py`:定义了所有涉及到的可选参数，如需改动某一参数可以在运行的时候将命令修改成
        
        python scripts/slu_baseline.py --<arg> <value>
    其中，`<arg>`为要修改的参数名，`<value>`为修改后的值
+ `utils/initialization.py`:初始化系统设置，包括设置随机种子和显卡/CPU
+ `utils/vocab.py`:构建编码输入输出的词表
+ `utils/word2vec.py`:读取词向量
+ `utils/example.py`:读取数据
+ `utils/batch.py`:将数据以批为单位转化为输入
+ `model/slu_baseline_tagging.py`:baseline模型
+ `scripts/slu_baseline.py`:主程序脚本
+ `asrcor.py`:针对 train.json, 对语音识别结果于以修正

#### run.sh使用说明
为了快速进行实验、规范记录实验参数，可以运行run.sh脚本传入参数。
例如`bash run.sh bert 0.00001 32 10 0.5 50 1`表示使用bert作为预训练模型，学习率为0.00001，batch_size为32，lr scheduler的step size为10、gamma为0.5，epoch数为50，gpuid为1。

运行命令后，在`exp/bert_lr_0.00001_bs_32_step_10_gamma_0.5_ep_50/checkpoint`下面会保存最佳模型文件，在`exp/bert_lr_0.00001_bs_32_step_10_gamma_0.5_ep_50/config.json`中会保存实验配置，在`exp/bert_lr_0.00001_bs_32_step_10_gamma_0.5_ep_50/log.txt`中会保存训练日志

### 有关预训练语言模型
已使用的预训练模型：
+ pycorrector: https://github.com/shibing624/pycorrector

本次代码中没有加入有关预训练语言模型的代码，如需使用预训练语言模型我们推荐使用下面几个预训练模型，若使用预训练语言模型，不要使用large级别的模型
+ Bert: https://huggingface.co/bert-base-chinese
+ Bert-WWM: https://huggingface.co/hfl/chinese-bert-wwm-ext
+ Roberta-WWM: https://huggingface.co/hfl/chinese-roberta-wwm-ext
+ MacBert: https://huggingface.co/hfl/chinese-macbert-base

### 推荐使用的工具库
+ transformers
  + 使用预训练语言模型的工具库: https://huggingface.co/
+ nltk
  + 强力的NLP工具库: https://www.nltk.org/
+ stanza
  + 强力的NLP工具库: https://stanfordnlp.github.io/stanza/
+ jieba
  + 中文分词工具: https://github.com/fxsjy/jieba

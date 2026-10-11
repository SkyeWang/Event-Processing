# -*- coding: utf-8 -*-
"""
论文4 复现 · 诊断：训练管线能否学会一个"显然可学"的合成任务
构造合成数据：随机字序列，凡出现特殊字符 'X' 即标为一个单字实体（S-ly），其余为 O。
若作者代码的 BiLSTM+CRF 训练管线正常，F1 应迅速接近 1；否则说明训练/解码环节有问题。
"""
import os
import sys
import random
import numpy as np
import torch

NER = r"C:\Users\LiYi\Desktop\大创文件\_appx\p4\ner"
sys.path.insert(0, NER)
os.chdir(NER)

from utils import extend_maps, prepocess_data_for_lstmcrf      # noqa: E402
from models.config import TrainingConfig, LSTMConfig           # noqa: E402
from models.bilstm_crf import BILSTM_Model                     # noqa: E402
from BIOES_eval import classification_report                   # noqa: E402

TrainingConfig.epoches = int(sys.argv[1]) if len(sys.argv) > 1 else 5
LSTMConfig.use_entity = False          # 先测最纯粹的 BiLSTM+CRF
LSTMConfig.use_similarity = False
LSTMConfig.word_level = False

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

VOCAB = list("abcdefg")
LEN = 20


def gen(n):
    ws, ts = [], []
    for _ in range(n):
        w, t = [], []
        for _ in range(LEN):
            c = random.choice(VOCAB)
            if random.random() < 0.15:
                c = "X"
            w.append(c)
            t.append("S-ly" if c == "X" else "O")
        ws.append(w); ts.append(t)
    return ws, ts


train_w, train_t = gen(400)
dev_w, dev_t = gen(100)
test_w, test_t = gen(100)

word2id = {"<unk>": 0}
tag2id = {}
for seq in train_w:
    for w in seq:
        word2id.setdefault(w, len(word2id))
for seq in train_t:
    for t in seq:
        tag2id.setdefault(t, len(tag2id))

crf_word2id, crf_tag2id = extend_maps(word2id, tag2id, for_crf=True)
train_w, train_t = prepocess_data_for_lstmcrf(train_w, train_t)
dev_w, dev_t = prepocess_data_for_lstmcrf(dev_w, dev_t)
test_w, test_t = prepocess_data_for_lstmcrf(test_w, test_t, test=True)

print(f"合成任务：字表 {len(word2id)}，标签 {crf_tag2id}")
print(f"一句话里平均 {sum(1 for s in train_t for x in s if x=='S-ly'):.0f} 个实体标记")

model = BILSTM_Model(len(crf_word2id), len(crf_tag2id), [], crf=True)
for e in range(1, TrainingConfig.epoches + 1):
    model.epoches = 1
    model.train(train_w, train_t, dev_w, dev_t, crf_word2id, crf_tag2id)
    pred, gold = model.test_with_btach(dev_w[:100], dev_t[:100], crf_word2id, crf_tag2id)
    _, mv, _ = classification_report(gold, pred)
    print(f"  第 {e} 轮：dev F1 = {mv['f1']:.4f}   样例 pred[:30]={pred[0][:30]}")
    print(f"            gold[:30]={gold[0][:30]}")

pred, gold = model.test_with_btach(test_w, test_t, crf_word2id, crf_tag2id)
_, mv, _ = classification_report(gold, pred)
print(f"\n最终 test micro = {mv}")
print(f"pred 中出现的标签集合 = {sorted(set(x for s in pred for x in s))}")
print(f"gold 中出现的标签集合 = {sorted(set(x for s in gold for x in s))}")

# -*- coding: utf-8 -*-
"""
论文4 复现 · 诊断：作者 cal_lstm_crf_loss 的数值正确性
做法：构造与真实标签集同构的 tag2id（含 <unk>/<pad>/<start>/<end>），
      用随机 emission + 随机 transition 组装 crf_scores，
      把作者函数的 loss 与「暴力枚举全部标签路径」得到的精确 NLL 对比。
"""
import os
import sys
import itertools
import numpy as np
import torch

NER = r"C:\Users\LiYi\Desktop\大创文件\_appx\p4\ner"
sys.path.insert(0, NER)
os.chdir(NER)

from models.util import cal_lstm_crf_loss, indexed   # noqa: E402

TAG = ["O", "B", "I", "E", "S", "B-ly", "I-ly", "E-ly",
       "<unk>", "<pad>", "<start>", "<end>"]
tag2id = {t: i for i, t in enumerate(TAG)}
T = len(TAG)
PAD, START, END = tag2id["<pad>"], tag2id["<start>"], tag2id["<end>"]
print("标签数 T =", T, " pad/start/end =", PAD, START, END)


def brute_nll(emission, transition, tags, use_regular_end=True):
    """精确 NLL：montecarlo 之外的做法 —— 枚举全部路径。
    emission:[L,T] transition:[T,T] tags:[L]（含末尾的 <end>）
    路径空间与作者递归一致：长度为 L，最后一位取 <end>，其余位取任意真实标签。"""
    L = emission.shape[0]
    real = [i for i in range(T) if i not in (PAD, START, END)]
    best = -np.inf
    scores = []
    for combo in itertools.product(real, repeat=L - 1):
        y = list(combo) + [END]
        s = transition[START, y[0]] + emission[0, y[0]]
        for t in range(1, L):
            s += transition[y[t - 1], y[t]] + emission[t, y[t]]
        scores.append(s)
        if s > best:
            best = s
    logZ = float(np.logaddexp.reduce(scores))
    g = transition[START, tags[0]] + emission[0, tags[0]]
    for t in range(1, L):
        g += transition[tags[t - 1], tags[t]] + emission[t, tags[t]]
    return logZ - g, logZ, g


def author_nll(emission, transition, tag_lists):
    """按作者模型 forward 的方式组装 crf_scores，再调用作者的损失函数。"""
    emission = torch.tensor(emission, dtype=torch.float32)
    transition = torch.tensor(transition, dtype=torch.float32)
    B = emission.shape[0]
    crf_scores = emission.unsqueeze(2).expand(-1, -1, T, -1) + transition.unsqueeze(0)
    maxlen = max(len(t) for t in tag_lists)
    tgt = torch.full((B, maxlen), PAD, dtype=torch.long)
    for i, tl in enumerate(tag_lists):
        tgt[i, :len(tl)] = torch.tensor(tl, dtype=torch.long)
    loss = cal_lstm_crf_loss(crf_scores, tgt, tag2id)
    return float(loss) * B       # 作者函数除了 batch_size，这里还原成总和


rng = np.random.default_rng(0)
print("\n" + "=" * 84)
print("单序列、无 padding：作者 loss vs 暴力枚举精确 NLL")
print("=" * 84)
print(f"{'用例':<6}{'L':>4}{'作者 loss(总和)':>20}{'精确 NLL':>16}{'差值':>14}{'符号':>8}")
for k in range(6):
    L = int(rng.integers(4, 8))
    emission = rng.normal(0, 1.0, (1, L, T))
    transition = rng.normal(0, 0.5, (T, T))
    tags = [int(rng.choice([i for i in range(T) if i not in (PAD, START, END)]))
            for _ in range(L - 1)] + [END]
    a = author_nll(emission, transition, [tags])
    b, logZ, gold = brute_nll(emission[0], transition, tags)
    print(f"{k:<6}{L:>4}{a:>20.6f}{b:>16.6f}{a-b:>14.6f}{'负' if a < 0 else '正':>8}")

print("\n" + "=" * 84)
print("批量 + padding（模拟真实训练）：作者 loss 是否可能为负")
print("=" * 84)
neg = 0
N = 200
for k in range(N):
    lens = [int(rng.integers(3, 9)) for _ in range(4)]
    emission = rng.normal(0, 1.0, (4, max(lens), T))
    transition = rng.normal(0, 0.5, (T, T))
    tl = []
    for L in lens:
        tl.append([int(rng.choice([i for i in range(T) if i not in (PAD, START, END)]))
                   for _ in range(L - 1)] + [END])
    a = author_nll(emission, transition, tl)
    if a < 0:
        neg += 1
print(f"  {N} 次批量随机测试中，作者 loss < 0 的次数 = {neg}")

# 复核：把 transition 设成模型初始化值 1/T，emission 用真实训练分布量级
print("\n" + "=" * 84)
print("附加复核：transition 取模型初始化值 1/T（源码 bilstm_crf.py:494-495）")
print("=" * 84)
transition = np.full((T, T), 1.0 / T)
for scale in (0.1, 1.0, 3.0):
    neg = 0
    for k in range(100):
        lens = [int(rng.integers(3, 9)) for _ in range(4)]
        emission = rng.normal(0, scale, (4, max(lens), T))
        tl = [[int(rng.choice([i for i in range(T) if i not in (PAD, START, END)]))
               for _ in range(L - 1)] + [END] for L in lens]
        if author_nll(emission, transition, tl) < 0:
            neg += 1
    print(f"  emission 标准差={scale}: 100 次中 loss<0 出现 {neg} 次")

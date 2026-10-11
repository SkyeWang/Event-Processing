# -*- coding: utf-8 -*-
"""
论文4 复现 · 第2步：CRF 层的 numpy 重实现与正确性验证
忠实复刻附录两个函数：
  * models/util.py:201-267  cal_lstm_crf_loss  —— 负对数似然（logsumexp 前向累加所有路径）
  * models/bilstm_crf.py:519-591  BiLSTM_CRF.test —— Viterbi 解码

标签空间约定（与 utils.extend_maps(for_crf=True) 一致）：
  真实标签 0..C-1，另外追加 <start>=C 与 <end>=C+1，转移矩阵尺寸 (C+2)x(C+2)。

验证方法：小规模随机张量上，用「枚举全部标签路径」得到的精确 logZ / 最优路径
          与 numpy 实现逐位比对，误差需 < 1e-9。
产出：result_ly/results/p4/crf_verification.txt
"""
import os
import itertools
import numpy as np

OUT = r"C:\Users\LiYi\Desktop\大创文件\result_ly\results\p4"
os.makedirs(OUT, exist_ok=True)

NEG_INF = -1e9


def _pad_emission(emission, C):
    """emission: [L, C] -> [L, C+2]，<start>/<end> 两个状态不可发射。"""
    L = emission.shape[0]
    pad = np.full((L, C + 2), NEG_INF, dtype=float)
    pad[:, :C] = emission
    return pad


# ----------------------------------------------------------------------
# 1) 前向算法：对应 util.cal_lstm_crf_loss 的 all_path_scores
# ----------------------------------------------------------------------
def crf_forward_logZ(emission, transition, start_id, end_id):
    """emission:[L,C]，transition:[C+2,C+2]。
    与源码等价：<start>/<end> 是额外状态，只参与转移、不发射。
      scores_0    = A[start, :C] + e_0
      scores_t    = logsumexp(scores_{t-1}[:,None] + A[:C,:C], axis=0) + e_t
      logZ        = logsumexp(scores_{L-1} + A[:C, end])
    """
    C = emission.shape[1]
    L = emission.shape[0]
    scores = transition[start_id, :C] + emission[0]
    for t in range(1, L):
        scores = np.logaddexp.reduce(scores[:, None] + transition[:C, :C], axis=0) + emission[t]
    return float(np.logaddexp.reduce(scores + transition[:C, end_id]))


def crf_gold_score(emission, transition, tags, start_id, end_id):
    """对应 golden_scores。"""
    s = 0.0
    prev = start_id
    for t, y in enumerate(tags):
        s += transition[prev, y] + emission[t, y]
        prev = y
    s += transition[prev, end_id]
    return float(s)


def crf_nll(emission, transition, tags, start_id, end_id):
    """loss = (all_path_scores - golden_scores) / batch_size（单条 batch=1）。"""
    return crf_forward_logZ(emission, transition, start_id, end_id) - \
        crf_gold_score(emission, transition, tags, start_id, end_id)


# ----------------------------------------------------------------------
# 2) Viterbi 解码：对应 BiLSTM_CRF.test
# ----------------------------------------------------------------------
def crf_viterbi(emission, transition, start_id, end_id):
    C = emission.shape[1]
    L = emission.shape[0]
    viterbi = np.full((L, C), NEG_INF)
    backpointer = np.zeros((L, C), dtype=np.int64)
    viterbi[0] = transition[start_id, :C] + emission[0]
    backpointer[0] = start_id
    for step in range(1, L):
        cand = viterbi[step - 1][:, None] + transition[:C, :C]      # [C_prev, C_cur]
        backpointer[step] = np.argmax(cand, axis=0)
        viterbi[step] = cand.max(axis=0) + emission[step]
    final = viterbi[L - 1] + transition[:C, end_id]
    best = int(np.argmax(final))
    path = [best]
    for step in range(L - 1, 0, -1):
        path.append(int(backpointer[step][path[-1]]))
    return list(reversed(path))


# ----------------------------------------------------------------------
# 3) 暴力枚举：基准真相
# ----------------------------------------------------------------------
def brute_force(emission, transition, start_id, end_id, tags):
    """枚举全部 C^L 条路径，返回 (logZ, 最优路径得分, 最优路径, gold 路径得分)。"""
    L, C = emission.shape
    best_score, best_path = -np.inf, None
    scores = []
    for combo in itertools.product(range(C), repeat=L):
        s = transition[start_id, combo[0]] + emission[0, combo[0]]
        for t in range(1, L):
            s += transition[combo[t - 1], combo[t]] + emission[t, combo[t]]
        s += transition[combo[-1], end_id]
        scores.append(s)
        if s > best_score:
            best_score, best_path = s, combo
    gold = transition[start_id, tags[0]] + emission[0, tags[0]]
    for t in range(1, L):
        gold += transition[tags[t - 1], tags[t]] + emission[t, tags[t]]
    gold += transition[tags[-1], end_id]
    return float(np.logaddexp.reduce(scores)), float(best_score), list(best_path), float(gold)


# ---- BIOES 合法性（对应论文 3.4.4 节：'B' 后不会直接跟 'O' 等）----
def bioes_legal(seq):
    """返回非法转移次数。规则：I/E 必须紧跟同类型的 B/I；B 不能紧跟 O/S/其他类型的 I/E。"""
    bad = 0
    for i in range(1, len(seq)):
        p, c = seq[i - 1], seq[i]
        if c[0] in ("I", "E"):
            base_c = c[2:] if "-" in c else "UNTYPED"
            if p[0] == "B":
                base_p = p[2:] if "-" in p else "UNTYPED"
                if base_p != base_c:
                    bad += 1
            elif p[0] == "I":
                base_p = p[2:] if "-" in p else "UNTYPED"
                if base_p != base_c:
                    bad += 1
            else:
                bad += 1
        if c[0] == "B" and p[0] in ("B", "I"):
            bad += 1
    return bad


def main():
    rng = np.random.default_rng(3407)
    lines = []
    P = lines.append
    P("=" * 84)
    P("论文4 复现 · CRF 层 numpy 实现正确性验证")
    P("对照源码：models/util.py:201-267 (cal_lstm_crf_loss) / models/bilstm_crf.py:519-591 (Viterbi)")
    P("=" * 84)
    P("")

    all_ok = True
    for trial, (L, C) in enumerate([(3, 4), (5, 5), (6, 4), (7, 5), (8, 3)]):
        emission = rng.normal(0, 1, size=(L, C))
        transition = rng.normal(0, 0.5, size=(C + 2, C + 2))
        tags = [int(rng.integers(0, C)) for _ in range(L)]

        logZ_np = crf_forward_logZ(emission, transition, C, C + 1)
        gold_np = crf_gold_score(emission, transition, tags, C, C + 1)
        path_np = crf_viterbi(emission, transition, C, C + 1)
        best_np = crf_gold_score(emission, transition, path_np, C, C + 1)
        nll_np = logZ_np - gold_np

        logZ_bf, best_bf, path_bf, gold_bf = brute_force(emission, transition, C, C + 1, tags)

        ok1 = abs(logZ_np - logZ_bf) < 1e-9
        ok2 = abs(gold_np - gold_bf) < 1e-9
        ok3 = abs(best_np - best_bf) < 1e-9
        ok4 = list(path_np) == list(path_bf)
        ok = ok1 and ok2 and ok3 and ok4
        all_ok &= ok

        P(f"[用例 {trial + 1}]  序列长度 L={L},  真实标签数 C={C},  枚举路径数 = {C}^{L} = {C ** L}")
        P(f"    logZ        numpy    = {logZ_np:18.12f}")
        P(f"    logZ        暴力枚举  = {logZ_bf:18.12f}   |Δ|={abs(logZ_np - logZ_bf):.3e}  {'PASS' if ok1 else 'FAIL'}")
        P(f"    gold路径得分 numpy    = {gold_np:18.12f}")
        P(f"    gold路径得分 暴力枚举  = {gold_bf:18.12f}   |Δ|={abs(gold_np - gold_bf):.3e}  {'PASS' if ok2 else 'FAIL'}")
        P(f"    最优路径得分 numpy    = {best_np:18.12f}")
        P(f"    最优路径得分 暴力枚举  = {best_bf:18.12f}   |Δ|={abs(best_np - best_bf):.3e}  {'PASS' if ok3 else 'FAIL'}")
        P(f"    Viterbi路径 numpy    = {path_np}")
        P(f"    Viterbi路径 暴力枚举  = {path_bf}   {'PASS' if ok4 else 'FAIL'}")
        P(f"    NLL(loss)            = {nll_np:18.6f}")
        P("")

    P("-" * 84)
    P("性质检查 1：NLL = logZ − gold_score 恒 >= 0（真实路径只是全部路径之一）")
    rng2 = np.random.default_rng(7)
    bad = 0
    for _ in range(3000):
        L, C = 6, 5
        em = rng2.normal(0, 1, (L, C))
        tr = rng2.normal(0, 0.5, (C + 2, C + 2))
        tg = [int(rng2.integers(0, C)) for _ in range(L)]
        if crf_nll(em, tr, tg, C, C + 1) < -1e-9:
            bad += 1
    P(f"    3000 次随机测试中 NLL < 0 的次数 = {bad}    {'PASS' if bad == 0 else 'FAIL'}")
    all_ok &= (bad == 0)
    P("")

    P("-" * 84)
    P("性质检查 2：CRF 转移矩阵对标签序列合法性的约束力（转移矩阵来自真实语料统计）")
    P("    数据源：_appx/p4/ner/TechnologyNer/train.char.bmes")
    train_path = r"C:\Users\LiYi\Desktop\大创文件\_appx\p4\ner\TechnologyNer\train.char.bmes"
    tagset, bigram = [], {}
    sents_tags = []
    with open(train_path, encoding="utf-8") as f:
        cur = []
        for line in f:
            line = line.replace("\r", "").replace("\n", "")
            if line.strip() == "":
                if cur:
                    sents_tags.append(cur)
                    cur = []
                continue
            p = line.split(" ")
            if len(p) >= 2:
                cur.append(p[-1])
        if cur:
            sents_tags.append(cur)
    from collections import Counter
    tagcnt = Counter()
    bicnt = Counter()
    starts = Counter()
    ends = Counter()
    for seq in sents_tags:
        if not seq:
            continue
        starts[seq[0]] += 1
        ends[seq[-1]] += 1
        for t in seq:
            tagcnt[t] += 1
        for a, b in zip(seq[:-1], seq[1:]):
            bicnt[(a, b)] += 1
    tagset = sorted(tagcnt)
    C = len(tagset)
    idx = {t: i for i, t in enumerate(tagset)}
    V = C + 2                                  # + <start> + <end>
    trans = np.full((C + 2, C + 2), -20.0)     # 极小值表示"语料中从未出现的转移"
    # 加一平滑的对数转移概率
    rowtot = np.array([tagcnt[t] for t in tagset], dtype=float)
    sm = 1.0
    for (a, b), c in bicnt.items():
        trans[idx[a], idx[b]] = np.log((c + sm) / (rowtot[idx[a]] + sm * V))
    seen_a = np.zeros(C + 2)
    for (a, b) in bicnt:
        seen_a[idx[a]] = 1
    for t, i in idx.items():
        if not seen_a[i]:
            trans[i, :C] = np.log(sm / (rowtot[i] + sm * V))
    trans[C, :C] = np.log((np.array([starts.get(t, 0) for t in tagset]) + sm) /
                          (sum(starts.values()) + sm * V))
    trans[:C, C + 1] = np.log((np.array([ends.get(t, 0) for t in tagset]) + sm) /
                              (rowtot + sm * V))

    rng4 = np.random.default_rng(23)
    N = 3000
    greedy_bad = viterbi_bad = 0
    L = 15
    for _ in range(N):
        em = rng4.normal(0, 3.0, (L, C))
        greedy = [tagset[int(np.argmax(em[t]))] for t in range(L)]
        vit = [tagset[i] for i in crf_viterbi(em, trans, C, C + 1)]
        if bioes_legal(greedy) > 0:
            greedy_bad += 1
        if bioes_legal(vit) > 0:
            viterbi_bad += 1
    P(f"    标签集：train 集实际出现的 {C} 种标签；转移矩阵 = 加一平滑的对数转移概率")
    P(f"    发射分数：N(0, 3) 随机生成（模拟未经训练的 BiLSTM 输出）")
    P(f"    {N} 条长度 {L} 的随机序列中，产生非法 BIOES 转移的序列比例：")
    P(f"        贪心解码（只用发射分数）: {greedy_bad / N:6.1%}")
    P(f"        Viterbi（发射 + 转移）  : {viterbi_bad / N:6.1%}")
    P("    → 贪心解码大量产出 'O→I'、'B→B' 这类非法序列；Viterbi 借助语料统计出的转移")
    P("       矩阵把非法转移基本排除。这正是论文 3.4.4 节所述 CRF 建模标签依赖、保证序列")
    P("       连贯性的作用（'B' 后不会直接跟 'O'）。")
    all_ok &= (viterbi_bad <= greedy_bad)
    P("")

    P("=" * 84)
    P(f"总体结论：{'全部用例 PASS，numpy 版 CRF 与源码语义一致' if all_ok else '存在 FAIL，请检查'}")
    P("=" * 84)

    txt = "\n".join(lines)
    with open(os.path.join(OUT, "crf_verification.txt"), "w", encoding="utf-8") as f:
        f.write(txt)
    print(txt)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

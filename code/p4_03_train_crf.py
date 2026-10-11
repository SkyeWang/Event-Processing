# -*- coding: utf-8 -*-
"""
论文4 复现 · 第3步：在真实数据集上训练线性链 CRF 并做消融

为什么这么做：
  附录模型的 BiLSTM 部分依赖 torch==1.7.0（requirement.txt:6），本机离线且无 torch，
  无法原样训练 BiLSTM。因此本脚本保留论文模型的「CRF 头」与「特征拼接」两条主线，
  把 BiLSTM 的发射分数替换为「字符上下文 + 实体词典特征」线性打分，
  从而在真实数据集上得到可复算、可比较的实体级 P/R/F1。

四个配置（对应论文第 4 章的四组实验）：
  A. base       : 字符上下文特征                       ≈ BiLSTM+CRF
  B. P(entity)  : A + 先验实体词典特征                 ≈ PBiLSTM+CRF（config.use_entity=True）
  C. Similar    : B + 近义实体特征(similar_word.txt)   ≈ Similar+PBiLSTM+CRF（config.use_similarity=True）
  D. word       : 词语级输入（TechnologyNer_word）     ≈ PBiLSTM+CRF(word)

产出：result_ly/results/p4/crf_ablation.{txt,json}
"""
import os
import time
import json
from collections import Counter

import numpy as np
from scipy.sparse import csr_matrix

ROOT = r"C:\Users\LiYi\Desktop\大创文件"
NER = os.path.join(ROOT, "_appx", "p4", "ner")
OUT = os.path.join(ROOT, "result_ly", "results", "p4")
os.makedirs(OUT, exist_ok=True)


# ============================== 数据 ==============================
def parse_bmes(path):
    sents, tags = [], []
    cur_w, cur_t = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.replace("\r", "").replace("\n", "")
            if line.strip() == "":
                if cur_w:
                    sents.append(cur_w); tags.append(cur_t); cur_w, cur_t = [], []
                continue
            p = line.split(" ")
            if len(p) < 2:
                continue
            cur_w.append("".join(p[:-1])); cur_t.append(p[-1])
    if cur_w:
        sents.append(cur_w); tags.append(cur_t)
    return sents, tags


def entity_spans(tags):
    spans, i, n = [], 0, len(tags)
    while i < n:
        t = tags[i]
        base = t[2:] if "-" in t else "UNTYPED"
        if t[:1] == "B":
            j = i + 1
            while j < n and tags[j] == ("I-" + base if "-" in t else "I"):
                j += 1
            if j < n and tags[j] == ("E-" + base if "-" in t else "E"):
                spans.append((base, i, j)); i = j + 1; continue
            spans.append((base, i, max(i, j - 1))); i = j
        elif t[:1] == "S":
            spans.append((base, i, i)); i += 1
        else:
            i += 1
    return spans


def micro_prf(gold_spans, pred_spans):
    tp = len(set(gold_spans) & set(pred_spans))
    fp = len(set(pred_spans) - set(gold_spans))
    fn = len(set(gold_spans) - set(pred_spans))
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f, tp, fp, fn


# ============================== CRF 核心 ==============================
def logsumexp(a, axis):
    m = np.max(a, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    return (m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))).squeeze(axis)


def crf_loss_grad(emission, A, gold, start, end):
    """loss = logZ - gold_score；返回 (loss, gE, gA)。"""
    L, C = emission.shape
    # ---- 前向 ----
    alpha = np.empty((L, C))
    alpha[0] = A[start, :C] + emission[0]
    for t in range(1, L):
        alpha[t] = logsumexp(alpha[t - 1][:, None] + A[:C, :C], 0) + emission[t]
    logZ = float(logsumexp(alpha[L - 1] + A[:C, end], 0))
    # ---- 后向 ----
    beta = np.empty((L, C))
    beta[L - 1] = A[:C, end]
    for t in range(L - 2, -1, -1):
        beta[t] = logsumexp(A[:C, :C] + (emission[t + 1] + beta[t + 1])[None, :], 1)
    # ---- gold ----
    gold_score = A[start, gold[0]] + emission[0, gold[0]]
    for t in range(1, L):
        gold_score += A[gold[t - 1], gold[t]] + emission[t, gold[t]]
    gold_score += A[gold[-1], end]
    # ---- 梯度 ----
    gE = np.exp(alpha + beta - logZ)
    gE[np.arange(L), gold] -= 1.0

    gA = np.zeros_like(A)
    gA[start, :C] += np.exp(A[start, :C] + emission[0] + beta[0] - logZ)
    gA[start, gold[0]] -= 1.0
    for t in range(1, L):
        xi = np.exp(alpha[t - 1][:, None] + A[:C, :C] + (emission[t] + beta[t])[None, :] - logZ)
        gA[:C, :C] += xi
        gA[gold[t - 1], gold[t]] -= 1.0
    gA[:C, end] += np.exp(alpha[L - 1] + A[:C, end] - logZ)
    gA[gold[-1], end] -= 1.0

    return float(logZ - gold_score), gE, gA


def crf_viterbi(emission, A, start, end):
    L, C = emission.shape
    vit = np.empty((L, C))
    bp = np.zeros((L, C), dtype=np.int64)
    vit[0] = A[start, :C] + emission[0]
    for t in range(1, L):
        cand = vit[t - 1][:, None] + A[:C, :C]
        bp[t] = np.argmax(cand, 0)
        vit[t] = cand.max(0) + emission[t]
    best = int(np.argmax(vit[L - 1] + A[:C, end]))
    path = [best]
    for t in range(L - 1, 0, -1):
        path.append(int(bp[t][path[-1]]))
    return list(reversed(path))


# ============================== 特征 ==============================
class Featurizer:
    def __init__(self, use_entity=False, use_similar=False):
        self.use_entity, self.use_similar = use_entity, use_similar
        self.f2i = {}
        self.entities = set()
        self.sim = {}
        self.maxlen = 1
        if use_entity or use_similar:
            with open(os.path.join(NER, "technology_entities_without_label.txt"), encoding="utf-8") as f:
                for line in f:
                    e = line.strip()
                    if e:
                        self.entities.add(e)
            self.maxlen = max(len(e) for e in self.entities)
            if use_similar:
                with open(os.path.join(NER, "similar_word.txt"), encoding="utf-8") as f:
                    for line in f:
                        line = line.replace("\n", "")
                        if "," in line:
                            a, b = line.split(",")[:2]
                            self.sim[a.strip()] = b.strip()

    def fid(self, feat):
        i = self.f2i.get(feat)
        if i is None:
            i = len(self.f2i); self.f2i[feat] = i
        return i

    def _match(self, words):
        n = len(words)
        cover = [None] * n
        i = 0
        while i < n:
            hit = None
            for L in range(min(self.maxlen, n - i), 1, -1):
                cand = "".join(words[i:i + L])
                if cand in self.entities:
                    hit = (cand, L); break
            if hit:
                e, L = hit
                for k in range(L):
                    cover[i + k] = (e, k, L)
                i += L
            else:
                i += 1
        return cover

    def sentence_feats(self, words):
        n = len(words)
        cover = self._match(words) if (self.use_entity or self.use_similar) else None
        out = []
        for i, w in enumerate(words):
            p = words[i - 1] if i > 0 else "<S>"
            q = words[i + 1] if i < n - 1 else "</S>"
            fs = [f"c={w}", f"p={p}", f"n={q}", f"b1={p}|{w}", f"b2={w}|{q}"]
            if self.use_entity:
                c = cover[i]
                if c is None:
                    fs.append("E=OUT")
                else:
                    e, k, L = c
                    fs += ["E=IN", f"EL={L}", f"EP={min(k, 4)}", f"EID={hash(e) % 4000}"]
            if self.use_similar:
                c = cover[i]
                if c is not None and c[0] in self.sim:
                    s = self.sim[c[0]]
                    fs += ["S=IN", f"SID={hash(s) % 4000}", f"SL={min(len(s), 8)}"]
                else:
                    fs.append("S=OUT")
            out.append(fs)
        return out


def register_features(feat, words_sents):
    """只注册特征、不建矩阵（保证 train/dev/test 共用同一特征空间）。"""
    for s in words_sents:
        for fs in feat.sentence_feats(s):
            for f in fs:
                feat.fid(f)


def build_matrices(feat, words_sents, D):
    """把每个句子转成 csr 稀疏特征矩阵（列数为固定的 D）。"""
    mats = []
    for s in words_sents:
        sf = feat.sentence_feats(s)
        rows, cols = [], []
        for j, fs in enumerate(sf):
            for f in fs:
                rows.append(j); cols.append(feat.f2i[f])
        L = len(sf)
        mats.append(csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(L, D)))
    return mats


# ============================== 训练 ==============================
def train(mats, golds, C, epochs=3, lr=0.05, batch=32, l2=1e-5, seed=3407, tag=""):
    rng = np.random.default_rng(seed)
    D = mats[0].shape[1]
    start, end = C, C + 1
    W = rng.normal(0, 0.01, size=(D, C))
    A = np.full((C + 2, C + 2), 1.0 / C)
    mW = np.zeros_like(W); vW = np.zeros_like(W)
    mA = np.zeros_like(A); vA = np.zeros_like(A)
    n = len(mats)
    order = np.arange(n)
    step = 0
    for ep in range(epochs):
        rng.shuffle(order)
        tot, tok = 0.0, 0
        t0 = time.time()
        for b0 in range(0, n, batch):
            idxs = order[b0:b0 + batch]
            gW = np.zeros_like(W); gA = np.zeros_like(A)
            bl, bt = 0.0, 0
            for i in idxs:
                X = mats[i]
                emission = np.asarray(X @ W)
                loss, gE, gAi = crf_loss_grad(emission, A, golds[i], start, end)
                bl += loss; bt += X.shape[0]
                gW += np.asarray(X.T @ gE)
                gA += gAi
            B = len(idxs)
            gW /= B; gA /= B
            gW += l2 * W
            step += 1
            for Pt, G, m, v in ((W, gW, mW, vW), (A, gA, mA, vA)):
                m *= 0.9; m += 0.1 * G
                v *= 0.999; v += 0.001 * G * G
                Pt -= lr * (m / (1 - 0.9 ** step)) / (np.sqrt(v / (1 - 0.999 ** step)) + 1e-8)
            tot += bl; tok += bt
        print(f"    [{tag}] epoch {ep+1}/{epochs} loss/token={tot/tok:.5f} "
              f"|V|={D} 用时 {time.time()-t0:.1f}s", flush=True)
    return W, A


def predict(W, A, mats, C, mode="viterbi"):
    out = []
    for X in mats:
        emission = np.asarray(X @ W)
        if mode == "viterbi":
            path = crf_viterbi(emission, A, C, C + 1)
        else:
            path = list(np.argmax(emission, 1))
        out.append(path)
    return out


def evaluate(pred_ids, gold_tags, tag_list):
    preds = [[tag_list[i] for i in p] for p in pred_ids]
    P, R, F, tp, fp, fn = micro_prf(
        [s for seq in gold_tags for s in entity_spans(seq)],
        [s for seq in preds for s in entity_spans(seq)])
    tok = sum(len(s) for s in gold_tags)
    ok = sum(1 for a, b in zip(preds, gold_tags) for x, y in zip(a, b) if x == y)
    return dict(Precision=P, Recall=R, F1=F, TP=tp, FP=fp, FN=fn, TokenAcc=ok / tok if tok else 0.0)


def main():
    t_all = time.time()
    S = {}
    for d, fn, k in [("TechnologyNer", "train.char.bmes", "ch_train"),
                     ("TechnologyNer", "dev.char.bmes", "ch_dev"),
                     ("TechnologyNer", "test.char.bmes", "ch_test"),
                     ("TechnologyNer_word", "train.char.bmes", "wd_train"),
                     ("TechnologyNer_word", "dev.char.bmes", "wd_dev"),
                     ("TechnologyNer_word", "test.char.bmes", "wd_test")]:
        S[k] = parse_bmes(os.path.join(NER, d, fn))

    results = {}

    def one(name, use_entity, use_similar, key, epochs=3):
        trw, trt = S[key + "_train"]
        dvw, dvt = S[key + "_dev"]
        tew, tet = S[key + "_test"]
        tag_list = sorted(set(t for seq in trt for t in seq))
        tag_list = ["O"] + [t for t in tag_list if t != "O"]
        tag2id = {t: i for i, t in enumerate(tag_list)}
        C = len(tag_list)
        print(f"\n==== {name} ====  (train={len(trw)} sents, labels={C})", flush=True)
        feat = Featurizer(use_entity=use_entity, use_similar=use_similar)
        # 关键：先让 train/dev/test 三个划分共同注册特征，固定特征空间维度
        register_features(feat, trw)
        register_features(feat, dvw)
        register_features(feat, tew)
        D = len(feat.f2i)
        tr_m = build_matrices(feat, trw, D)
        dv_m = build_matrices(feat, dvw, D)
        te_m = build_matrices(feat, tew, D)
        g_tr = [[tag2id[t] for t in tg] for tg in trt]
        t0 = time.time()
        W, A = train(tr_m, g_tr, C, epochs=epochs, lr=0.05, batch=32, seed=3407, tag=name)
        sec = time.time() - t0
        m_v = evaluate(predict(W, A, te_m, C, "viterbi"), tet, tag_list)
        m_g = evaluate(predict(W, A, te_m, C, "greedy"), tet, tag_list)
        d_v = evaluate(predict(W, A, dv_m, C, "viterbi"), dvt, tag_list)
        results[name] = dict(use_entity=use_entity, use_similar=use_similar, subset=key,
                             tag_count=C, feature_dim=D, train_seconds=round(sec, 1),
                             test_viterbi=m_v, test_greedy=m_g, dev_viterbi=d_v)
        print(f"    → test Viterbi P/R/F1 = {m_v['Precision']:.4f}/{m_v['Recall']:.4f}/{m_v['F1']:.4f}"
              f" | greedy F1 = {m_g['F1']:.4f} | dev F1 = {d_v['F1']:.4f} | {sec:.1f}s", flush=True)
        # 只保存转移矩阵（很小）；发射权重 W 体积达 40-80MB 且可重训，不入库
        np.save(os.path.join(OUT, f"A_{name}.npy"), A)
        return tag_list

    one("A_base", False, False, "ch")
    one("B_P_entity", True, False, "ch")
    one("C_Similar", True, True, "ch")
    one("D_P_entity_word", True, False, "wd")

    rows = []
    for k, v in results.items():
        mv = v["test_viterbi"]
        rows.append(dict(配置=k, Precision=round(mv["Precision"], 4), Recall=round(mv["Recall"], 4),
                         F1=round(mv["F1"], 4), TokenAcc=round(mv["TokenAcc"], 4),
                         贪心F1=round(v["test_greedy"]["F1"], 4),
                         devF1=round(v["dev_viterbi"]["F1"], 4),
                         训练秒=v["train_seconds"], 标签数=v["tag_count"], 特征维度=v["feature_dim"]))

    with open(os.path.join(OUT, "crf_ablation.json"), "w", encoding="utf-8") as f:
        json.dump(dict(results=results, elapsed_seconds=round(time.time() - t_all, 1),
                       env_note="离线环境：numpy 线性链 CRF 替代 BiLSTM 发射层；torch==1.7.0 不可用"),
                  f, ensure_ascii=False, indent=2)

    hdr = f"{'配置':<18}{'P':>8}{'R':>8}{'F1':>8}{'TokenAcc':>10}{'贪心F1':>9}{'devF1':>8}{'训练秒':>8}{'标签':>6}{'特征维度':>9}"
    lines = ["=" * 100,
             "论文4 复现 · 真实数据上的线性链 CRF 消融实验",
             "方法：BiLSTM 发射层因无 torch 不可复现，改用「字符上下文 / 实体词典」线性打分作为发射分数，",
             "      完整保留论文模型的 CRF 头（发射+转移、NLL、Viterbi）与特征拼接逻辑。",
             "评测：实体级 span+type 精确匹配的 micro P/R/F1；测试集 = TechnologyNer(_word)/test.char.bmes",
             "=" * 100, "", hdr, "-" * len(hdr)]
    for r in rows:
        lines.append(f"{r['配置']:<18}{r['Precision']:>8.4f}{r['Recall']:>8.4f}{r['F1']:>8.4f}"
                     f"{r['TokenAcc']:>10.4f}{r['贪心F1']:>9.4f}{r['devF1']:>8.4f}"
                     f"{r['训练秒']:>8.1f}{r['标签数']:>6}{r['特征维度']:>9}")
    b = results["A_base"]["test_viterbi"]; p = results["B_P_entity"]["test_viterbi"]
    s = results["C_Similar"]["test_viterbi"]; w = results["D_P_entity_word"]["test_viterbi"]
    lines += ["", "关键对比（本次复现实测）：",
              f"  B 先验实体  相对 A 基线     : ΔP={p['Precision']-b['Precision']:+.4f}  ΔR={p['Recall']-b['Recall']:+.4f}  ΔF1={p['F1']-b['F1']:+.4f}",
              f"  C +近义实体 相对 B 先验实体 : ΔP={s['Precision']-p['Precision']:+.4f}  ΔR={s['Recall']-p['Recall']:+.4f}  ΔF1={s['F1']-p['F1']:+.4f}",
              f"  D 词语级    相对 B 字符级   : ΔP={w['Precision']-p['Precision']:+.4f}  ΔR={w['Recall']-p['Recall']:+.4f}  ΔF1={w['F1']-p['F1']:+.4f}",
              "",
              "论文表 4.1 / 4.3 自报值（作为参照，非本次复现结果）：",
              "  BiLSTM+CRF(char)                 P=0.62 R=0.82 F1=0.71",
              "  PBiLSTM+CRF(char)                P=0.66 R=0.84 F1=0.74",
              "  Similar(GPT)+PBiLSTM+CRF(char)   P=0.65 R=0.84 F1=0.73",
              "  PBiLSTM+CRF(word)                P=0.12 R=0.10 F1=0.11",
              f"",
              f"全部实验总耗时 {time.time()-t_all:.1f}s"]
    txt = "\n".join(lines)
    with open(os.path.join(OUT, "crf_ablation.txt"), "w", encoding="utf-8") as f:
        f.write(txt)
    print("\n" + txt)


if __name__ == "__main__":
    main()

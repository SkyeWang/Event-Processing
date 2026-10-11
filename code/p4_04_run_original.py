# -*- coding: utf-8 -*-
"""
论文4 复现 · 第4步：直接运行附录原始代码（需要 torch）

不修改附录里的任何文件。通过运行时 patch 控制配置，得到论文表 4.1 / 4.2 / 4.3 各行的对应结果：
  A  BiLSTM+CRF(char)                  use_entity=False
  B  PBiLSTM+CRF(char)                  use_entity=True（附录 main_entity.py 的默认配置）
  B2 PBiLSTM+CRF(char) 修正版           use_entity=True + 修复 entity_pos 未填充的 bug
  C  Similar+PBiLSTM+CRF(char)          use_entity=True + use_similarity=True
  D  PBiLSTM+CRF(word)                  use_entity=True + word_level=True

用法：
  python p4_04_run_original.py --config B --epochs 20
"""
import os
import sys
import re
import ast
import json
import time
import argparse
import difflib

ROOT = r"C:\Users\LiYi\Desktop\大创文件"
NER = os.path.join(ROOT, "_appx", "p4", "ner")
OUT = os.path.join(ROOT, "result_ly", "results", "p4")
os.makedirs(OUT, exist_ok=True)

CFG = {
    "A": dict(subset="TechnologyNer", use_entity=False, use_similarity=False, word_level=False, entity_fix=False,
              desc="BiLSTM+CRF(char)"),
    "B": dict(subset="TechnologyNer", use_entity=True, use_similarity=False, word_level=False, entity_fix=False,
              desc="PBiLSTM+CRF(char) 原样"),
    "B2": dict(subset="TechnologyNer", use_entity=True, use_similarity=False, word_level=False, entity_fix=True,
               desc="PBiLSTM+CRF(char) 修复 entity_pos"),
    "C": dict(subset="TechnologyNer", use_entity=True, use_similarity=True, word_level=False, entity_fix=False,
              desc="Similar+PBiLSTM+CRF(char) 原样"),
    "C2": dict(subset="TechnologyNer", use_entity=True, use_similarity=True, word_level=False, entity_fix=True,
               desc="Similar+PBiLSTM+CRF(char) 修复 entity_pos"),
    "D": dict(subset="TechnologyNer_word", use_entity=True, use_similarity=False, word_level=True, entity_fix=False,
              desc="PBiLSTM+CRF(word) 原样"),
    "D2": dict(subset="TechnologyNer_word", use_entity=True, use_similarity=False, word_level=True, entity_fix=True,
               desc="PBiLSTM+CRF(word) 修复 entity_pos"),
}

FIX_LINE = "entity_pos.update(range(item_entity_pos[0], item_entity_pos[1]))"


def build_patched_methods():
    """从 models/bilstm_crf.py 源码里取出构建先验实体张量的四个方法
    （train_step / validate / test / predict），只在 entity_pos 的处理处插入一行修复，
    并打印 unified diff 供报告引用。"""
    src = open(os.path.join(NER, "models", "bilstm_crf.py"), encoding="utf-8").read()
    tree = ast.parse(src)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "BILSTM_Model")
    targets = ("train_step", "validate", "test", "predict")
    msrc = {}
    for n in cls.body:
        if isinstance(n, ast.FunctionDef) and n.name in targets:
            seg = ast.get_source_segment(src, n)
            if "for pos_index in range(" in seg:
                msrc[n.name] = seg

    patched, diffs = {}, []
    for name, s in msrc.items():
        lines = s.split("\n")
        out = []
        for ln in lines:
            out.append(ln)
            m = re.match(r"^(\s*)for item_entity_pos in item_entity\[1\]:", ln)
            if m:
                # 在 `for item_entity_pos ...` 之后、`for pos_index ...` 之前插入一行，
                # 与本循环体同级，这样 entity_pos 才会被填满；插在别处会破坏缩进结构。
                out.append(m.group(1) + "    " + FIX_LINE)
        new = "\n".join(out)
        patched[name] = new
        diffs.append("\n".join(difflib.unified_diff(
            s.split("\n"), new.split("\n"),
            fromfile=f"附录原版 {name}", tofile=f"修复版 {name}", lineterm="", n=2)))
    return patched, "\n\n".join(diffs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=list(CFG))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--chunk", type=int, default=64, help="测试时分块大小，避免 CRF 打分张量爆内存")
    ap.add_argument("--limit", type=int, default=0, help="仅用前 N 条训练样本（冒烟测试用）")
    ap.add_argument("--probe", type=int, default=256, help="每轮在 dev 前 N 条上探一次实体级 F1（0=关闭）")
    ap.add_argument("--cache-similar", action="store_true",
                    help="对 util.get_similarity_from_file 做结果一致的记忆化（原实现每次都重新打开并线性扫描 8435 行文件）")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    cfg = CFG[args.config]
    os.chdir(NER)
    sys.path.insert(0, NER)

    # ---------------- 配置注入（必须在 import 模型之前/之后均可，因为都是运行时读取） ----------------
    from models.config import TrainingConfig, LSTMConfig
    TrainingConfig.epoches = args.epochs
    LSTMConfig.use_entity = cfg["use_entity"]
    LSTMConfig.use_similarity = cfg["use_similarity"]
    LSTMConfig.word_level = cfg["word_level"]
    LSTMConfig.gpt_embedding = False
    LSTMConfig.entity_gpt = False
    if not hasattr(LSTMConfig, "simi_rate"):
        LSTMConfig.simi_rate = 0.5          # config.py 里该属性只在 use_similarity=True 时才定义

    from data import build_corpus_technology
    from utils import extend_maps, prepocess_data_for_lstmcrf
    from models.bilstm_crf import BILSTM_Model
    from BIOES_eval import classification_report
    import torch

    torch.manual_seed(3407)

    log = []
    def P(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        log.append(s)

    P("=" * 88)
    P(f"论文4 复现 · 直接运行附录原始代码   配置 {args.config} = {cfg['desc']}")
    P(f"torch {torch.__version__}  device=cpu  epochs={args.epochs}  数据={cfg['subset']}")
    P(f"use_entity={cfg['use_entity']}  use_similarity={cfg['use_similarity']}  "
      f"word_level={cfg['word_level']}  entity_fix={cfg['entity_fix']}")
    P("=" * 88)

    # ---------------- 数据 ----------------
    d = "./" + cfg["subset"]
    P("\n[1] 读取数据 ...")
    train_w, train_t, word2id, tag2id = build_corpus_technology("train.char.bmes", data_dir=d)
    dev_w, dev_t = build_corpus_technology("dev.char.bmes", make_vocab=False, data_dir=d)
    test_w, test_t = build_corpus_technology("test.char.bmes", make_vocab=False, data_dir=d)
    if args.limit:
        train_w, train_t = train_w[:args.limit], train_t[:args.limit]
    P(f"    train={len(train_w)}  dev={len(dev_w)}  test={len(test_w)}  "
      f"词表={len(word2id)}  标签={len(tag2id)}")

    P("\n[2] 扩展词表 + CRF 预处理 ...")
    crf_word2id, crf_tag2id = extend_maps(word2id, tag2id, for_crf=True)
    train_w, train_t = prepocess_data_for_lstmcrf(train_w, train_t)
    dev_w, dev_t = prepocess_data_for_lstmcrf(dev_w, dev_t)
    test_w, test_t = prepocess_data_for_lstmcrf(test_w, test_t, test=True)
    P(f"    CRF 词表={len(crf_word2id)}  CRF 标签={len(crf_tag2id)}")

    entities = []
    if cfg["use_entity"]:
        with open("technology_entities_without_label.txt", encoding="utf-8") as f:
            entities = list(set(x.strip() for x in f if x.strip()))
        P(f"    先验实体数={len(entities)}")

    # ---------------- 可选修复 ----------------
    if cfg["entity_fix"]:
        patched, diff = build_patched_methods()
        P("\n[3] 应用 entity_pos 修复补丁（唯一改动：补上 entity_pos 的填充）")
        P(diff)
        with open(os.path.join(OUT, "entity_fix.diff"), "w", encoding="utf-8") as f:
            f.write(diff)
        # 必须在目标模块自己的命名空间里 exec，否则拿不到 tensorized / util / LSTMConfig 等全局量
        import models.bilstm_crf as MC
        g = MC.__dict__
        for nm, s in patched.items():
            exec(compile(s, f"<patched {nm}>", "exec"), g)
            setattr(MC.BILSTM_Model, nm, g[nm])
        P(f"    补丁已注入（{'、'.join(patched)}）")
    else:
        P("\n[3] 不应用补丁（原样运行）")

    # ---------------- 可选：相似度查表记忆化（结果一致，只是不再反复读文件） ----------------
    if args.cache_similar:
        import models.util as U
        _raw_sim = U.get_similarity_from_file
        _sim_cache = {}
        _miss = [0]

        def cached_sim(name):
            if name not in _sim_cache:
                _sim_cache[name] = _raw_sim(name)
                _miss[0] += 1
            return _sim_cache[name]

        U.get_similarity_from_file = cached_sim
        import time as _t
        _k = "物联网企业"
        t0 = _t.time()
        for _ in range(200):
            cached_sim(_k)
        t_cached = (_t.time() - t0) / 200
        t0 = _t.time()
        for _ in range(20):
            _raw_sim(_k)
        t_raw = (_t.time() - t0) / 20
        P(f"\n[3b] 相似度查表记忆化已启用：原实现单次 {t_raw*1000:.2f} ms，"
          f"缓存后 {t_cached*1000:.4f} ms，加速 {t_raw/max(t_cached,1e-9):.0f}×")

    # ---------------- 训练 ----------------
    P("\n[4] 训练 ...")
    # 探针必须用 test 划分：dev 在 prepocess_data_for_lstmcrf 中未按 test=True 处理，
    # gold 末尾多一个 <end>，而 BIOES_eval.get_entities 会把整个集合拍平成一个长串
    # 再按绝对位置匹配，逐句错位会让 F1 严重失真（实测：pred 与 gold 逐字相同时 F1 也只有 0.16）。
    probe_w, probe_t = test_w[:args.probe], test_t[:args.probe]
    curve = []
    if args.probe:
        _orig_validate = BILSTM_Model.validate

        def validate_with_probe(self, dw, dt, w2id, t2id):
            loss = _orig_validate(self, dw, dt, w2id, t2id)
            try:
                pp, gg = [], []
                for i in range(0, len(probe_w), 32):
                    a, b = self.test_with_btach(probe_w[i:i + 32], probe_t[i:i + 32], w2id, t2id)
                    pp.extend(a); gg.extend(b)
                _, mv, _ = classification_report(gg, pp)
                f1 = mv["f1"] if isinstance(mv, dict) else 0.0
                curve.append(dict(epoch=len(curve) + 1, val_loss=loss, test_probe_f1=f1))
                P(f"    [probe] epoch {len(curve)}  test前{len(probe_w)}条 F1={f1:.4f}  val_loss={loss:.4f}")
            except Exception as e:
                P(f"    [probe] 失败：{e}")
            return loss

        BILSTM_Model.validate = validate_with_probe
        P(f"    已注入逐轮探针（test 前 {len(probe_w)} 条，口径与最终评测一致）")

    model = BILSTM_Model(len(crf_word2id), len(crf_tag2id), entities, crf=True)
    t0 = time.time()
    model.train(train_w, train_t, dev_w, dev_t, crf_word2id, crf_tag2id)
    train_sec = time.time() - t0
    P(f"    训练完成，用时 {train_sec:.1f}s")

    # ---------------- 分块测试 ----------------
    P(f"\n[5] 分块测试（chunk={args.chunk}）...")
    preds, golds = [], []
    for i in range(0, len(test_w), args.chunk):
        cw = test_w[i:i + args.chunk]
        ct = test_t[i:i + args.chunk]
        p, g = model.test_with_btach(cw, ct, crf_word2id, crf_tag2id)
        preds.extend(p)
        golds.extend(g)
    P(f"    预测条数={len(preds)}")

    report, micro_avg, details = classification_report(golds, preds)
    P("\n[6] 实体级（span）评测结果：")
    P(report)
    P(str(micro_avg))

    res = dict(config=args.config, desc=cfg["desc"], epochs=args.epochs,
               use_entity=cfg["use_entity"], use_similarity=cfg["use_similarity"],
               word_level=cfg["word_level"], entity_fix=cfg["entity_fix"],
               subset=cfg["subset"], train_size=len(train_w), test_size=len(test_w),
               tag_count=len(crf_tag2id), vocab=len(crf_word2id),
               train_seconds=round(train_sec, 1), curve=curve,
               micro=dict(micro_avg) if isinstance(micro_avg, dict) else micro_avg,
               report=report, torch=torch.__version__)
    nm = args.tag or args.config
    with open(os.path.join(OUT, f"orig_{nm}.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    with open(os.path.join(OUT, f"orig_{nm}.log"), "w", encoding="utf-8") as f:
        f.write("\n".join(log))
    P(f"\n[OK] {os.path.join(OUT, f'orig_{nm}.json')}")


if __name__ == "__main__":
    main()

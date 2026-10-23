"""生成 24 条合成知识抽取样本，用于跑通 LoRA + EWC + 回放流程。

附录没有随带论文训练用的 JSONL（原文件在作者的 AutoDL 机器上），
因此本脚本手写 24 条中英文科技新闻样本，格式与原项目一致：
``messages = [system, user, assistant]``，assistant 是 JSON 字符串，
包含 year、knowledge_type、triples。

输出目录：
  data/train_data/ch/batch_1/{train,val}.jsonl
  data/train_data/ch/batch_2/{train,val}.jsonl
  data/train_data/en/batch_1/{train,val}.jsonl

每批 6 条 train / 2 条 val。样本只用于验证训练与接口链路，
不作为论文数据或指标依据。
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "train_data")

SYSTEM_CH = (
    "从科技新闻中抽取知识图谱三元组。输出JSON格式，包含year、knowledge_type和triples数组。"
    "每个triple包含entity1、relation、entity2、relation_type、hierarchy_level。"
    "关系必须从预定义集合中选择：发布、提出、发明、创建、研发、开发、设计、制造、收购、投资、合作、"
    "应用、超越、导致、取代、融合、升级、基于、是、属于、包含、举办、担任、攻克、验证、探讨。"
    "只输出JSON，不要解释。"
)
SYSTEM_EN = (
    "Extract knowledge graph triples from tech news. Output JSON with year, knowledge_type, "
    "and triples array. Each triple contains entity1, relation, entity2, relation_type, "
    "hierarchy_level. Relations must be from: release, propose, invent, create, develop, design, "
    "manufacture, acquire, invest, cooperate, apply, surpass, cause, replace, integrate, upgrade, "
    "based_on, is, belong_to, contain, host, serve_as, overcome, verify, discuss, has_feature. "
    "Output JSON only, no explanation."
)


def triple(e1, rel, e2, rt="causal", level="1"):
    """构造一条三元组 dict，字段与训练 JSONL 的 assistant 输出一致。"""
    return {"entity1": e1, "relation": rel, "entity2": e2, "relation_type": rt, "hierarchy_level": level}


CH_BATCH1 = [
    ("2025年3月，中芯国际宣布实现5纳米芯片量产，该工艺基于FinFET技术。", 2025, "dynamic",
     [triple("中芯国际", "实现", "5纳米芯片"), triple("5纳米芯片", "基于", "FinFET技术", "hierarchical", "2")]),
    ("宁德时代与特斯拉签署合作协议，将在上海建设储能电池工厂。", 2025, "dynamic",
     [triple("宁德时代", "合作", "特斯拉"), triple("宁德时代", "建设", "储能电池工厂", "causal", "2")]),
    ("华为发布昇腾910C人工智能芯片，性能较上一代提升30%。", 2025, "dynamic",
     [triple("华为", "发布", "昇腾910C"), triple("昇腾910C", "升级", "人工智能芯片", "hierarchical", "2")]),
    ("比亚迪收购一家锂矿企业，以保障电池原材料供应。", 2024, "dynamic",
     [triple("比亚迪", "收购", "锂矿企业")]),
    ("2025年1月，国家电网启动特高压输电工程建设，总投资超过200亿元。", 2025, "dynamic",
     [triple("国家电网", "创建", "特高压输电工程"), triple("国家电网", "投资", "200亿元", "causal", "2")]),
    ("隆基绿能研发的钙钛矿太阳能电池转换效率达到33.9%。", 2024, "static",
     [triple("隆基绿能", "研发", "钙钛矿太阳能电池"), triple("钙钛矿太阳能电池", "是", "高效光伏器件", "hierarchical", "2")]),
    ("上海电气与西门子能源合作开发海上风电机组。", 2024, "dynamic",
     [triple("上海电气", "合作", "西门子能源"), triple("上海电气", "开发", "海上风电机组", "causal", "2")]),
    ("2024年12月，工信部发布集成电路产业扶持政策。", 2024, "dynamic",
     [triple("工信部", "发布", "集成电路产业扶持政策")]),
]

CH_BATCH2 = [
    ("2025年2月，脑虎科技完成数亿元融资，用于柔性脑机接口研发。", 2025, "dynamic",
     [triple("脑虎科技", "投资", "数亿元融资"), triple("脑虎科技", "研发", "柔性脑机接口", "causal", "2")]),
    ("Neuralink为第二位患者植入脑机接口芯片，设备运行稳定。", 2025, "dynamic",
     [triple("Neuralink", "应用", "脑机接口芯片"), triple("Neuralink", "是", "脑机接口公司", "hierarchical", "2")]),
    ("宇树科技发布人形机器人H1，具备奔跑与跳跃能力。", 2025, "dynamic",
     [triple("宇树科技", "发布", "人形机器人H1"), triple("人形机器人H1", "包含", "奔跑能力", "hierarchical", "2")]),
    ("优必选与比亚迪合作，将人形机器人引入汽车生产线。", 2025, "dynamic",
     [triple("优必选", "合作", "比亚迪"), triple("人形机器人", "应用", "汽车生产线", "causal", "2")]),
    ("中国科学技术大学实现千公里级量子密钥分发。", 2024, "static",
     [triple("中国科学技术大学", "实现", "量子密钥分发"), triple("量子密钥分发", "属于", "量子通信", "hierarchical", "2")]),
    ("本源量子发布72比特超导量子计算机。", 2024, "dynamic",
     [triple("本源量子", "发布", "72比特超导量子计算机")]),
    ("2025年4月，北京人形机器人创新中心成立。", 2025, "dynamic",
     [triple("北京人形机器人创新中心", "创建", "创新中心")]),
    ("微软投资Figure公司，推动人形机器人商业化。", 2025, "dynamic",
     [triple("微软", "投资", "Figure公司"), triple("Figure公司", "研发", "人形机器人", "causal", "2")]),
]

EN_BATCH1 = [
    ("In March 2025, OpenAI released GPT-4.5, a multimodal model with improved reasoning.", 2025, "dynamic",
     [triple("OpenAI", "release", "GPT-4.5"), triple("GPT-4.5", "is", "multimodal model", "hierarchical", "2")]),
    ("NVIDIA unveiled the Blackwell B200 GPU for AI training.", 2025, "dynamic",
     [triple("NVIDIA", "release", "Blackwell B200 GPU")]),
    ("TSMC will build a 2nm fabrication plant in Arizona.", 2025, "dynamic",
     [triple("TSMC", "create", "2nm fabrication plant")]),
    ("Microsoft and OpenAI expanded their cloud computing partnership.", 2025, "dynamic",
     [triple("Microsoft", "cooperate", "OpenAI")]),
    ("Samsung acquired a semiconductor packaging company to expand its AI chip business.", 2024, "dynamic",
     [triple("Samsung", "acquire", "semiconductor packaging company")]),
    ("Google DeepMind introduced AlphaFold 3 for protein structure prediction.", 2024, "static",
     [triple("Google DeepMind", "release", "AlphaFold 3"), triple("AlphaFold 3", "apply", "protein structure prediction", "causal", "2")]),
    ("Tesla plans to produce humanoid robots at scale in 2026.", 2025, "dynamic",
     [triple("Tesla", "develop", "humanoid robots")]),
    ("The U.S. Department of Commerce announced new export controls on advanced chips.", 2025, "dynamic",
     [triple("U.S. Department of Commerce", "release", "export controls")]),
]


def write_batch(items, lang, batch):
    """把一批样本写成 train/val 两个 JSONL。

    Args:
        items: ``(content, year, knowledge_type, triples)`` 列表。
        lang: ``ch`` 或 ``en``，决定用中文还是英文 system prompt。
        batch: 批次名，如 ``batch_1``，决定输出子目录。
    """
    system = SYSTEM_CH if lang == "ch" else SYSTEM_EN
    rows = []
    for content, year, kt, triples in items:
        # assistant 内容统一转成 JSON 字符串，与原始数据格式一致
        output = json.dumps(
            {"year": str(year), "knowledge_type": kt, "triples": triples},
            ensure_ascii=False,
        )
        rows.append({"messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
            {"role": "assistant", "content": output},
        ]})
    folder = os.path.join(OUT, lang, batch)
    os.makedirs(folder, exist_ok=True)
    # 固定前 6 条做 train、后 2 条做 val，保证每次生成结果可复现
    for name, part in (("train.jsonl", rows[:6]), ("val.jsonl", rows[6:])):
        with open(os.path.join(folder, name), "w", encoding="utf-8") as fh:
            for row in part:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(lang, batch, len(rows[:6]), len(rows[6:]))


if __name__ == "__main__":
    write_batch(CH_BATCH1, "ch", "batch_1")
    write_batch(CH_BATCH2, "ch", "batch_2")
    write_batch(EN_BATCH1, "en", "batch_1")

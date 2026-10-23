from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_PATH = "/root/autodl-tmp/fjy/models/Qwen/Qwen2.5-7B-Instruct"
LORA_PATH = "/root/autodl-tmp/fjy/output/kg_extractor_lora_v2/lora_latest"
MERGED_PATH = "/root/autodl-tmp/fjy/output/kg_extractor_merged_v2"

SYSTEM_PROMPT_CH = "从科技新闻中抽取知识图谱三元组。输出JSON格式，包含year、knowledge_type和triples数组。每个triple包含entity1、relation、entity2、relation_type、hierarchy_level。只输出JSON，不要解释。"
SYSTEM_PROMPT_EN = "Extract knowledge graph triples from tech news. Output JSON with year, knowledge_type, and triples array. Each triple contains entity1, relation, entity2, relation_type, hierarchy_level. Output JSON only, no explanation."

print("加载基础模型...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

print("合并LoRA权重...")
model = PeftModel.from_pretrained(base_model, LORA_PATH)
model = model.merge_and_unload()

print(f"保存合并模型到 {MERGED_PATH}...")
model.save_pretrained(MERGED_PATH)
tokenizer.save_pretrained(MERGED_PATH)
print("保存完成！")

test_cases = [
    (SYSTEM_PROMPT_CH, "2024年3月，OpenAI与Figure公司合作推出了Figure 01人形机器人。"),
    (SYSTEM_PROMPT_CH, "脑机接口技术为失语者重建沟通的桥梁，医疗康复成为当前脑机接口技术应用最成熟的领域。"),
    (SYSTEM_PROMPT_CH, "华为发布了全新的昇腾910B芯片，用于大模型训练加速。"),
    (SYSTEM_PROMPT_EN, "Google DeepMind released Gemini 2.0, a multimodal AI model that outperforms GPT-4 on several benchmarks."),
    (SYSTEM_PROMPT_EN, "Neuralink successfully implanted its brain-computer interface device in a second human patient in 2025."),
]

print("\n" + "=" * 50)
print("测试推理 (中文+英文)")
print("=" * 50)

for sys_prompt, text in test_cases:
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": text},
    ]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.1, do_sample=True)
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"\n输入: {text}")
    print(f"输出: {response}")
    print("-" * 50)

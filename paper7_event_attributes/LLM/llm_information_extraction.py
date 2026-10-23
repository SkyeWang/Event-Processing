# 论文七：作者原始提示词、语言分类示例、属性 schema 和回复解析。
# 中文注释为提交整理时补充，实验算法沿用本机已有版本。
import re
import json

from rich import print
from rich.console import Console
from transformers import AutoTokenizer, AutoModel


# 分类
class_examples = {
    '中文': '2018年，美国在对中国发动贸易战的同时，还制裁了中国的中兴通讯，这被认为是美国发动科技战的开端。',
    '英文': 'An employee showcases a semiconductor integrated circuit at an Industry Expo on October 31.'
    # ......
}
class_list = list(class_examples.keys())

CLS_PATTERN = f"“{{}}”是 {class_list} 里的什么类别？"


# 定义不同实体下的属性
schema = {
    '中文': ['实体', '关系', '时间', '国家'],
    '英文': ['实体', '关系', '形容词', '副词', '时间', '国家']
}

IE_PATTERN = "{}\n\n提取上述句子中{}类型的实体，并按照JSON格式输出，上述句子中不存在的信息用['原文中未提及']来表示，多个值之间用','分隔。"


# 属性抽取
ie_examples = {
        '中文': [
            {
               'content': '2018年，美国在对中国发动贸易战的同时，还制裁了中国的中兴通讯，这被认为是美国发动科技战的开端。',
               'answers': {
                   '实体': ['中兴通讯', '科技战', '开端'], 
                   '关系': ['制裁'], 
                   '时间': ['2018年'], 
                   '国家': ['美国', '中国']
               }
           },
            # ......
        ],
        '英文': [
            {
                'content': 'AI will provide the computing power of the future, while quantum communication will be a new tool.',
                'answers': {
                    '实体': ['AI', 'computing power', 'quantum communication', 'tool'],
                    '关系': ['provide'],
                    '形容词': ['computing', 'new'],
                    '副词': ['原文中未提及'],
                    '时间': ['原文中未提及'],
                    '国家': ['原文中未提及']
                }
            },
            # ......
        ],
}


# 把分类与属性示例组织为固定对话历史；每个待预测句子均复用这些 few-shot。
def init_prompts():
    class_list = list(class_examples.keys())
    cls_pre_history = [
        (
            f'现在你是一个文本分类器，你需要按照要求将我给你的句子分类到：{class_list}类别中。',
            f'好的。'
        )
    ]

    for _type, example in class_examples.items():
        cls_pre_history.append((f'“{example}”是 {class_list} 里的什么类别？', _type))

    ie_pre_history = [
        (
            "现在你需要帮助我完成信息抽取任务，当我给你一个句子时，你需要帮我抽取出句子中三元组，并按照JSON的格式输出，上述句子中没有的信息用['原文中未提及']来表示，多个值之间用','分隔。",
            '好的，请输入您的句子。'
        )
    ]

    for _type, example_list in ie_examples.items():
        for example in example_list:
            sentence = example['content']
            properties_str = ', '.join(schema[_type])
            schema_str_list = f'“{_type}”({properties_str})'
            sentence_with_prompt = IE_PATTERN.format(sentence, schema_str_list)
            ie_pre_history.append((
                f'{sentence_with_prompt}',
                f"{json.dumps(example['answers'], ensure_ascii=False)}"
            ))

    return {'ie_pre_history': ie_pre_history, 'cls_pre_history': cls_pre_history}


# 按作者规则清理模型回复并尝试解析 JSON；失败时保留文本，便于核查。
def clean_response(response: str):
    if '```json' in response:
        res = re.findall(r'```json(.*?)```', response)
        if len(res) and res[0]:
            response = res[0]
        response.replace('、', ',')
    try:
        return json.loads(response)
    except:
        return response


# 每句先调用模型判断语言，再根据该语言对应的字段进行第二次抽取。
def inference(
        sentences: list,
        custom_settings: dict
    ):
    
    for sentence in sentences:
        with console.status("[bold bright_green] Model Inference..."):
            sentence_with_cls_prompt = CLS_PATTERN.format(sentence)
            cls_res, _ = model.chat(tokenizer, sentence_with_cls_prompt, history=custom_settings['cls_pre_history'])

            if cls_res not in schema:
                print(f'The type model inferenced {cls_res} which is not in schema dict, exited.')
                exit()

            properties_str = ', '.join(schema[cls_res])
            schema_str_list = f'“{cls_res}”({properties_str})'
            sentence_with_ie_prompt = IE_PATTERN.format(sentence, schema_str_list)
            ie_res, _ = model.chat(tokenizer, sentence_with_ie_prompt, history=custom_settings['ie_pre_history'])
            ie_res = clean_response(ie_res)
        print(f'>>> [bold bright_red]sentence: {sentence}')
        print(f'>>> [bold bright_green]inference answer: ')
        print(ie_res)


if __name__ == '__main__':
    console = Console()
    
    # GPU调用
    device = 'cuda:0'
    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
    model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).quantize(4).half().cuda()
    model.to(device)
    
    # CPU调用
    # tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b-int8", trust_remote_code=True)
    # model = AutoModel.from_pretrained("THUDM/chatglm-6b-int8", trust_remote_code=True).float()
    # model = model.eval()

    sentences = [
        # '此外，近年来美国对于基础研究投资不足，而基础研究正是创新的种子。',
        '2022年8月9日，拜登政府签署了投资2800亿美元的《芯片与科学法案》，其主要意图是要提升美国在芯片产业链的自给自足率，促进制造业本土化，减少对海外供应链的依赖，加强美国经济和国家安全。',
        '2022年10月7日，美国商务部工业与安全局发布《出口管制条例》的最新修订规则，新增限制主要针对的是在中国开发和生产的先进计算集成电路、任何类型的半导体制造设备、超级计算机。',
        # '近日，有媒体爆料，希捷科技在2020年美国制裁规则生效后约一年内向华为出货了 740 万块硬盘，成为华为公司的唯一硬盘供应商，因此被罚款 3 亿美元。',
        # '其实，对中国半导体产业的制裁，已经宣布了中美科技战的进一步加剧，甚至演变为一场全方位科技战。特别是最近很火的大模型人工智能领域掀起的算力革命，也成为了中美竞争的新指标。',
        # '美国对华发起的贸易战已经持续5年了，其主轴从特朗普政府时期的贸易战演进为拜登政府时期的科技战。',
        # '去年10月12日，拜登政府发布任内首份《国家安全战略》报告，48页的报告提及“技术”87次，位居关键词榜首，这表明华盛顿已将尖端科技创新与争夺STEM人才置于赢得国家竞争的重要地位。',
        # '上世纪80年代，美国力保其芯片领先地位，为隐形战机和其他武器提供动力，给苏联造成不可承受的成本；后又以“国家安全”“非法倾销”为由，通过严厉的关税制裁遏制日本芯片产业的发展势头。',
        # '华盛顿这种逆全球化操作必将引发欧洲、日本、韩国等全球半导体优势国家与地区的警觉，导致它们都希望通过税收减免、研究合作、工人培训补贴等支持手段，继续维护本土芯片产业链实力。'
        # 'a robot performs a massage for a visitor at the 2021 world artificial intelligence conference',
        # 'photo: vcgtop officials from china’s ministry of industry and information technology (miit) said the country will promote new emerging industry clusters, including the layout of industry chains in cutting-edge sectors such as artificial intelligence (ai) and quantum communication, to ensure the resilience of the industry and supply chains of the world’s second-largest economy',
        # 'tian yulong, chief engineer with the miit, said that the country will speed up efforts to create industry chains in forward-looking sectors such as ai and quantum communication',
        # 'such an approach will be coupled with an upgrading in more traditional sectors including engineering equipment, nonferrous metals, construction materials and petrochemicals to consolidate china’s lead among the world’s industrial powers, along with efforts to scale up china’s advantage in emerging sectors including bullet trains, electric equipment, new energy and telecommunications, tian told a state council information office press conference on monday',
        # 'china enjoys leading status in ai and quantum communication technologies, chinese analysts and industry insiders said, but there is urgency to beef up technology self-reliance to boost supply chain security, given the looming risk of a technology sector decoupling from the us',
        # 'chen chen, a beijing-based expert in smart manufacturing, said that ai and quantum communication will be common aspects of the next generation of information technology',
        # 'ai will provide the computing power of the future, while quantum communication will be a new tool',
        # 'china is actually at the forefront of the world in ai, running almost neck-and-neck with the us, liu gang, director of the nankai institute of economics and chief economist at the chinese institute of new generation artificial intelligence development strategies, told the global times on monday',
        # 'liu said that the number of ai enterprises in china is second only to the us, and the numbers of patents and published papers have already surpassed those of the us',
        # ......
    ]

    custom_settings = init_prompts()
    inference(
        sentences,
        custom_settings
    )

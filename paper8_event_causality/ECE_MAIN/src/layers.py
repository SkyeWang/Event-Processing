# 论文八：事件类型作为条件的层归一化，沿用作者的标准差计算公式。
# 中文注释为提交整理时补充，实验算法沿用本机已有版本。
from utils import *


class ConditionalLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super(ConditionalLayerNorm, self).__init__()
        self.eps = eps
        self.gamma_dense = nn.Linear(hidden_size, hidden_size, bias=False)
        self.beta_dense = nn.Linear(hidden_size, hidden_size, bias=False)
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))

        nn.init.zeros_(self.gamma_dense.weight)
        nn.init.zeros_(self.beta_dense.weight)

    def forward(self, x, condition):
        '''
        :param x: [b, t, e]
        :param condition: [b, e]
        :return:
        '''
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        # 将类型向量广播到各文本位置，生成条件缩放 gamma 和偏置 beta。
        condition = condition.unsqueeze(1).expand_as(x)
        gamma = self.gamma_dense(condition) + self.gamma
        beta = self.beta_dense(condition) + self.beta
        x = gamma * (x - mean) / (std + self.eps) + beta
        return x

"""Trusted Python method registry. New algorithms do not change the ledger."""
from dataclasses import dataclass
from typing import Callable


@dataclass
class Method:
    key: str
    title: str
    paper: str
    pages: str
    description: str
    steps: list
    trainer: Callable | None = None
    frequency: str = 'any'
    validator: Callable | None = None
    predictor_factory: Callable | None = None


METHODS={}


def register(method):
    if method.key in METHODS:raise ValueError('重复方法名称')
    METHODS[method.key]=method


for key,title in [('hybrid','CNN + LSTM 混合网络'),('cnn','单 CNN 基线'),('lstm','单 LSTM 基线')]:
    register(Method(key,title,'孙宏鑫、魏先华 · 2021','PDF 4–9页',
        '历史原始价格与局部趋势并行编码，投影相加后预测方向和持续时间。',
        ['数据检查','前缀趋势分段','双路样本','卷积观察','长短期记忆','特征融合','评估与发布']))
register(Method('stgp','随机多目标遗传规划','龚雅键等 · 2020','PDF 4–11页',
    '带类型的表达式树，在按代变化的随机多目标适应度下进化，并滚动冻结预测。',
    ['日频数据','30因子构建','强类型种群','随机目标权重','交叉与变异','滚动样本外预测','评估与发布'],frequency='1d'))


def info():
    return [dict(key=m.key,title=m.title,paper=m.paper,pages=m.pages,description=m.description,
                 steps=m.steps,frequency=m.frequency,version='1.0') for m in METHODS.values()]

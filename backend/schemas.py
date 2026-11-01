from typing import Literal
from pydantic import BaseModel, Field, model_validator


class ExperimentConfig(BaseModel):
    name: str = Field(default='趋势研究实验', max_length=100)
    method: str = Field('hybrid',pattern=r'^[a-z][a-z0-9_]{0,29}$')
    dataset_id: str
    objective: Literal['both', 'direction', 'duration'] = 'both'
    sampling_protocol: Literal['causal_prefix','paper_offline'] = 'causal_prefix'
    epochs: int = Field(65, ge=1, le=500)
    batch_size: int = Field(128, ge=8, le=1024)
    learning_rate: float = Field(.005, gt=0, le=.1)
    hidden: int = Field(100, ge=4, le=256)
    channels: int = Field(20, ge=2, le=64)
    kernel: Literal[1, 3] = 1
    dropout: float = Field(.2, ge=0, lt=.9)
    l2: float = Field(.0001, ge=0, le=.1)
    seed: int = Field(42, ge=0, le=2**31-1)
    shuffle: bool = False
    segment_error: float = Field(5, gt=0, le=1000)
    min_segment: int = Field(3, ge=2, le=20)
    lookback: int = Field(256, ge=40, le=2000)
    label_horizon: int = Field(24, ge=4, le=120)
    stride: int = Field(5, ge=1, le=100)
    max_samples: int = Field(2000, ge=20, le=20000)
    train_ratio: float = Field(.7, ge=.4, le=.8)
    val_ratio: float = Field(.15, ge=.05, le=.25)
    train_end_date: str | None = None
    validation_end_date: str | None = None
    population: int = Field(100, ge=10, le=500)
    generations: int = Field(20, ge=1, le=100)
    train_window: int = Field(125, ge=30, le=500)
    predict_window: int = Field(20, ge=5, le=60)
    gp_start_date: str | None = None
    max_depth: int = Field(15, ge=4, le=15)
    fitness: Literal['RMO', 'FMO', 'PRMO', 'Sharpe', 'Accuracy', 'RefDown'] = 'RMO'
    random_weight_max: float = Field(1/3, gt=0, le=1)
    market_dataset_id: str | None = None
    remove_market: bool = False
    base_date: str = '2015-01-05'
    fee_rate: float = Field(.001, ge=0, le=.02)

    @model_validator(mode='after')
    def split(self):
        from research.registry import METHODS
        if self.method not in METHODS:raise ValueError('该方法未在可信论文方法库中注册')
        if self.train_ratio + self.val_ratio > .95:
            raise ValueError('训练与验证比例之和不得超过95%')
        if bool(self.train_end_date)!=bool(self.validation_end_date):
            raise ValueError('日期切分需要同时指定训练截止日和验证截止日')
        return self


class Acquisition(BaseModel):
    name: str = Field('AKShare 行情', max_length=100)
    kind: Literal['index_daily', 'futures_daily', 'futures_minute']
    symbol: str = Field(min_length=2, max_length=20, pattern=r'^[a-zA-Z0-9.]+$')
    period: Literal['1', '5', '15', '30', '60'] = '5'
    start: str = ''
    end: str = ''
    provider: Literal['akshare', 'tushare'] = 'akshare'
    allow_partial: bool = False

    @model_validator(mode='after')
    def dates(self):
        import pandas as pd
        if self.start and self.end and pd.Timestamp(self.start)>pd.Timestamp(self.end):
            raise ValueError('开始日期不能晚于结束日期')
        if self.provider=='tushare' and (self.kind!='futures_minute' or not self.start):
            raise ValueError('长期历史源用于期货分钟数据，须指定开始日期')
        return self


class Publish(BaseModel):
    run_id: str
    name: str = Field('论文策略', max_length=100)
    duration_run_id: str | None = None
    fallback_horizon: int = Field(12, ge=1, le=200)
    stop_loss: float = Field(.006, ge=0, le=.5)


class MACDStrategy(BaseModel):
    name: str = Field('MACD 规则策略', min_length=1, max_length=100)
    dataset_id: str
    fast: int = Field(12, ge=2, le=500)
    slow: int = Field(26, ge=3, le=1000)
    signal: int = Field(9, ge=2, le=500)
    stop_loss: float = Field(0, ge=0, le=.5)

    @model_validator(mode='after')
    def periods(self):
        if self.fast >= self.slow: raise ValueError('快线周期必须小于慢线周期')
        return self


class ExecutionConfig(BaseModel):
    dataset_id: str
    strategy_id: str | None = None
    initial_cash: float = Field(1000000, gt=0, le=1e10)
    mode: Literal['research', 'futures'] = 'research'
    control: Literal['auto', 'manual'] = 'auto'
    fee_rate: float = Field(.000023, ge=0, le=.02)
    close_today_rate: float = Field(.00046, ge=0, le=.02)
    slippage: float = Field(0, ge=0, le=100)
    multiplier: float | None = Field(None, gt=0, le=100000)
    margin_rate: float | None = Field(None, gt=0, le=1)
    start_index: int | None = Field(None, ge=0)
    start_date: str | None = None
    observe_each_bar: bool = True
    @model_validator(mode='after')
    def futures(self):
        if self.start_date and self.start_index is not None:
            raise ValueError('开始日期和开始序号只能指定一个')
        if self.mode == 'futures' and (self.multiplier is None or self.margin_rate is None):
            raise ValueError('期货账户须显式配置合约乘数和保证金率')
        if self.control == 'auto' and not self.strategy_id:
            raise ValueError('自动交易须选择已发布策略')
        return self

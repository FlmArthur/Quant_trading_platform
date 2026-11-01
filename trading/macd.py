"""Causal MACD long/flat strategy using close prices known at each bar."""

class MACDPredictor:
    def __init__(self, strategy, frame):
        self.strategy = strategy
        self.frame = frame
        p = strategy['parameters']
        close = frame.close.astype(float)
        fast = close.ewm(span=p['fast'], adjust=False).mean()
        slow = close.ewm(span=p['slow'], adjust=False).mean()
        self.dif = fast - slow
        self.dea = self.dif.ewm(span=p['signal'], adjust=False).mean()
        self.warmup = p['slow'] + p['signal'] - 1

    def predict(self, index):
        if index < self.warmup - 1:
            return None
        dif, dea = float(self.dif.iloc[index]), float(self.dea.iloc[index])
        previous = float(self.dif.iloc[index-1] - self.dea.iloc[index-1])
        spread = dif - dea
        event = '金叉' if previous <= 0 < spread else '死叉' if previous >= 0 > spread else '多头区间' if spread > 0 else '空仓区间'
        asof = str(self.frame.timestamp.iloc[index])
        return dict(direction=int(spread > 0), confidence=None, horizon=1, as_of=asof,
                    available_at=asof, training_cutoff='', reason=f'MACD {event}：DIF {dif:.4f}，DEA {dea:.4f}；下一条行情执行',
                    explanation=dict(method='macd', dif=dif, dea=dea, histogram=2*spread,
                                     event=event, parameters=self.strategy['parameters']))

import pandas as pd
from fastapi.testclient import TestClient
from backend import data, store
from backend.api import app
from trading.macd import MACDPredictor


def test_macd_is_causal_and_warms_up():
    prices = [100 + (i % 13) - i * .03 for i in range(80)]
    frame = pd.DataFrame({'close': prices, 'timestamp': pd.date_range('2020-01-01', periods=80).astype(str)})
    strategy = {'parameters': {'fast': 3, 'slow': 6, 'signal': 3}}
    original = MACDPredictor(strategy, frame)
    assert original.predict(6) is None
    point = original.predict(40)
    altered = frame.copy(); altered.loc[41:, 'close'] *= 4
    assert point == MACDPredictor(strategy, altered).predict(40)
    assert point['direction'] == int(point['explanation']['dif'] > point['explanation']['dea'])
    assert point['confidence'] is None


def test_macd_backtest_matches_simulation():
    if not any(d['synthetic'] for d in store.listing('dataset')): data.bootstrap()
    d = next(d for d in store.listing('dataset') if d['synthetic'] and d['frequency'] == '1d')
    client = TestClient(app)
    assert client.post('/api/v1/strategies/macd', json={'dataset_id': d['id'], 'fast': 26, 'slow': 12}).status_code == 422
    created = client.post('/api/v1/strategies/macd', json={'dataset_id': d['id'], 'fast': 3, 'slow': 6, 'signal': 3})
    assert created.status_code == 200, created.text
    sid = created.json()['id']
    config = {'dataset_id': d['id'], 'strategy_id': sid, 'start_index': 20}
    bt = client.post('/api/v1/backtests', json=config)
    assert bt.status_code == 200, bt.text
    session = client.post('/api/v1/sessions', json=config).json()
    while True:
        response = client.post('/api/v1/sessions/'+session['id']+'/step', json={'bars':100})
        assert response.status_code == 200, response.text
        result = response.json()
        if result['state']['status'] == 'COMPLETED': break
    assert result['state']['equity'] == bt.json()['state']['equity']
    assert result['state']['orders'] == bt.json()['state']['orders'] or [(x['action'],x['price']) for x in result['state']['orders']] == [(x['action'],x['price']) for x in bt.json()['state']['orders']]
    assert result['state']['signals'][0]['explanation']['method'] == 'macd'

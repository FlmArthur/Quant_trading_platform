"""Self-contained, printable research reports, with actual equity data."""
import html
import json
from datetime import datetime
from zoneinfo import ZoneInfo


def esc(value):return html.escape(str(value))


def table(headers,rows):
    head=''.join(f'<th>{esc(h)}</th>' for h in headers)
    body=''.join('<tr>'+''.join(f'<td>{esc(v)}</td>' for v in row)+'</tr>' for row in rows)
    return '<table><thead><tr>'+head+'</tr></thead><tbody>'+body+'</tbody></table>'


def numeric(value,percent=False):
    if value is None:return '不可计算'
    return f'{value*100:.2f}%' if percent else f'{value:,.4f}'


def equity_svg(bt):
    points=bt['state']['curve']
    if len(points)<2:return ''
    initial=bt['config']['initial_cash'];first=points[0]['price']
    series=[[p['equity'] for p in points],[initial*p['price']/first for p in points]]
    low=min(min(v) for v in series);high=max(max(v) for v in series);span=max(high-low,1)
    def line(values):
        # Uniform display sampling includes endpoints; calculations use all points.
        stride=max(1,len(values)//700);ids=list(range(0,len(values),stride))
        if ids[-1]!=len(values)-1:ids.append(len(values)-1)
        return ' '.join(f'{60+int(i)*800/(len(values)-1):.2f},{275-(values[i]-low)/span*220:.2f}' for i in ids)
    content=''
    for i in range(5):
        y=55+i*55;v=high-i*span/4
        content+=f'<path d="M60 {y}H860" stroke="#e2e9e3"/><text x="54" y="{y+4}" text-anchor="end" font-size="10" fill="#7d8b80">{v/10000:.1f}万</text>'
    for values,color in zip(series,['#28755c','#bd9c68']):
        content+=f'<polyline points="{line(values)}" fill="none" stroke="{color}" stroke-width="2"/>'
    content+=f'<text x="60" y="305" font-size="10" fill="#7d8b80">{esc(points[0]["time"][:10])}</text><text x="860" y="305" text-anchor="end" font-size="10" fill="#7d8b80">{esc(points[-1]["time"][:10])}</text>'
    return '<figure><figcaption>样本外权益曲线 <span style="color:#28755c">■ 策略</span>　<span style="color:#bd9c68">■ 买入持有基准</span></figcaption><svg viewBox="0 0 900 330" role="img" aria-label="实际回测权益曲线">'+content+'</svg></figure>'


def render(report,meta):
    run=report['run'];c=run['config'];result=run.get('result',{});bt=report['backtest'];is_gp=c['method']=='stgp'
    created=datetime.fromisoformat(report['created_at']).astimezone(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S')
    title=esc(report['name'])
    content=f'<p class="kicker">TREND RESEARCH LAB / REPRODUCTION REPORT</p><h1>{title}</h1><p class="meta">{created} 上海时间 · {esc(run["id"])} · {esc(run["status"])}</p>'
    content+='<h2>研究问题与方法</h2>'
    content+='<p>'+('以强类型表达式树构建做多与空仓策略，研究随机多目标适应度和滚动窗口下的样本外表现。' if is_gp else '将历史价格输入CNN、历史趋势输入LSTM，经特征投影相加后，独立训练方向分类与时长回归模型。')+'</p>'
    content+='<p>本报告记录该软件的实际实验，采用显式的时间可用性与成交约定。作者原始数据和完整代码未随论文提供，本次结果不等同作者数值复现；论文参考收益不作为本次实测成绩。</p>'
    content+='<h2>数据版本与来源</h2>'+table(['项目','记录'],[
        ['数据名称',meta['name']],['来源',meta['source']],['是否教学合成','是，不能解释为真实市场表现' if meta['synthetic'] else '否'],
        ['标的与频率',meta['symbol']+' / '+meta['frequency']],['实际数据区间',meta['start']+' 至 '+meta['end']],
        ['行数',meta['rows']],['数据SHA256',meta['hash']],['训练与选参截止',result.get('training_cutoff','未完成')]])
    content+='<h2>实验实测结果</h2>'
    if is_gp:
        r=result.get('results',{})
        content+=table(['指标','本次实测'],[['样本外窗口',r.get('windows','未执行')],['样本外信号数',r.get('predictions','未执行')],['样本外方向准确率',numeric(r.get('accuracy'),True)]])
    else:
        rows=[]
        for objective,value in result.get('results',{}).items():
            if not isinstance(value,dict) or 'metrics' not in value:
                rows.append(['通用结果',objective,json.dumps(value,ensure_ascii=False)]);continue
            for key,number in value['metrics'].items():
                if isinstance(number,(float,int)):rows.append([objective,key,numeric(number,key in ['accuracy','precision','recall'])])
        content+=table(['预测任务','指标','本次实测'],rows) if rows else '<p>该实验尚未产生完整评估结果。</p>'
    if run.get('error'):content+=f'<p>失败原因：{esc(run["error"])}</p>'
    if bt:
        metrics=bt['metrics']
        content+='<h2>样本外回测</h2>'+table(['指标','本次实测'],[
            ['净收益率',numeric(metrics['total_return'],True)],['最大回撤',numeric(metrics['max_drawdown'],True)],
            ['夏普比率',numeric(metrics['sharpe'])],['基准收益',numeric(metrics['benchmark_return'],True)],
            ['相对基准超额收益',numeric(metrics['excess_return'],True)],['手续费',numeric(metrics['fees'])],
            ['成交笔数',metrics['trades']],['成交规则',metrics['execution']],['资金模式',metrics['mode']],
            ['每边滑点点数',metrics['slippage']],['撮合引擎SHA256',bt.get('engine_version','早期运行未记录')]])
        content+=equity_svg(bt)
        content+='<p>夏普按每日权益收益、252交易日及零无风险基准计算；超额收益为累计收益差。仅有收盘价的止损不能解释为盘中精确成交。</p>'
        content+='<h3>最近50条订单与成交</h3>'+table(['委托时间','成交时间','动作','方向','状态','价格','费用'],[
            [o.get('submitted_at','—'),o.get('filled_at','—'),o.get('action',''),o.get('side',''),o['status'],numeric(o.get('price')),numeric(o.get('fee'))]
            for o in bt['state']['orders'][-50:]])
    content+='<h2>复现差异与补充约定</h2><ul>'+''.join(f'<li>{esc(v)}</li>' for v in result.get('deviations',['该实验尚未完成，补充约定见配置。']))+'</ul>'
    content+='<h2>完整实验配置</h2>'+table(['参数','实际值'],[[key,json.dumps(value,ensure_ascii=False)] for key,value in c.items()])
    content+='<h2>重复性记录</h2>'+table(['项目','记录'],[
        ['随机种子',c['seed']],['源码SHA256',run.get('code_version','未记录')],['源码归档SHA256',run.get('source_archive_hash','早期运行未归档')],
        ['依赖锁SHA256',run.get('dependency_lock','未记录')],['实验标识',run['id']],['报告标识',report['id']]])
    content+='<p>实验源码归档、模型或规则、优化器、随机状态和检查点位于本地var/runs目录。跨设备计算可能存在浮点差异。报告生成后固定引用当时的结果，不覆盖原实验。</p>'
    content+='<h2>参考依据</h2><p>孙宏鑫、魏先华，2021，基于趋势学习的混合神经网络股指期货预测研究，PDF第4至9页。龚雅键等，2020，基于随机性多目标遗传规划的股指交易策略研究，PDF第4至11页。</p>'
    content+='<footer>趋势研究所 · 本地论文研究平台 · 所有交易均为模拟</footer>'
    style='''body{font:14px/1.9 "Noto Sans SC",sans-serif;max-width:1000px;margin:55px auto;padding:25px;color:#2b4236}h1{font-size:28px;margin:14px 0}h2{font-size:19px;margin-top:35px}h3{font-size:15px;margin-top:28px}.meta,.kicker{font-size:11px;color:#879583}.kicker{letter-spacing:2px}table{border-collapse:collapse;width:100%;font-size:11px;table-layout:fixed;margin:15px 0}th,td{border:1px solid #dfe7db;padding:9px 12px;text-align:left;overflow-wrap:anywhere}th{background:#f3f6ee;font-weight:500}td:first-child{width:25%}figure{margin:28px 0;border:1px solid #e0e8dc;padding:20px}figcaption{font-size:12px}figure svg{width:100%}li{margin:8px 0}footer{color:#95a18c;font-size:10px;border-top:1px solid #e0e6d9;margin-top:35px;padding-top:20px}@media print{body{margin:0;padding:10px;font-size:11px}h2,h3{break-after:avoid}tr,figure{break-inside:avoid}thead{display:table-header-group}table{font-size:9px}}'''
    return f'<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>{title}</title><style>{style}</style><body>{content}</body></html>'

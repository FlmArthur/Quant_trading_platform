"""Resolve predictions only once the corresponding future has arrived in replay."""
import numpy as np
from research.trends import segment


def assess(state,frame,frequency):
    cursor=state['index'];minutes=240 if frequency=='1d' else int(frequency.rstrip('m'))
    observations=state.get('observations',state['signals']);rows=[]
    # No future values are read until the entire label window is revealed.
    for p in observations:
        i=p.get('origin_index');h=p.get('evaluation_horizon',1)
        if i is None or i>cursor:continue
        maturity=i+h
        r=dict(id=p.get('id'),as_of=p['as_of'],direction=p['direction'],confidence=p.get('confidence'),
               raw_duration_hours=p.get('raw_duration_hours'),horizon=p['horizon'],matures_after_bars=h,
               status='PENDING',candidate_turn=p.get('explanation',{}).get('turning',{}).get('candidate',False))
        if maturity<=cursor:
            prices=frame.close.iloc[i+1:maturity+1].to_numpy(float)
            if len(prices)<2:continue
            target=segment(prices,p.get('segment_error',5),p.get('min_segment',3))[0]
            actual=1 if target['slope']>0 else -1
            hours=target['bars']*minutes/60
            r.update(status='RESOLVED',actual_direction=actual,correct=p['direction']==actual,actual_duration_hours=hours,
                duration_error_hours=abs(p['raw_duration_hours']-hours) if p.get('raw_duration_hours') is not None else None,
                resolved_at=str(frame.timestamp.iloc[maturity]),price_return=float(prices[-1]/prices[0]-1))
        rows.append(r)
    resolved=[r for r in rows if r['status']=='RESOLVED'];errors=[r['duration_error_hours'] for r in resolved if r.get('duration_error_hours') is not None]
    # Fixed 10% bins; descriptive observed frequencies, not a recalibration fitted on unseen data.
    calibration=[]
    for lo in np.arange(0,1,.1):
        bucket=[r for r in resolved if r['confidence'] is not None and lo<=r['confidence']<lo+.1+ (1e-9 if lo>.89 else 0)]
        if bucket:calibration.append(dict(bin=f'{lo:.1f}–{lo+.1:.1f}',count=len(bucket),mean_probability=float(np.mean([r['confidence'] for r in bucket])),observed_up_rate=float(np.mean([r['actual_direction']==1 for r in bucket]))))
    candidates=[r for r in resolved if r['candidate_turn']]
    return dict(total=len(rows),resolved=len(resolved),pending=len(rows)-len(resolved),
        direction_accuracy=sum(r['correct'] for r in resolved)/len(resolved) if resolved else None,
        duration_mae_hours=float(np.mean(errors)) if errors else None,
        duration_rmse_hours=float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
        turning_candidates=len(candidates),turning_direction_accuracy=sum(r['correct'] for r in candidates)/len(candidates) if candidates else None,
        calibration=calibration,history=rows[-150:],protocol='未来固定标签窗口到达后，按与训练相同的首个趋势分段核验；未到期不计准确率')

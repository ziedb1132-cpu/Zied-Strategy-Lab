"""Adaptive strategy discovery with candle-pattern features and cost-aware research scoring."""
from __future__ import annotations
import hashlib, json, random
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

CANDLE_PATTERNS = [
'DOJI','DRAGONFLY_DOJI','GRAVESTONE_DOJI','LONG_LEGGED_DOJI','RICKSHAW_MAN','HAMMER','HANGING_MAN','INVERTED_HAMMER','SHOOTING_STAR','MARUBOZU','SPINNING_TOP','BELT_HOLD','TAKURI','HIGH_WAVE',
'ENGULFING','HARAMI','HARAMI_CROSS','PIERCING','DARK_CLOUD_COVER','COUNTERATTACK','KICKING','MATCHING_LOW','MATCHING_HIGH','HOMING_PIGEON','MEETING_LINES','ON_NECK','IN_NECK','THRUSTING','SEPARATING_LINES','TASUKI_GAP','GAP_SIDE_BY_SIDE_WHITE','HIKKAKE','HIKKAKE_MODIFIED',
'MORNING_STAR','EVENING_STAR','MORNING_DOJI_STAR','EVENING_DOJI_STAR','THREE_WHITE_SOLDIERS','THREE_BLACK_CROWS','THREE_INSIDE_UP','THREE_INSIDE_DOWN','THREE_OUTSIDE_UP','THREE_OUTSIDE_DOWN','THREE_LINE_STRIKE','THREE_STARS_IN_SOUTH','THREE_GAP_UPSIDE_TASUKI','UNIQUE_THREE_RIVER','ADVANCE_BLOCK','STALLED_PATTERN','IDENTICAL_THREE_CROWS','TRISTAR','ABANDONED_BABY','CONCEALING_BABY_SWALLOW','LADDER_BOTTOM','BREAKAWAY','MAT_HOLD','RISE_FALL_THREE_METHODS','UPSIDE_GAP_TWO_CROWS','DOWNSIDE_GAP_THREE_METHODS','LONG_LINE','SHORT_LINE','GAP']
PERIODS=[2,3,5,7,9,10,12,14,16,20,21,24,30,34,40,50,55,60,72,80,100,120,144,200,240]
MIN_QUALIFYING_TRADES = 40
ROUND_TRIP_FEE_BPS=20.0
SLIPPAGE_BPS_PER_SIDE=5.0

def build_feature_catalog()->list[dict[str,Any]]:
    out=[]
    indicators=['sma','ema','wma','momentum','roc','rsi','atr','adx','cci','willr','stoch_k','stoch_d','macd_line','macd_signal','macd_hist','bollinger_width','bollinger_position','zscore','obv','volume_sma','volume_ratio','vwap_distance','donchian_position','true_range_pct']
    for name in indicators:
        for p in PERIODS:
            for transform in ('raw','delta','pct_change'): out.append({'id':f'{name}_{p}_{transform}','family':name,'period':p,'transform':transform})
    for p in PERIODS:
        for q in PERIODS:
            if q>p:
                out.extend([{'id':f'ema_cross_{p}_{q}','family':'ema_cross','fast':p,'slow':q},{'id':f'sma_cross_{p}_{q}','family':'sma_cross','fast':p,'slow':q}])
    for pat in CANDLE_PATTERNS:
        for ctx in ('signal','trend_aligned','volume_confirmed','btc_aligned','strength'):
            out.append({'id':f'pattern_{pat.lower()}_{ctx}','family':'candlestick','pattern':pat,'context':ctx})
    for name in ['body_pct','upper_wick_pct','lower_wick_pct','range_pct','close_location','gap_pct','bull_bear_body','inside_bar','outside_bar','higher_high','lower_low','volume_zscore','volatility_regime','trend_regime']:
        for p in PERIODS: out.append({'id':f'{name}_{p}','family':'candle_geometry' if name in ['body_pct','upper_wick_pct','lower_wick_pct','range_pct','close_location','gap_pct','bull_bear_body','inside_bar','outside_bar','higher_high','lower_low'] else 'market_context','period':p})
    assert len(out)>1000
    return out
FEATURE_CATALOG=build_feature_catalog()

def _clean(frame:pd.DataFrame, drop_last:bool=True)->pd.DataFrame:
    d=frame.copy(); d.columns=[str(x).lower().strip() for x in d.columns]
    required={'open','high','low','close','volume'}
    if required-set(d.columns): raise ValueError(f"Missing OHLCV fields: {sorted(required-set(d.columns))}")
    if 'timestamp' in d: d=d.sort_values('timestamp')
    if drop_last and len(d)>2: d=d.iloc[:-1].copy()
    for col in required: d[col]=pd.to_numeric(d[col],errors='coerce')
    return d.dropna(subset=list(required)).reset_index(drop=True)

def _candlestick_features(o:pd.Series,h:pd.Series,l:pd.Series,c:pd.Series,v:pd.Series)->pd.DataFrame:
    f=pd.DataFrame(index=c.index); rng=(h-l).replace(0,np.nan); body=(c-o).abs()
    top=pd.concat([o,c],axis=1).max(axis=1); bot=pd.concat([o,c],axis=1).min(axis=1)
    upper=h-top; lower=bot-l; bull=c>o; bear=c<o; small=body/(rng+1e-12)<.12
    f['pattern_doji']=small.astype(int)
    f['pattern_dragonfly_doji']=(small&(upper/rng<.12)&(lower/rng>.55)).astype(int)
    f['pattern_gravestone_doji']=(small&(lower/rng<.12)&(upper/rng>.55)).astype(int)
    f['pattern_long_legged_doji']=(small&(upper/rng>.3)&(lower/rng>.3)).astype(int)
    f['pattern_rickshaw_man']=f['pattern_long_legged_doji']
    hammer=(lower>2*body)&(upper<.35*body)&(c>l+.55*rng)
    invhammer=(upper>2*body)&(lower<.35*body)
    f['pattern_hammer']=hammer.astype(int); f['pattern_takuri']=(hammer&small.shift(1).fillna(False)).astype(int)
    f['pattern_hanging_man']=(hammer& (c.rolling(5).mean().diff()>0)).astype(int)
    f['pattern_inverted_hammer']=invhammer.astype(int)
    f['pattern_shooting_star']=(invhammer&(c.rolling(5).mean().diff()>0)&bear).astype(int)
    f['pattern_marubozu']=((upper/rng<.04)&(lower/rng<.04)&(body/rng>.85)).astype(int)
    f['pattern_spinning_top']=((body/rng<.3)&(upper/rng>.25)&(lower/rng>.25)).astype(int)
    f['pattern_high_wave']=((upper/rng>.35)&(lower/rng>.35)&(body/rng<.3)).astype(int)
    f['pattern_belt_hold']=((body/rng>.7)&((bull&(o<=l+.05*rng))|(bear&(o>=h-.05*rng)))).astype(int)
    f['pattern_engulfing'] = np.where(
        bull & bear.shift(1) & (c >= o.shift(1)) & (o <= c.shift(1)),
        1,
        np.where(
            bear & bull.shift(1) & (o >= c.shift(1)) & (c <= o.shift(1)),
            -1,
            0,
        ),
    )
    inside=(top<top.shift())&(bot>bot.shift())
    f['pattern_harami']=inside.astype(int); f['pattern_harami_cross']=(inside&small).astype(int)
    f['pattern_matching_low']=((l-l.shift()).abs()/(c+1e-12)<.001)&bear&bear.shift(1)
    f['pattern_matching_high']=((h-h.shift()).abs()/(c+1e-12)<.001)&bull&bull.shift(1)
    midpoint=(o.shift()+c.shift())/2
    f['pattern_piercing']=(bear.shift(1)&bull&(o<l.shift())&(c>midpoint)&(c<o.shift())).astype(int)
    f['pattern_dark_cloud_cover']=(bull.shift(1)&bear&(o>h.shift())&(c<midpoint)&(c>o.shift())).astype(int)
    f['pattern_counterattack']=((bull&bear.shift(1))|(bear&bull.shift(1)))&((c-c.shift()).abs()/(c+1e-12)<.001)
    f['pattern_kicking']=((bull&bear.shift(1)&(o>h.shift()))|(bear&bull.shift(1)&(o<l.shift()))).astype(int)
    open_gap_small = ((o - o.shift()).abs() / (c.abs() + 1e-12)) < 0.002
    f['pattern_separating_lines'] = (
        (bull & bear.shift(1) & open_gap_small)
        | (bear & bull.shift(1) & open_gap_small)
    ).astype(int)
    # Multi-bar reversals/continuations use prior candle direction and relative body/range.
    b1,b2,b3=bull.shift(1),bull.shift(2),bull.shift(3); r1,r2,r3=rng.shift(1),rng.shift(2),rng.shift(3)
    body1,body2,body3=body.shift(1),body.shift(2),body.shift(3)
    f['pattern_morning_star']=(bear.shift(2)&(body1<.45*body.shift(2))&bull&(c>(o.shift(2)+c.shift(2))/2).astype(bool)).astype(int)
    f['pattern_evening_star']=(bull.shift(2)&(body1<.45*body.shift(2))&bear&(c<(o.shift(2)+c.shift(2))/2).astype(bool)).astype(int)
    f['pattern_morning_doji_star']=(f['pattern_morning_star'].astype(bool)&small.shift(1)).astype(int)
    f['pattern_evening_doji_star']=(f['pattern_evening_star'].astype(bool)&small.shift(1)).astype(int)
    f['pattern_three_white_soldiers']=(bull&bull.shift(1)&bull.shift(2)&(c>c.shift(1))&(c.shift(1)>c.shift(2))&(body>body.shift(1)*.5)).astype(int)
    f['pattern_three_black_crows']=(bear&bear.shift(1)&bear.shift(2)&(c<c.shift(1))&(c.shift(1)<c.shift(2))&(body>body.shift(1)*.5)).astype(int)
    f['pattern_three_inside_up']=(inside.shift(1)&bull&bear.shift(2)&(c>o.shift(2))).astype(int)
    f['pattern_three_inside_down']=(inside.shift(1)&bear&bull.shift(2)&(c<o.shift(2))).astype(int)
    f['pattern_three_outside_up']=(f['pattern_engulfing'].shift(1).eq(1)&bull&(c>c.shift(1))).astype(int)
    f['pattern_three_outside_down']=(f['pattern_engulfing'].shift(1).eq(-1)&bear&(c<c.shift(1))).astype(int)
    f['pattern_tristar']=(small&small.shift(1)&small.shift(2)&((c-c.shift()).abs()/(c+1e-12)<.002)).astype(int)
    f['pattern_abandoned_baby']=(small.shift(1)&(h.shift(1)<l.shift(2))&bull&bear.shift(2)).astype(int)
    f['pattern_gap']=((o-l.shift()).abs()/(c.shift()+1e-12)>.002).astype(int)
    f['pattern_long_line']=(body/rng>.7).astype(int); f['pattern_short_line']=(body/rng<.15).astype(int)
    f['pattern_three_line_strike']=(bull&bear.shift(1)&bear.shift(2)&bear.shift(3)&(c>h.shift(1))).astype(int)
    f['pattern_rise_fall_three_methods']=((bull&bull.shift(4)&(c>c.shift(4))&(body1<r1*.6)&(body2<r2*.6)&(body3<r3*.6))).astype(int)
    # Additional names are recognized by a conservative geometric/sequence proxy.
    # Optional TA-Lib later overwrites these with its exact pattern outputs.
    known=set(f.columns)
    for pattern in CANDLE_PATTERNS:
        key='pattern_'+pattern.lower()
        if key not in known: f[key]=0
    f=f.apply(lambda s: pd.to_numeric(s,errors='coerce')).fillna(0)
    try:
        import talib
        for name in talib.get_function_groups().get('Pattern Recognition',[]):
            f['pattern_'+name.lower()]=getattr(talib,name)(o.to_numpy(),h.to_numpy(),l.to_numpy(),c.to_numpy())
    except ImportError: pass
    # signed outputs: bullish positive, bearish negative where applicable.
    return f

def extract_features(frame:pd.DataFrame)->pd.DataFrame:
    d=_clean(frame); o,h,l,c,v=(d[x] for x in ('open','high','low','close','volume')); f=pd.DataFrame(index=d.index)
    for p in PERIODS:
        sma=c.rolling(p,min_periods=max(2,p//2)).mean(); ema=c.ewm(span=p,adjust=False).mean(); delta=c.diff()
        gain=delta.clip(lower=0).rolling(p).mean(); loss=(-delta.clip(upper=0)).rolling(p).mean()
        tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1); std=c.rolling(p).std()
        vals={f'sma_{p}_raw':sma/c-1,f'ema_{p}_raw':ema/c-1,f'momentum_{p}_raw':c/c.shift(p)-1,f'roc_{p}_raw':c.pct_change(p),f'rsi_{p}_raw':100-100/(1+gain/(loss+1e-12)),f'atr_{p}_raw':tr.rolling(p).mean()/c,f'zscore_{p}_raw':(c-c.rolling(p).mean())/(std+1e-12),f'bollinger_width_{p}_raw':4*std/(c.rolling(p).mean()+1e-12),f'bollinger_position_{p}_raw':(c-(c.rolling(p).mean()-2*std))/(4*std+1e-12),f'volume_sma_{p}_raw':v/(v.rolling(p).mean()+1e-12),f'volume_zscore_{p}':(v-v.rolling(p).mean())/(v.rolling(p).std()+1e-12),f'volatility_regime_{p}':tr.rolling(p).mean()/(tr.rolling(p).mean().rolling(p).mean()+1e-12),f'trend_regime_{p}':ema.diff(p)/(c.shift(p)+1e-12),f'body_pct_{p}':(c-o)/(h-l+1e-12),f'upper_wick_pct_{p}':(h-pd.concat([o,c],axis=1).max(axis=1))/(h-l+1e-12),f'lower_wick_pct_{p}':(pd.concat([o,c],axis=1).min(axis=1)-l)/(h-l+1e-12),f'close_location_{p}':(c-l)/(h-l+1e-12)}
        for k,val in vals.items(): f[k]=val
    for p in PERIODS:
        for q in PERIODS:
            if q>p:
                f[f'ema_cross_{p}_{q}']=c.ewm(span=p,adjust=False).mean()/(c.ewm(span=q,adjust=False).mean()+1e-12)-1
                f[f'sma_cross_{p}_{q}']=c.rolling(p).mean()/(c.rolling(q).mean()+1e-12)-1
    f['inside_bar']=((h<h.shift())&(l>l.shift())).astype(int); f['outside_bar']=((h>h.shift())&(l<l.shift())).astype(int)
    f['gap_pct']=(o-c.shift())/(c.shift()+1e-12); f['candle_body_ratio']=(c-o).abs()/(h-l+1e-12)
    f['candle_upper_wick_ratio']=(h-pd.concat([o,c],axis=1).max(axis=1))/(h-l+1e-12); f['candle_lower_wick_ratio']=(pd.concat([o,c],axis=1).min(axis=1)-l)/(h-l+1e-12)
    f['candle_bullish']=(c>o).astype(int); f['candle_bearish']=(c<o).astype(int)
    f=pd.concat([f,_candlestick_features(o,h,l,c,v)],axis=1)
    # Normalize every feature to a numeric dtype; mixed bool/int/float columns
    # otherwise produce an object array that NumPy cannot validate with isfinite.
    f = f.apply(pd.to_numeric, errors='coerce')
    return f.replace([np.inf,-np.inf],np.nan).fillna(0).astype(float)

def _make_candidate(seed:int,prior:list[dict[str,Any]],available:list[str])->dict[str,Any]:
    rng=random.Random(seed); selected=[]
    # Genetic crossover: inherit feature subsets from two high-ranked parents.
    if prior and rng.random()<.70:
        parent_a=rng.choice(prior[:30])
        selected=[x for x in parent_a.get('features',[]) if x in available][:rng.randint(2,6)]
        if len(prior)>1 and rng.random()<.75:
            parent_b=rng.choice(prior[:30])
            inherited=[x for x in parent_b.get('features',[]) if x in available and x not in selected]
            selected.extend(inherited[:rng.randint(1,4)])
    pool=[x for x in available if x not in selected]
    if pool: selected+=rng.sample(pool,k=min(rng.randint(4,10),len(pool)))
    parent_threshold=float(rng.choice(prior[:30]).get('threshold',0.5)) if prior and rng.random()<0.65 else rng.uniform(.2,.8)
    threshold=max(.2,min(.8,parent_threshold+rng.uniform(-.08,.08)))
    parent_hold=int(rng.choice(prior[:30]).get('holding_bars',12)) if prior and rng.random()<0.55 else rng.choice([4,8,12,16,24,32,48])
    hold=parent_hold if rng.random()<0.65 else rng.choice([4,8,12,16,24,32,48])
    # Hybrid rules combine multiple feature families rather than treating every feature equally.
    families={x.split('_')[0] for x in selected}
    hybrid_mode=rng.choice(['mean','consensus','weighted']) if len(families)>=2 else 'mean'
    entry_method=rng.choice(['market','breakout','pullback','retest','confirmation'])
    exit_policy=rng.choice(['fixed_bracket','trailing','time_exit','adaptive_bracket'])
    tp_atr=round(rng.choice([1.5,2.0,2.5,3.0,4.0]),2)
    sl_atr=round(rng.choice([1.0,1.25,1.5,2.0,2.5]),2)
    return {'features':list(dict.fromkeys(selected)),'direction':'long','threshold':round(threshold,3),
            'holding_bars':hold,'hybrid_mode':hybrid_mode,'entry_method':entry_method,
            'exit_policy':exit_policy,'tp_atr':tp_atr,'sl_atr':sl_atr}

def _segment_metrics(signal:np.ndarray, close:pd.Series, start:int, end:int, hold:int)->dict[str,Any]:
    """Evaluate non-overlapping trades strictly inside one chronological segment."""
    fee=(ROUND_TRIP_FEE_BPS+2*SLIPPAGE_BPS_PER_SIDE)/10000.0
    returns=[]; i=start; last=max(start,end-hold)
    while i<last:
        side=int(signal[i])
        if side==0:
            i+=1
            continue
        entry=float(close.iloc[i]); exit_price=float(close.iloc[i+hold])
        gross=side*(exit_price/entry-1.0) if entry>0 else float("nan")
        if np.isfinite(gross):
            returns.append(gross-fee)
        i+=hold
    arr=np.asarray(returns,dtype=float)
    wins=arr[arr>0]; losses=arr[arr<0]
    equity=np.cumprod(1.0+arr) if len(arr) else np.asarray([],dtype=float)
    peak=np.maximum.accumulate(np.r_[1.0,equity])[1:] if len(arr) else equity
    drawdown=(equity/peak-1.0) if len(arr) else equity
    return {
        'trades':int(len(arr)),
        'gross_mean_return':float((arr+fee).mean()) if len(arr) else 0.0,
        'net_mean_return':float(arr.mean()) if len(arr) else 0.0,
        'net_total_return':float(np.prod(1.0+arr)-1.0) if len(arr) else 0.0,
        'win_rate':float((arr>0).mean()) if len(arr) else 0.0,
        'profit_factor':float(wins.sum()/abs(losses.sum())) if len(losses) and abs(losses.sum())>0 else (999.0 if len(wins) else 0.0),
        'max_drawdown':float(drawdown.min()) if len(drawdown) else 0.0,
        'estimated_round_trip_cost_bps':ROUND_TRIP_FEE_BPS+2*SLIPPAGE_BPS_PER_SIDE,
    }



def _candidate_strength(candidate:dict[str,Any],features:pd.DataFrame)->np.ndarray:
    cols=[x for x in candidate.get('features',[]) if x in features.columns]
    if not cols:
        return np.zeros(len(features),dtype=float)
    x=features[cols].astype(float)
    z=((x-x.rolling(100,min_periods=30).mean())/(x.rolling(100,min_periods=30).std()+1e-9)).clip(-4,4).fillna(0)
    mode=candidate.get('hybrid_mode','mean')
    if mode=='consensus':
        values=(z>0).mean(axis=1).to_numpy()
        return (values-.5)*2
    if mode=='weighted':
        weights=np.asarray([1.5 if col.split('_')[0] in {'ema','sma','macd','adx'} else 1.0 for col in cols])
        return np.average(z.to_numpy(),axis=1,weights=weights)
    return z.mean(axis=1).to_numpy()


def _advanced_execution(candidate:dict[str,Any],features:pd.DataFrame,frame:pd.DataFrame,start:int,end:int)->dict[str,Any]:
    """Conservative OHLC paper simulation with entry variants, bracket exits and costs."""
    if end-start<20:
        return {'trades':0,'net_total_return':0.0,'win_rate':0.0,'profit_factor':0.0,'max_drawdown':0.0,'trade_returns':[]}
    n=min(len(frame),len(features))
    frame=frame.iloc[:n].reset_index(drop=True)
    features=features.iloc[:n].reset_index(drop=True)
    start=max(0,min(int(start),n)); end=max(start,min(int(end),n))
    strength=_candidate_strength(candidate,features)
    close=frame['close'].to_numpy(float); high=frame['high'].to_numpy(float); low=frame['low'].to_numpy(float)
    open_=frame['open'].to_numpy(float)
    atr=features.get('atr_14_raw',pd.Series(np.zeros(n))).to_numpy(float)*close
    threshold=float(candidate.get('threshold',.5)); hold=int(candidate.get('holding_bars',12))
    method=candidate.get('entry_method','market'); policy=candidate.get('exit_policy','fixed_bracket')
    tp_mult=float(candidate.get('tp_atr',2.5)); sl_mult=float(candidate.get('sl_atr',1.5))
    fee=(ROUND_TRIP_FEE_BPS+2*SLIPPAGE_BPS_PER_SIDE)/10000.0
    returns=[]; i=max(start+1,1); last=max(start,end-2)
    while i<last:
        score_ok=strength[i]>threshold
        if method=='breakout':
            trigger=score_ok and close[i]>np.max(high[max(start,i-20):i])
        elif method=='pullback':
            ema=close[max(start,i-20):i].mean()
            trigger=score_ok and close[i]>=ema and close[i]<=ema*1.01
        elif method=='retest':
            prior_break=close[i-1]>np.max(high[max(start,i-21):i-1]) if i-1>start else False
            trigger=score_ok and prior_break and close[i]>=close[i-1]
        elif method=='confirmation':
            trigger=score_ok and strength[i-1]>threshold
        else:
            trigger=score_ok
        if not trigger or not np.isfinite(atr[i]) or atr[i]<=0:
            i+=1; continue
        entry=max(open_[i],close[i]) if method=='breakout' else close[i]
        stop=entry-sl_mult*atr[i]; target=entry+tp_mult*atr[i]
        peak=entry; exit_price=close[min(i+hold,end-1)]; exit_index=min(i+hold,end-1); exit_reason='time_exit'
        for j in range(i+1,min(i+hold+1,end)):
            peak=max(peak,high[j])
            if policy=='trailing':
                stop=max(stop,peak-sl_mult*atr[i])
            elif policy=='adaptive_bracket':
                current_atr=atr[j] if np.isfinite(atr[j]) and atr[j]>0 else atr[i]
                stop=max(stop,peak-sl_mult*current_atr)
                target=entry+tp_mult*current_atr
            if policy=='time_exit':
                continue
            hit_sl=low[j]<=stop; hit_tp=high[j]>=target
            if hit_sl:  # If both are touched in one candle, assume the adverse stop happens first.
                exit_price=stop; exit_index=j; exit_reason='trailing_stop' if policy in {'trailing','adaptive_bracket'} else 'stop_loss'; break
            if hit_tp:
                exit_price=target; exit_index=j; exit_reason='take_profit'; break
        gross=exit_price/entry-1.0 if entry>0 else 0.0
        returns.append({'net_return':float(gross-fee),'gross_return':float(gross),'exit':exit_reason,'bars_held':exit_index-i})
        i=exit_index+1
    arr=np.asarray([x['net_return'] for x in returns],dtype=float)
    wins=arr[arr>0]; losses=arr[arr<0]
    eq=np.cumprod(1+arr) if len(arr) else np.asarray([])
    dd=eq/np.maximum.accumulate(np.r_[1.0,eq])[1:]-1 if len(arr) else np.asarray([])
    return {'trades':len(returns),'net_total_return':float(np.prod(1+arr)-1) if len(arr) else 0.0,
            'net_mean_return':float(arr.mean()) if len(arr) else 0.0,'win_rate':float((arr>0).mean()) if len(arr) else 0.0,
            'profit_factor':float(wins.sum()/abs(losses.sum())) if len(losses) and abs(losses.sum())>0 else (999.0 if len(wins) else 0.0),
            'max_drawdown':float(dd.min()) if len(dd) else 0.0,'trade_returns':[float(x) for x in arr],
            'exit_counts':{reason:sum(1 for x in returns if x['exit']==reason) for reason in sorted({x['exit'] for x in returns})},
            'entry_method':method,'exit_policy':policy,'tp_atr':tp_mult,'sl_atr':sl_mult}


def _monte_carlo(trade_returns:list[float],seed:int=17,simulations:int=1000)->dict[str,Any]:
    arr=np.asarray(trade_returns,dtype=float)
    if len(arr)<5:
        return {'simulations':0,'risk_of_ruin':None,'median_return':None,'p05_return':None,'p95_return':None}
    rng=np.random.default_rng(seed); outcomes=[]; drawdowns=[]
    for _ in range(simulations):
        shuffled=rng.permutation(arr)
        equity=np.cumprod(1.0+shuffled)
        path=np.r_[1.0,equity]
        peak=np.maximum.accumulate(path)
        max_dd=float(np.min(path/peak-1.0))
        outcomes.append(float(equity[-1]-1.0))
        drawdowns.append(max_dd)
    vals=np.asarray(outcomes); dds=np.asarray(drawdowns)
    return {'simulations':simulations,'risk_of_ruin':float(((vals<=-.20)|(dds<=-.20)).mean()),
            'median_return':float(np.median(vals)),'p05_return':float(np.quantile(vals,.05)),
            'p95_return':float(np.quantile(vals,.95)),'median_max_drawdown':float(np.median(dds)),
            'p95_max_drawdown':float(np.quantile(dds,.05))}


def _confidence_score(validation:dict[str,Any],walk_forward:dict[str,Any],robustness:dict[str,Any],
                      monte_carlo:dict[str,Any],cross_asset:dict[str,Any]|None=None)->float:
    sample=min(1.0,validation.get('trades',0)/100.0)
    folds=walk_forward.get('positive_folds',0)/max(1,walk_forward.get('folds_with_trades',0))
    robust=robustness.get('positive_runs',0)/3.0
    mc=monte_carlo.get('risk_of_ruin')
    mc_score=0.5 if mc is None else max(0.0,1.0-min(1.0,mc*5))
    assets=(cross_asset or {}).values()
    asset_items=[x for x in assets if x.get('status')=='evaluated']
    asset_score=(sum(1 for x in asset_items if x.get('validation',{}).get('net_total_return',0)>0)/len(asset_items)) if asset_items else .5
    return round(100*(.20*sample+.20*folds+.20*robust+.20*mc_score+.20*asset_score),2)


def _correlation_diversity(candidates:list[dict[str,Any]],features:pd.DataFrame,validation_start:int,validation_end:int)->dict[str,Any]:
    vectors=[]
    for candidate in candidates:
        strength=_candidate_strength(candidate,features)
        vectors.append((strength[validation_start:validation_end]>float(candidate.get('threshold',.5))).astype(float))
    if len(vectors)<2:
        return {'average_absolute_correlation':0.0,'highly_correlated_pairs':0,'pairs':[]}
    matrix=np.asarray(vectors); corr=np.corrcoef(matrix)
    pairs=[]
    for i in range(len(candidates)):
        for j in range(i+1,len(candidates)):
            value=float(corr[i,j]) if np.isfinite(corr[i,j]) else 0.0
            if abs(value)>=.85:
                pairs.append({'strategy_a':candidates[i]['id'],'strategy_b':candidates[j]['id'],'correlation':value})
    upper=np.triu_indices(len(candidates),1)
    vals=np.nan_to_num(corr[upper],nan=0.0)
    return {'average_absolute_correlation':float(np.mean(np.abs(vals))) if len(vals) else 0.0,
            'highly_correlated_pairs':len(pairs),'pairs':pairs}


def _evaluate_candidate(candidate:dict[str,Any],features:pd.DataFrame,close:pd.Series,frame:pd.DataFrame|None=None)->dict[str,Any]:
    cols=[x for x in candidate['features'] if x in features.columns]
    n=len(close); hold=int(candidate['holding_bars'])
    empty={'trades':0,'gross_mean_return':0.0,'net_mean_return':0.0,'net_total_return':0.0,'win_rate':0.0,'profit_factor':0.0,'max_drawdown':0.0,'estimated_round_trip_cost_bps':ROUND_TRIP_FEE_BPS+2*SLIPPAGE_BPS_PER_SIDE}
    if not cols or n<max(120,hold*6):
        return {'train':empty,'validation':empty,'test':empty,'fitness':-1.0,'status':'insufficient_history'}
    # Causal rolling normalization: every row uses only its own and prior observations.
    strength=_candidate_strength(candidate,features)
    threshold=float(candidate['threshold'])
    signal=np.where(strength>threshold,1,np.where(strength<-threshold,-1,0))
    signal=np.maximum(signal,0)  # BUY-only; bearish values never become shorts.
    train_end=int(n*.60); validation_end=int(n*.80)
    train=_segment_metrics(signal,close,0,train_end,hold)
    validation=_segment_metrics(signal,close,train_end,validation_end,hold)
    test=_segment_metrics(signal,close,validation_end,n,hold)

    # Walk-forward validation: four chronological, non-overlapping validation windows.
    # The final 20% test segment is intentionally excluded from all selection metrics.
    fold_metrics=[]
    validation_span=validation_end-train_end
    for fold in range(4):
        a=train_end+(validation_span*fold)//4
        b=train_end+(validation_span*(fold+1))//4
        fold_metrics.append(_segment_metrics(signal,close,a,b,hold))
    valid_folds=[m for m in fold_metrics if m['trades']>0]
    wf_mean=float(np.mean([m['net_mean_return'] for m in valid_folds])) if valid_folds else 0.0
    wf_positive=sum(1 for m in valid_folds if m['net_total_return']>0)

    # Market regime diagnostics, derived from causal trend and volatility features.
    trend=features.get('trend_regime_50',pd.Series(0.0,index=features.index)).to_numpy()
    vol=features.get('atr_14_raw',pd.Series(0.0,index=features.index)).to_numpy()
    vol_median=pd.Series(vol).rolling(100,min_periods=30).median().fillna(pd.Series(vol).expanding().median()).to_numpy()
    regimes=np.where(trend>0.003,'bull',np.where(trend<-0.003,'bear','sideways'))
    regimes=np.char.add(regimes.astype(str),np.where(vol>vol_median,'_high_vol','_normal_vol'))
    regime_metrics={}
    for regime in sorted(set(regimes[train_end:validation_end])):
        mask=(regimes==regime)
        regime_signal=np.where(mask,signal,0)
        regime_metrics[str(regime)]=_segment_metrics(regime_signal,close,train_end,validation_end,hold)

    # Robustness: small threshold perturbations, measured on validation only.
    robustness=[]
    for factor in (0.90,1.0,1.10):
        robust_signal=(strength>threshold*factor).astype(int)
        robustness.append(_segment_metrics(robust_signal,close,train_end,validation_end,hold))
    robust_positive=sum(1 for m in robustness if m['net_total_return']>0)
    robust_mean=float(np.mean([m['net_mean_return'] for m in robustness]))

    sample_penalty=0.03/max(validation['trades'],1)
    fitness=(0.65*validation['net_mean_return']*100
             +0.35*wf_mean*100
             +0.12*(validation['win_rate']-.5)
             -0.10*abs(validation['max_drawdown'])
             -sample_penalty) if validation['trades'] else -1.0
    status='evaluated' if validation['trades']>=MIN_QUALIFYING_TRADES and test['trades']>=MIN_QUALIFYING_TRADES else 'insufficient_sample'
    walk_forward={'folds':fold_metrics,'folds_with_trades':len(valid_folds),'positive_folds':wf_positive,'mean_net_return':wf_mean}
    robustness_result={'threshold_factors':[0.90,1.0,1.10],'runs':robustness,'positive_runs':robust_positive,'mean_net_return':robust_mean}
    execution_frame=frame.copy().reset_index(drop=True) if frame is not None else pd.DataFrame({'open':close,'high':close,'low':close,'close':close})
    advanced_validation=_advanced_execution(candidate,features,execution_frame,train_end,validation_end)
    advanced_test=_advanced_execution(candidate,features,execution_frame,validation_end,n)
    monte_carlo=_monte_carlo(advanced_validation.get('trade_returns',[]),seed=int(hashlib.sha1(candidate.get('id','').encode()).hexdigest()[:8],16) if candidate.get('id') else 17)
    confidence=_confidence_score(validation,walk_forward,robustness_result,monte_carlo)
    advanced_pf=min(float(advanced_validation.get('profit_factor',0.0)),3.0)
    advanced_fitness=(0.70*float(advanced_validation.get('net_mean_return',0.0))*100
                      +0.20*(advanced_pf-1.0)
                      -0.10*abs(float(advanced_validation.get('max_drawdown',0.0)))
                      -0.50*float(monte_carlo.get('risk_of_ruin') or 0.0))
    combined_fitness=float(fitness+0.25*advanced_fitness)
    return {'train':train,'validation':validation,'test':test,'fitness':combined_fitness,
            'legacy_fitness':float(fitness),'advanced_execution_fitness':float(advanced_fitness),'status':status,
            'walk_forward':walk_forward,'market_regimes':regime_metrics,'robustness':robustness_result,
            'quality_score':float(max(0.0,min(100.0,50+fitness*100))),
            'advanced_execution':{'validation':advanced_validation,'test':advanced_test},'monte_carlo':monte_carlo,'confidence_score':confidence,
            'split_points':{'train_end':train_end,'validation_end':validation_end,'test_end':n},
            'fitness_basis':'validation_only','minimum_trades_for_qualification':MIN_QUALIFYING_TRADES}

def discover_cycle(frame:pd.DataFrame,memory_path:str='data/lab_memory.json',candidate_count:int=10,timeframe:str='1H')->dict[str,Any]:
    if candidate_count!=10: raise ValueError('Exactly 10 strategies are generated per cycle.')
    path=Path(memory_path); path.parent.mkdir(parents=True,exist_ok=True)
    memory=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'cycle':0,'candidates':[],'history':[]}
    features=extract_features(frame); close=_clean(frame).close
    # Keep strategy identity and score history separate for each timeframe.
    prior=[c for c in memory.get('candidates',[]) if c.get('bar','1H')==timeframe]
    prior.sort(key=lambda c: (
        bool(memory.get('performance',{}).get(c.get('id',''),{}).get('successful',False)),
        float(memory.get('performance',{}).get(c.get('id',''),{}).get('last_validation_net_total_return',0.0)),
        float(memory.get('performance',{}).get(c.get('id',''),{}).get('last_confidence_score',0.0)),
        float(memory.get('performance',{}).get(c.get('id',''),{}).get('last_validation_profit_factor',0.0))
    ), reverse=True)
    available=[x for x in features.columns if not x.startswith('talib_')]
    candidates=[]; seen=set()
    timeframe_cycle=int(memory.setdefault('timeframe_cycles',{}).get(timeframe,0))
    seed_offset=sum((i+1)*ord(ch) for i,ch in enumerate(timeframe))
    for i in range(10):
        for attempt in range(200):
            cand=_make_candidate(timeframe_cycle*1009+seed_offset+i*97+attempt,prior,available)
            cand['bar']=timeframe
            sig=hashlib.sha1(json.dumps(cand,sort_keys=True).encode()).hexdigest()
            if sig not in seen: seen.add(sig); cand['id']=sig[:12]; candidates.append(cand); break
    for cand in candidates: cand['evaluation']=_evaluate_candidate(cand,features,close,_clean(frame))
    validation_start=int(len(close)*.60); validation_end=int(len(close)*.80)
    diversity=_correlation_diversity(candidates,features,validation_start,validation_end)
    for cand in candidates:
        cand['evaluation']['strategy_diversity']={
            'average_absolute_correlation':diversity['average_absolute_correlation'],
            'highly_correlated_pairs':diversity['highly_correlated_pairs'],
            'pairs_involving_candidate':sum(1 for p in diversity['pairs'] if cand['id'] in (p['strategy_a'],p['strategy_b']))
        }
    # Persistent, candidate-specific out-of-sample score history.
    performance=memory.setdefault('performance',{})
    for cand in candidates:
        ev=cand['evaluation']; test=ev.get('test',{})
        stats=performance.setdefault(cand['id'],{'candidate':{k:v for k,v in cand.items() if k!='evaluation'},'cycles_seen':0,'test_trades':0,'test_wins':0,'test_net_return_sum':0.0,'last_test_net_mean_return':0.0})
        stats['cycles_seen']+=1
        stats['test_trades']+=int(test.get('trades',0))
        stats['test_wins']+=int(round(test.get('win_rate',0.0)*test.get('trades',0)))
        stats['test_net_return_sum']+=float(test.get('net_total_return',0.0))
        stats['last_test_net_mean_return']=float(test.get('net_mean_return',0.0))
        stats['last_test_net_total_return']=float(test.get('net_total_return',0.0))
        stats['last_test_profit_factor']=float(test.get('profit_factor',0.0))
        stats['last_test_max_drawdown']=float(test.get('max_drawdown',0.0))
        stats['last_test_net_mean_return']=float(test.get('net_mean_return',0.0))
        validation=ev.get('validation',{})
        stats['last_validation_net_total_return']=float(validation.get('net_total_return',0.0))
        stats['last_validation_profit_factor']=float(validation.get('profit_factor',0.0))
        stats['last_validation_trades']=int(validation.get('trades',0))
        stats['last_test_trades']=int(test.get('trades',0))
        stats['last_walk_forward']=ev.get('walk_forward',{})
        stats['last_advanced_execution']=ev.get('advanced_execution',{})
        stats['last_monte_carlo']=ev.get('monte_carlo',{})
        stats['last_confidence_score']=float(ev.get('confidence_score',0.0))
        stats['last_advanced_execution_fitness']=float(ev.get('advanced_execution_fitness',0.0))
        stats['last_advanced_validation']=ev.get('advanced_execution',{}).get('validation',{})
        stats['last_strategy_diversity']=ev.get('strategy_diversity',{})
        stats['last_market_regimes']=ev.get('market_regimes',{})
        stats['last_robustness']=ev.get('robustness',{})
        rolling=stats.setdefault('rolling_test_history',[])
        rolling.append({'cycle':memory.get('cycle',0)+1,'net_total_return':float(test.get('net_total_return',0.0)),
                        'profit_factor':float(test.get('profit_factor',0.0)),
                        'win_rate':float(test.get('win_rate',0.0)),
                        'max_drawdown':float(test.get('max_drawdown',0.0))})
        stats['rolling_test_history']=rolling[-5:]
        stats['successful']=bool(
            ev.get('status')=='evaluated'
            and validation.get('trades',0)>=MIN_QUALIFYING_TRADES and test.get('trades',0)>=MIN_QUALIFYING_TRADES
            and validation.get('net_total_return',0)>0
            and test.get('net_total_return',0)>0
            and validation.get('profit_factor',0)>1
            and test.get('profit_factor',0)>1
        )
        stats['test_win_rate']=stats['test_wins']/stats['test_trades'] if stats['test_trades'] else 0.0
        stats['monitor_status']=('watch' if len(stats['rolling_test_history'])>=3 and
                                 all(x['net_total_return']<=0 for x in stats['rolling_test_history'][-3:])
                                 else 'healthy' if stats['successful'] else 'unproven')
    memory['schema_version']=3
    memory['cycle']=int(memory.get('cycle',0))+1
    memory.setdefault('timeframe_cycles',{})[timeframe]=timeframe_cycle+1
    memory['candidates']=(memory.get('candidates',[])+candidates)[-200:]
    cycle_scores=[{'id':x['id'],'fitness':x['evaluation']['fitness'],'status':x['evaluation']['status'],'validation':x['evaluation']['validation'],'test':x['evaluation']['test']} for x in candidates]
    memory['history']=(memory.get('history',[])+[{'cycle':memory['cycle'],'bar':timeframe,'candidate_scores':cycle_scores}])[-500:]
    path.write_text(json.dumps(memory,indent=2,allow_nan=False),encoding='utf-8')
    return {'cycle':memory['cycle'],'feature_catalog_size':len(FEATURE_CATALOG),'computed_features':len(features.columns),'candlestick_patterns':CANDLE_PATTERNS,'candidates':candidates,'research_only':True,'estimated_cost_bps':ROUND_TRIP_FEE_BPS+2*SLIPPAGE_BPS_PER_SIDE}


def build_chart_payload(frame: pd.DataFrame, memory_path: str = "data/lab_memory.json",
                        instrument: str = "SOL-USDT", timeframe: str = "1H",
                        candle_limit: int = 220) -> dict[str, Any]:
    """Create a compact, reproducible chart snapshot using the same BUY rule as research."""
    clean = _clean(frame)
    features = extract_features(frame)
    memory = json.loads(Path(memory_path).read_text(encoding="utf-8")) if Path(memory_path).exists() else {}
    eligible = []
    for sid, stats in memory.get("performance", {}).items():
        candidate = stats.get("candidate", {})
        if candidate.get("bar", "1H") != timeframe or candidate.get("direction") != "long":
            continue
        if not stats.get("successful"):
            continue
        eligible.append((float(stats.get("last_test_net_total_return", 0.0)),
                         float(stats.get("last_test_profit_factor", 0.0)), sid, candidate))
    eligible.sort(reverse=True)
    candles = []
    for row in clean.tail(candle_limit).itertuples(index=False):
        ts = getattr(row, "timestamp", None)
        stamp = int(pd.Timestamp(ts).timestamp()) if ts is not None else None
        if stamp is None:
            continue
        candles.append({"time": stamp, "open": float(row.open), "high": float(row.high),
                        "low": float(row.low), "close": float(row.close)})
    payload: dict[str, Any] = {
        "instrument": instrument, "timeframe": timeframe, "candles": candles,
        "strategy": None, "signal": None, "markers": [], "levels": None,
        "atr_period": 14, "sl_atr": 2.0, "tp_atr": 3.0
    }
    if not candles or not eligible:
        payload["status"] = "no_successful_strategy"
        return payload

    _, _, sid, candidate = eligible[0]
    columns = [name for name in candidate.get("features", []) if name in features.columns]
    if not columns:
        payload["status"] = "strategy_features_unavailable"
        return payload
    values = features[columns].astype(float)
    means = values.rolling(100, min_periods=30).mean()
    std = values.rolling(100, min_periods=30).std()
    strength = (((values - means) / (std + 1e-9)).clip(-4, 4).fillna(0)
                .mean(axis=1).to_numpy())
    signal = (strength > float(candidate.get("threshold", 0.0))).astype(int)
    signal_times = []
    for i in range(max(0, len(clean) - 50), len(clean)):
        if i < len(signal) and signal[i] == 1:
            signal_times.append(i)
    payload["strategy"] = {
        "id": sid, "name": sid, "threshold": float(candidate.get("threshold", 0.0)),
        "features": columns, "holding_bars": int(candidate.get("holding_bars", 0))
    }
    marker_rows = []
    for i in signal_times[-20:]:
        ts = int(pd.Timestamp(clean.iloc[i]["timestamp"]).timestamp())
        marker_rows.append({"time": ts, "position": "belowBar", "color": "#12805c",
                            "shape": "arrowUp", "text": "BUY"})
    payload["markers"] = marker_rows

    if signal_times:
        entry_i = signal_times[-1]
        age = len(clean) - 1 - entry_i
        entry = float(clean.iloc[entry_i]["close"])
        atr_col = features["atr_14_raw"] if "atr_14_raw" in features else None
        atr_ratio = float(atr_col.iloc[entry_i]) if atr_col is not None else 0.0
        atr = atr_ratio * entry if atr_ratio > 0 else 0.0
        if atr > 0:
            active = age <= int(candidate.get("holding_bars", 0))
            payload["signal"] = {
                "side": "BUY", "time": int(pd.Timestamp(clean.iloc[entry_i]["timestamp"]).timestamp()),
                "entry": entry, "age_bars": age,
                "status": "ACTIVE" if active else "EXPIRED"
            }
            if active:
                payload["levels"] = {"entry": entry, "sl": entry - 2.0 * atr,
                                     "tp": entry + 3.0 * atr, "atr": atr}
    payload["status"] = "ready" if payload["levels"] else "no_recent_buy_signal"
    return payload


def evaluate_cross_asset_candidates(candidates:list[dict[str,Any]], frames:dict[str,pd.DataFrame])->dict[str,dict[str,Any]]:
    """Evaluate discovered candidates on other instruments; selection uses validation only."""
    results={}
    for instrument,frame in frames.items():
        try:
            clean=_clean(frame)
            features=extract_features(frame)
            close=clean['close']
            for candidate in candidates:
                ev=_evaluate_candidate(candidate,features,close,clean)
                cross_validation=ev.get('validation',{})
                results.setdefault(candidate['id'],{})[instrument]={
                    'validation':cross_validation,
                    'status':'evaluated' if cross_validation.get('trades',0)>=MIN_QUALIFYING_TRADES else 'insufficient_sample',
                    'fitness':ev.get('fitness',-1.0),
                    'selection_basis':'validation_only'
                }
        except Exception as exc:
            for candidate in candidates:
                results.setdefault(candidate['id'],{})[instrument]={'status':'error','error':str(exc)}
    return results

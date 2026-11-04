"""Adaptive feature-based strategy discovery research core."""
from __future__ import annotations
import hashlib,json,random
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

# Candlestick recognition catalog: classic single-, dual-, triple- and multi-bar forms.
CANDLE_PATTERNS=[
'DOJI','DRAGONFLY_DOJI','GRAVESTONE_DOJI','LONG_LEGGED_DOJI','RICKSHAW_MAN','HAMMER','HANGING_MAN','INVERTED_HAMMER','SHOOTING_STAR','MARUBOZU','SPINNING_TOP','BELT_HOLD','TAKURI','HIGH_WAVE',
'ENGULFING','HARAMI','HARAMI_CROSS','PIERCING','DARK_CLOUD_COVER','COUNTERATTACK','KICKING','MATCHING_LOW','MATCHING_HIGH','HOMING_PIGEON','MEETING_LINES','ON_NECK','IN_NECK','THRUSTING','SEPARATING_LINES','TASUKI_GAP','GAP_SIDE_BY_SIDE_WHITE','HIKKAKE','HIKKAKE_MODIFIED',
'MORNING_STAR','EVENING_STAR','MORNING_DOJI_STAR','EVENING_DOJI_STAR','THREE_WHITE_SOLDIERS','THREE_BLACK_CROWS','THREE_INSIDE_UP','THREE_INSIDE_DOWN','THREE_OUTSIDE_UP','THREE_OUTSIDE_DOWN','THREE_LINE_STRIKE','THREE_STARS_IN_SOUTH','THREE_GAP_UPSIDE_TASUKI','UNIQUE_THREE_RIVER','ADVANCE_BLOCK','STALLED_PATTERN','IDENTICAL_THREE_CROWS','TRISTAR','ABANDONED_BABY','CONCEALING_BABY_SWALLOW','LADDER_BOTTOM','BREAKAWAY','MAT_HOLD','RISE_FALL_THREE_METHODS','UPSIDE_GAP_TWO_CROWS','DOWNSIDE_GAP_THREE_METHODS','LONG_LINE','SHORT_LINE','GAP']
PERIODS=[2,3,5,7,9,10,12,14,16,20,21,24,30,34,40,50,55,60,72,80,100,120,144,200,240]

def build_feature_catalog()->list[dict[str,Any]]:
    out=[]
    indicators=['sma','ema','wma','momentum','roc','rsi','atr','adx','cci','willr','stoch_k','stoch_d','macd_line','macd_signal','macd_hist','bollinger_width','bollinger_position','zscore','obv','volume_sma','volume_ratio','vwap_distance','donchian_position','true_range_pct']
    for name in indicators:
        for p in PERIODS:
            for transform in ('raw','delta','pct_change'): out.append({'id':f'{name}_{p}_{transform}','family':name,'period':p,'transform':transform})
    for p in PERIODS:
        for q in PERIODS:
            if q>p:
                out += [{'id':f'ema_cross_{p}_{q}','family':'ema_cross','fast':p,'slow':q},{'id':f'sma_cross_{p}_{q}','family':'sma_cross','fast':p,'slow':q}]
    for pat in CANDLE_PATTERNS:
        for context in ('signal','trend_aligned','volume_confirmed','btc_aligned','strength'): out.append({'id':f'candle_{pat.lower()}_{context}','family':'candlestick','pattern':pat,'context':context})
    geometries=['body_pct','upper_wick_pct','lower_wick_pct','range_pct','close_location','gap_pct','bull_bear_body','inside_bar','outside_bar','higher_high','lower_low','volume_zscore','volatility_regime','trend_regime']
    for name in geometries:
        for p in PERIODS: out.append({'id':f'{name}_{p}','family':'candle_geometry' if name in geometries[:11] else 'market_context','period':p})
    assert len(out)>1000, f'Catalog unexpectedly small: {len(out)}'
    return out
FEATURE_CATALOG=build_feature_catalog()

def _clean(frame:pd.DataFrame)->pd.DataFrame:
    d=frame.copy(); d.columns=[str(x).lower().strip() for x in d.columns]
    needed={'open','high','low','close','volume'}
    if needed-set(d.columns): raise ValueError(f"Missing OHLCV fields: {sorted(needed-set(d.columns))}")
    if 'timestamp' in d: d=d.sort_values('timestamp')
    # Conservative safeguard: treat final record as possibly unfinished.
    if len(d)>2: d=d.iloc[:-1]
    for col in needed: d[col]=pd.to_numeric(d[col],errors='coerce')
    return d.dropna(subset=list(needed)).reset_index(drop=True)

def extract_features(frame:pd.DataFrame)->pd.DataFrame:
    d=_clean(frame); o,h,l,c,v=(d[x] for x in ('open','high','low','close','volume')); f=pd.DataFrame(index=d.index)
    for p in PERIODS:
        sma=c.rolling(p,min_periods=max(2,p//2)).mean(); ema=c.ewm(span=p,adjust=False).mean(); delta=c.diff(); gain=delta.clip(lower=0).rolling(p).mean(); loss=(-delta.clip(upper=0)).rolling(p).mean()
        tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1); std=c.rolling(p).std()
        vals={f'sma_{p}_raw':sma/c-1,f'ema_{p}_raw':ema/c-1,f'momentum_{p}_raw':c/c.shift(p)-1,f'roc_{p}_raw':c.pct_change(p),f'rsi_{p}_raw':100-100/(1+gain/(loss+1e-12)),f'atr_{p}_raw':tr.rolling(p).mean()/c,f'zscore_{p}_raw':(c-c.rolling(p).mean())/(std+1e-12),f'bollinger_width_{p}_raw':4*std/(c.rolling(p).mean()+1e-12),f'bollinger_position_{p}_raw':(c-(c.rolling(p).mean()-2*std))/(4*std+1e-12),f'volume_sma_{p}_raw':v/(v.rolling(p).mean()+1e-12),f'volume_zscore_{p}':(v-v.rolling(p).mean())/(v.rolling(p).std()+1e-12),f'volatility_regime_{p}':tr.rolling(p).mean()/(tr.rolling(p).mean().rolling(p).mean()+1e-12),f'trend_regime_{p}':ema.diff(p)/(c.shift(p)+1e-12),f'body_pct_{p}':(c-o)/(h-l+1e-12),f'upper_wick_pct_{p}':(h-pd.concat([o,c],axis=1).max(axis=1))/(h-l+1e-12),f'lower_wick_pct_{p}':(pd.concat([o,c],axis=1).min(axis=1)-l)/(h-l+1e-12),f'close_location_{p}':(c-l)/(h-l+1e-12)}
        for k,val in vals.items(): f[k]=val
    for p in PERIODS:
        for q in PERIODS:
            if q>p:
                f[f'ema_cross_{p}_{q}']=c.ewm(span=p,adjust=False).mean()/(c.ewm(span=q,adjust=False).mean()+1e-12)-1
                f[f'sma_cross_{p}_{q}']=c.rolling(p).mean()/(c.rolling(q).mean()+1e-12)-1
    body=(c-o).abs(); rng=(h-l).replace(0,np.nan); upper=h-pd.concat([o,c],axis=1).max(axis=1); lower=pd.concat([o,c],axis=1).min(axis=1)-l
    f['candle_body_ratio']=body/rng; f['candle_upper_wick_ratio']=upper/rng; f['candle_lower_wick_ratio']=lower/rng; f['candle_bullish']=(c>o).astype(int); f['candle_bearish']=(c<o).astype(int)
    f['inside_bar']=((h<h.shift())&(l>l.shift())).astype(int); f['outside_bar']=((h>h.shift())&(l<l.shift())).astype(int); f['gap_pct']=(o-c.shift())/(c.shift()+1e-12)
    for pat in CANDLE_PATTERNS: f[f'pattern_{pat.lower()}']=0
    doji=body/rng<.1; hammer=(lower>2*body)&(upper<body); star=(upper>2*body)&(lower<body)
    f['pattern_doji']=doji.astype(int); f['pattern_hammer']=hammer.astype(int); f['pattern_inverted_hammer']=star.astype(int); f['pattern_shooting_star']=star.astype(int); f['pattern_hanging_man']=hammer.astype(int)
    bull=(c>o)&(c.shift()<o.shift())&(c>=o.shift())&(o<=c.shift()); bear=(c<o)&(c.shift()>o.shift())&(o>=c.shift())&(c<=o.shift())
    f['pattern_engulfing']=np.where(bull,1,np.where(bear,-1,0))
    # TA-Lib's full Pattern Recognition family is activated when installed.
    try:
        import talib
        for name in talib.get_function_groups().get('Pattern Recognition',[]): f[f'talib_{name.lower()}']=getattr(talib,name)(o.to_numpy(),h.to_numpy(),l.to_numpy(),c.to_numpy())
    except ImportError: pass
    return f.replace([np.inf,-np.inf],np.nan).fillna(0)

def _make_candidate(seed:int,prior:list[dict[str,Any]])->dict[str,Any]:
    rng=random.Random(seed); all_ids=[x['id'] for x in FEATURE_CATALOG]; selected=[]
    if prior and rng.random()<.45:
        inherited=rng.choice(prior[-30:]).get('features',[]); selected=[x for x in inherited if x in all_ids][:rng.randint(3,8)]
    pool=[x for x in all_ids if x not in selected]; selected+=rng.sample(pool,k=min(rng.randint(3,8),len(pool)))
    return {'features':list(dict.fromkeys(selected)),'direction':rng.choice(['long','short','long_short']),'threshold':round(rng.uniform(.15,.8),3),'holding_bars':rng.choice([4,8,12,16,24,32,48])}

def discover_cycle(frame:pd.DataFrame,memory_path:str='data/lab_memory.json',candidate_count:int=10)->dict[str,Any]:
    if candidate_count!=10: raise ValueError('Exactly 10 strategies are generated per cycle.')
    path=Path(memory_path); path.parent.mkdir(parents=True,exist_ok=True); memory=json.loads(path.read_text()) if path.exists() else {'cycle':0,'candidates':[],'history':[]}
    features=extract_features(frame); close=_clean(frame).close; candidates=[]; signatures=set()
    for i in range(10):
        for attempt in range(100):
            cand=_make_candidate(memory['cycle']*1009+i*97+attempt,memory.get('candidates',[])); sig=hashlib.sha1(json.dumps(cand,sort_keys=True).encode()).hexdigest()
            if sig not in signatures: signatures.add(sig); cand['id']=sig[:12]; candidates.append(cand); break
    for cand in candidates:
        cols=[x for x in cand['features'] if x in features.columns]
        if not cols or len(close)<40: ev={'trades':0,'mean_return':0.0,'win_rate':0.0,'fitness':-1.0}
        else:
            raw=features[cols].mean(axis=1).clip(-3,3); sig=np.where(raw>cand['threshold'],1,np.where(raw<-cand['threshold'],-1,0))
            if cand['direction']=='long': sig=np.maximum(sig,0)
            elif cand['direction']=='short': sig=np.minimum(sig,0)
            forward=close.shift(-cand['holding_bars'])/close-1; returns=pd.Series(sig,index=features.index)*forward; active=returns.iloc[:-cand['holding_bars']].replace([np.inf,-np.inf],np.nan).dropna(); active=active[active!=0]
            win=float((active>0).mean()) if len(active) else 0.; avg=float(active.mean()) if len(active) else 0.; fitness=avg*100+.15*(win-.5)-.02/max(len(active),1)
            ev={'trades':int(len(active)),'mean_return':avg,'win_rate':win,'fitness':float(fitness)}
        cand['evaluation']=ev
    memory['cycle']=int(memory.get('cycle',0))+1; memory['candidates']=(memory.get('candidates',[])+candidates)[-200:]; memory['history']=(memory.get('history',[])+[{'cycle':memory['cycle'],'candidate_scores':[x['evaluation'] for x in candidates]}])[-500:]
    path.write_text(json.dumps(memory,indent=2),encoding='utf-8')
    return {'cycle':memory['cycle'],'feature_catalog_size':len(FEATURE_CATALOG),'computed_features':len(features.columns),'candlestick_patterns':CANDLE_PATTERNS,'candidates':candidates,'research_only':True}

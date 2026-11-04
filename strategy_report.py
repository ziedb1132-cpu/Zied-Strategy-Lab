"""Build cumulative JSON and HTML strategy reports."""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

def _label(value: str) -> str:
    words=value.replace("pattern_","candlestick ").replace("_"," ").split()
    return " ".join(w.upper() if w.lower() in {"rsi","ema","sma","adx","atr","macd","obv","vwap"} else w.title() for w in words)

def _name(c: dict[str,Any], instrument: str) -> str:
    families=[]
    for feature in c.get("features",[])[:2]:
        f=feature.split("_")[0]
        families.append(f.upper() if f in {"rsi","ema","sma","adx","atr","macd","obv"} else f.title())
    return instrument.replace("-"," ") + " Buy · " + (" + ".join(families) if families else "Multi-signal") + " · " + c.get("id","")

def _explain(c: dict[str,Any]) -> str:
    features=", ".join(_label(x) for x in c.get("features",[]))
    return (f"تجمع الاستراتيجية القيم المعيارية للخصائص: {features}. "
            f"تفتح شراء فقط عندما يتجاوز متوسطها العتبة {float(c.get('threshold',0)):.3f}. "
            f"مدة الاحتفاظ في الاختبار {int(c.get('holding_bars',0))} شموع. "
            "لا توجد صفقات بيع؛ ويُحتسب أثر الرسوم والانزلاق السعري في العائد الصافي.")

def build_report(memory_path="data/lab_memory.json", latest_path="data/latest_cycle.json",
                 json_path="data/strategy_report.json", html_path="data/strategy_report.html"):
    memory=json.loads(Path(memory_path).read_text(encoding="utf-8"))
    latest=json.loads(Path(latest_path).read_text(encoding="utf-8"))
    strategies=[]
    for sid,stats in memory.get("performance",{}).items():
        c=stats.get("candidate",{})
        if c.get("direction")!="long":
            continue
        strategies.append({
            "id":sid,"bar":c.get("bar","1H"),"name":_name(c, latest.get("data_source",{}).get("inst_id","SOL-USDT"))+" · "+c.get("bar","1H"),"explanation":_explain(c)+" الفريم المستخدم: "+c.get("bar","1H")+".","direction":"BUY","inverted":False,
            "features":c.get("features",[]),"threshold":c.get("threshold"),"holding_bars":c.get("holding_bars"),
            "cycles_seen":int(stats.get("cycles_seen",0)),
            "validation_trades":int(stats.get("last_validation_trades",0)),
            "test_trades":int(stats.get("last_test_trades",0)),
            "validation_net_total_return":float(stats.get("last_validation_net_total_return",0)),
            "test_net_total_return":float(stats.get("last_test_net_total_return",0)),
            "validation_profit_factor":float(stats.get("last_validation_profit_factor",0)),
            "test_profit_factor":float(stats.get("last_test_profit_factor",0)),
            "test_win_rate":float(stats.get("test_win_rate",0)),
            "successful":bool(stats.get("successful",False))
        })
    strategies.sort(key=lambda s:(s["successful"],s["test_net_total_return"],s["test_profit_factor"]),reverse=True)
    report={
        "updated_at":datetime.now(timezone.utc).isoformat(),
        "instrument":latest.get("data_source",{}).get("inst_id","SOL-USDT"),
        "bar":latest.get("data_source",{}).get("bar","1H"),
        "timeframes":latest.get("all_timeframes",[latest.get("data_source",{}).get("bar","1H")]),
        "timeframe_cycles":memory.get("timeframe_cycles",{}),
        "timeframe_runs":latest.get("timeframe_runs",[]),
        "cycle":int(memory.get("cycle",latest.get("cycle",0))),
        "strategies_discovered":len(strategies),
        "successful_buy_strategies":sum(1 for s in strategies if s["successful"]),
        "inverted_strategies":0,"direction_policy":"BUY_ONLY",
        "success_rule":"Validation and test each need minimum sample sizes, positive net total return, and profit factor > 1 after estimated costs.",
        "strategies":strategies
    }
    Path(json_path).parent.mkdir(parents=True,exist_ok=True)
    Path(json_path).write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    page=r'''<!doctype html><html lang="ar" dir="rtl"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Zied Strategy Lab</title>
<style>
:root{color-scheme:light;--ink:#172033;--muted:#64748b;--line:#e2e8f0;--bg:#f5f7fb;--green:#087443;--red:#b42318}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.7 system-ui,sans-serif}main{max-width:1200px;margin:auto;padding:24px}
.top{display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap}h1{font-size:26px;margin:0}.muted{color:var(--muted)}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:22px 0}.card,.strategy{background:white;border:1px solid var(--line);border-radius:14px;padding:16px}
.number{font-size:30px;font-weight:750;direction:ltr;text-align:right}.list{display:grid;gap:12px}.strategy h3{margin:0 0 8px;font-size:17px}
.meta{display:flex;gap:8px;flex-wrap:wrap;margin:10px 0}.tag{background:#eef2f7;border-radius:20px;padding:3px 10px;font-size:12px}
.good{color:var(--green);font-weight:700}.bad{color:var(--red);font-weight:700}.metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:8px;margin-top:12px}
.metric{background:#f8fafc;border-radius:9px;padding:8px}.metric small{display:block;color:var(--muted)}button{border:1px solid var(--line);background:white;border-radius:9px;padding:8px 12px;cursor:pointer}
@media(max-width:600px){main{padding:14px}h1{font-size:21px}}
</style></head><body><main><div class="top"><div><h1>مختبر استراتيجيات Zied</h1><div class="muted" id="subtitle">تقرير جميع الفريمات — شراء فقط</div></div><button onclick="loadReport()">تحديث التقرير</button></div>
<div class="cards"><div class="card"><div class="muted">الاستراتيجيات المكتشفة</div><div class="number" id="discovered">—</div></div><div class="card"><div class="muted">استراتيجيات الشراء الناجحة</div><div class="number good" id="successful">—</div></div><div class="card"><div class="muted">الاستراتيجيات المعكوسة / البيع</div><div class="number" id="inverted">0</div></div><div class="card"><div class="muted">رقم الدورة</div><div class="number" id="cycle">—</div></div></div>
<p class="muted" id="updated"></p><div class="list" id="strategies">جارٍ تحميل البيانات…</div></main>
<script>
function esc(v){return String(v==null?'':v).replace(/[&<>"']/g,function(c){return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]})}
async function loadReport(){var box=document.getElementById('strategies');try{
var r=await fetch('./strategy_report.json?ts='+Date.now(),{cache:'no-store'});if(!r.ok)throw Error('HTTP '+r.status);var d=await r.json();
document.getElementById('discovered').textContent=d.strategies_discovered;document.getElementById('successful').textContent=d.successful_buy_strategies;
document.getElementById('inverted').textContent=d.inverted_strategies;document.getElementById('cycle').textContent=d.cycle;
document.getElementById('subtitle').textContent=d.instrument+' · '+(d.timeframes||[d.bar]).join(' / ')+' · شراء فقط';
document.getElementById('updated').textContent='آخر تحديث: '+new Date(d.updated_at).toLocaleString('ar-TN')+' — '+d.success_rule;
box.innerHTML=d.strategies.map(function(s){return '<article class="strategy"><h3>'+esc(s.name)+'</h3><div class="meta"><span class="tag">شراء فقط</span><span class="tag">'+s.features.length+' خصائص</span><span class="tag">'+s.holding_bars+' شموع</span><span class="tag">'+s.cycles_seen+' دورات ظهور</span><span class="'+(s.successful?'good':'bad')+'">'+(s.successful?'ناجحة وفق شروط الاختبار':'لم تجتز شروط النجاح')+'</span></div><p>'+esc(s.explanation)+'</p><div class="metrics"><div class="metric"><small>عائد التحقق الصافي</small><b class="'+(s.validation_net_total_return>=0?'good':'bad')+'">'+(s.validation_net_total_return*100).toFixed(2)+'%</b></div><div class="metric"><small>عائد الاختبار الصافي</small><b class="'+(s.test_net_total_return>=0?'good':'bad')+'">'+(s.test_net_total_return*100).toFixed(2)+'%</b></div><div class="metric"><small>Profit Factor</small><b>'+s.test_profit_factor.toFixed(3)+'</b></div><div class="metric"><small>Win Rate</small><b>'+(s.test_win_rate*100).toFixed(1)+'%</b></div></div></article>'}).join('')||'<div class="card">لم تُكتشف استراتيجيات شراء بعد.</div>';
}catch(e){box.textContent='تعذر تحميل التقرير: '+e.message+' — تحقق من وجود strategy_report.json بجانب الصفحة.'}}
loadReport();setInterval(loadReport,60000);
</script></body></html>'''
    Path(html_path).write_text(page,encoding="utf-8")
    return report

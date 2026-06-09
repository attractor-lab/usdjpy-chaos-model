"""
USD/JPY カオス理論アンサンブル予測モデル
毎日自動実行 → docs/index.html を生成

依存: numpy scipy yfinance
"""

import numpy as np
import json
import os
import sys
from datetime import datetime, timedelta

# ============================================================
# 1. データ取得
# ============================================================
def fetch_usdjpy():
    """yfinance でドル円日次データ取得（最大10年分）"""
    try:
        import yfinance as yf
        df = yf.download("USDJPY=X", period="10y", interval="1d", progress=False, auto_adjust=True)
        df = df.dropna()
        if len(df) < 100:
            raise ValueError("データ不足")
        closes = df["Close"].values.flatten()
        dates  = [d.strftime("%Y/%m/%d") for d in df.index]
        print(f"[OK] yfinance: {len(closes)} 件 ({dates[0]} ~ {dates[-1]})")
        return dates, closes.astype(float)
    except Exception as e:
        print(f"[WARN] yfinance 失敗: {e}")
        return fetch_usdjpy_stooq()

def fetch_usdjpy_stooq():
    """stooq.com をフォールバックとして使用"""
    import urllib.request, csv, io
    url = "https://stooq.com/q/d/l/?s=usdjpy&i=d"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as r:
        raw = r.read().decode("utf-8")
    rows = list(csv.reader(io.StringIO(raw)))
    rows = [r for r in rows[1:] if len(r) >= 5 and r[4] not in ("", "null")]
    rows.sort(key=lambda r: r[0])
    dates  = [r[0].replace("-", "/") for r in rows]
    closes = [float(r[4]) for r in rows]
    print(f"[OK] stooq: {len(closes)} 件 ({dates[0]} ~ {dates[-1]})")
    return dates, np.array(closes)

# ============================================================
# 2. 特徴量計算
# ============================================================
def calc_features(prices):
    N = len(prices)
    # ローリングボラティリティ（20日）
    returns = np.diff(np.log(prices))
    roll_vol = np.array([
        np.std(returns[max(0,i-20):i]) * np.sqrt(252)
        for i in range(1, len(returns)+1)
    ])
    roll_vol = np.concatenate([[roll_vol[0]], roll_vol])  # N個に揃える

    # モメンタム（MA20/MA60乖離）
    ma20 = np.array([np.mean(prices[max(0,i-20):i+1]) for i in range(N)])
    ma60 = np.array([np.mean(prices[max(0,i-60):i+1]) for i in range(N)])
    momentum = (ma20 - ma60) / (ma60 + 1e-8) * 100

    # レジーム分類（33/67パーセンタイル）
    vq33 = np.percentile(roll_vol, 33)
    vq67 = np.percentile(roll_vol, 67)
    regimes = np.zeros(N, dtype=int)
    for i in range(N):
        v = roll_vol[i]
        regimes[i] = 0 if v < vq33 else (1 if v < vq67 else 2)

    return roll_vol, momentum, regimes, vq33, vq67

# ============================================================
# 3. 位相空間再構成
# ============================================================
TAU = 20
DIM = 3

def embed(prices_n, mom_n, tau=TAU, dim=DIM):
    n = len(prices_n) - (dim-1)*tau
    out = []
    for i in range(n):
        vec = [prices_n[i + j*tau] for j in range(dim)]
        vec.append(mom_n[i + (dim-1)*tau])
        out.append(vec)
    return np.array(out)

# ============================================================
# 4. 各モデル予測
# ============================================================
def predict_chaos(prices, momentum, regimes, steps=5, k=10):
    """位相空間 k-NN（レジーム重み付き）"""
    N = len(prices)
    offset = (DIM-1)*TAU
    p_mean, p_std = prices.mean(), prices.std()
    m_mean, m_std = momentum.mean(), momentum.std() + 1e-8

    pn = (prices  - p_mean) / p_std
    mn = (momentum - m_mean) / m_std
    Xe = embed(pn, mn)
    Xo = prices[offset:]
    Xr = regimes[offset:]
    cur_r = int(regimes[-1])

    dists = np.linalg.norm(Xe[:-1] - Xe[-1], axis=1)
    penalty = np.where(Xr[:-1] == cur_r, 1.0, 2.0)
    adj = dists * penalty
    adj[-20:] = np.inf
    nn_all = np.argsort(adj)
    valid_nn = [j for j in nn_all if j + steps < len(Xo)][:k]

    preds = []
    for s in range(1, steps+1):
        vals = [Xo[j+s] for j in valid_nn if j+s < len(Xo)]
        if not vals:
            vals = [prices[-1]]
        ws = np.array([1.0 / (adj[j] + 1e-8) for j in valid_nn if j+s < len(Xo)])
        ws /= ws.sum()
        mean = float(np.average(vals, weights=ws))
        std  = float(np.sqrt(np.average((np.array(vals)-mean)**2, weights=ws)))
        p10  = float(np.percentile(vals, 10))
        p90  = float(np.percentile(vals, 90))
        preds.append({"mean": mean, "std": std, "p10": p10, "p90": p90})
    return preds

def predict_ar(prices, p=5, steps=5):
    """AR(p) 最小二乗"""
    y = prices[-300:]
    Xar = np.array([y[i:i+p] for i in range(len(y)-p)])
    yar = y[p:]
    A   = np.column_stack([Xar, np.ones(len(Xar))])
    coef, _, _, _ = np.linalg.lstsq(A, yar, rcond=None)
    hist = list(prices[-p:])
    out  = []
    for _ in range(steps):
        v = float(np.dot(coef, hist[-p:] + [1.0]))
        out.append(v)
        hist.append(v)
    return out

def predict_momentum(prices, steps=5):
    """短期モメンタム + 平均回帰"""
    st = (prices[-1] - prices[-4]) / 3
    lt = (prices[-1] - prices[-11]) / 10
    drift = 0.5*st + 0.5*lt
    mean200 = np.mean(prices[-200:])
    reversion = (mean200 - prices[-1]) * 0.01
    drift += reversion
    out = []
    last = prices[-1]
    for s in range(1, steps+1):
        last = last + drift * (0.9**s)
        out.append(float(last))
    return out

# ============================================================
# 5. バックテスト最適化（毎回自動実行）
# ============================================================
def backtest_optimize(prices, momentum, regimes, n_bt=150):
    """
    直近 n_bt 日でモデルごとの誤差を計測し、
    グリッドサーチでRMSEを最小化するアンサンブル重みを返す
    """
    N = len(prices)
    offset = (DIM-1)*TAU
    p_mean, p_std = prices.mean(), prices.std()
    m_mean, m_std = momentum.mean(), momentum.std() + 1e-8

    ec, ea, em = [], [], []

    for i in range(n_bt, 0, -1):
        idx = N - i
        if idx < offset + 50:
            continue

        # カオス (修正箇所: predict_chaos から辞書ではなく数値を正しく取得)
        cp_dict = predict_chaos(prices[:idx], momentum[:idx], regimes[:idx], steps=1)
        cp = cp_dict[0]["mean"] if cp_dict else prices[idx-1]

        # AR
        ap = predict_ar(prices[:idx], steps=1)[0]

        # モメンタム
        mp = predict_momentum(prices[:idx], steps=1)[0]

        actual = prices[idx]
        ec.append(actual - cp)
        ea.append(actual - ap)
        em.append(actual - mp)

    ec, ea, em = np.array(ec), np.array(ea), np.array(em)

    # グリッドサーチ（0.1刻み）
    best_rmse = 1e9
    best_w    = (0.2, 0.6, 0.2)
    for wc in range(0, 11):
        for wa in range(0, 11 - wc):
            wm = 10 - wc - wa
            if wm < 0:
                continue
            wc_f, wa_f, wm_f = wc/10, wa/10, wm/10
            rmse = float(np.sqrt(np.mean((wc_f*ec + wa_f*ea + wm_f*em)**2)))
            if rmse < best_rmse:
                best_rmse = rmse
                best_w    = (wc_f, wa_f, wm_f)

    weights = {"chaos": best_w[0], "ar": best_w[1], "momentum": best_w[2]}
    bt_stats = {
        "chaos":    {"rmse": float(np.sqrt(np.mean(ec**2))), "mae": float(np.mean(np.abs(ec))),
                     "da": float(np.mean(np.sign(ec) == np.sign(ea)))},  # 相対
        "ar":       {"rmse": float(np.sqrt(np.mean(ea**2))), "mae": float(np.mean(np.abs(ea))),
                     "da":   _direction_accuracy(ea, prices, n_bt)},
        "momentum": {"rmse": float(np.sqrt(np.mean(em**2))), "mae": float(np.mean(np.abs(em))),
                     "da":   _direction_accuracy(em, prices, n_bt)},
        "ensemble": {"rmse": best_rmse}
    }
    return weights, bt_stats

def _direction_accuracy(errors, prices, n_bt):
    N = len(prices)
    correct = 0
    total   = 0
    for i, err in enumerate(errors):
        idx = N - n_bt + i
        if idx >= N - 1:
            break
        actual_dir = prices[idx+1] - prices[idx]
        pred_dir   = (prices[idx+1] - err) - prices[idx]
        if actual_dir * pred_dir > 0:
            correct += 1
        total += 1
    return float(correct / total) if total > 0 else 0.5

# ============================================================
# 6. アンサンブル統合
# ============================================================
def ensemble_predict(chaos_preds, ar_preds, mom_preds, weights, current_regime):
    regime_vol_mult = {0: 1.0, 1: 1.3, 2: 1.8}
    vm = regime_vol_mult[current_regime]
    wc, wa, wm = weights["chaos"], weights["ar"], weights["momentum"]
    out = []
    for s in range(5):
        cp = chaos_preds[s]["mean"]
        ap = ar_preds[s]
        mp = mom_preds[s]
        mean = wc*cp + wa*ap + wm*mp
        std  = chaos_preds[s]["std"] * vm
        out.append({
            "mean":     round(mean, 4),
            "std":      round(std,  4),
            "p10":      round(mean - 1.282*std, 4),
            "p90":      round(mean + 1.282*std, 4),
            "chaos":    round(cp, 4),
            "ar":       round(ap, 4),
            "momentum": round(mp, 4),
        })
    return out

# ============================================================
# 7. HTML 生成
# ============================================================
def generate_html(payload):
    """payload (dict) を埋め込んだ index.html を返す"""
    data_json = json.dumps(payload, ensure_ascii=False, separators=(',', ':'))
    payload_json = data_json  # HTMLテンプレート内で参照

    return f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>USD/JPY カオス予測モデル</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
:root{{
  --bg:#0d0f14;--bg2:#141720;--bg3:#1c2030;--bg4:#242838;
  --border:#2a3045;--text:#e8eaf0;--text2:#9aa0b8;--text3:#6a7090;
  --blue:#4a9eff;--green:#3ddc84;--red:#ff6b6b;--amber:#f0a830;--purple:#a070ff;
  --green3:#0d4025;--amber3:#503010;--red3:#501818;
  --green2:#1a8a4a;--amber2:#a87020;--red2:#a83030;
  --blue2:#1a5fa8;--purple2:#6040c0;
}}
body{{background:var(--bg);color:var(--text);font-family:'Segoe UI',system-ui,sans-serif;font-size:14px;line-height:1.6}}
.wrap{{max-width:1100px;margin:0 auto;padding:20px 16px}}
h1{{font-size:19px;font-weight:600;margin-bottom:4px}}
.sub{{font-size:12px;color:var(--text3);margin-bottom:20px}}
.badge{{display:inline-flex;align-items:center;gap:5px;padding:5px 12px;border-radius:16px;font-size:12px;font-weight:600;float:right}}
.r0{{background:var(--green3);color:var(--green);border:1px solid var(--green2)}}
.r1{{background:var(--amber3);color:var(--amber);border:1px solid var(--amber2)}}
.r2{{background:var(--red3);color:var(--red);border:1px solid var(--red2)}}
.dot{{width:6px;height:6px;border-radius:50%;background:currentColor}}
.metrics{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-bottom:20px}}
.m{{background:var(--bg2);border:1px solid var(--border);border-radius:8px;padding:14px 16px}}
.ml{{font-size:11px;color:var(--text3);text-transform:uppercase;letter-spacing:.5px;margin-bottom:5px}}
.mv{{font-size:22px;font-weight:600;line-height:1.1}}
.ms{{font-size:11px;color:var(--text3);margin-top:3px}}
.tabs{{display:flex;gap:2px;border-bottom:1px solid var(--border);margin-bottom:20px;overflow-x:auto}}
.tab{{padding:9px 16px;font-size:13px;background:none;border:none;cursor:pointer;color:var(--text3);border-bottom:2px solid transparent;margin-bottom:-1px;white-space:nowrap}}
.tab.on{{color:var(--blue);border-bottom-color:var(--blue);font-weight:500}}
.pn{{display:none}}.pn.on{{display:block}}
.card{{background:var(--bg2);border:1px solid var(--border);border-radius:10px;padding:20px;margin-bottom:16px}}
.ct{{font-size:13px;font-weight:500;color:var(--text2);margin-bottom:14px}}
.cw{{position:relative;width:100%}}
.h280{{height:280px}}.h240{{height:240px}}.h200{{height:200px}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th{{text-align:left;padding:8px 10px;font-size:11px;color:var(--text3);text-transform:uppercase;letter-spacing:.4px;border-bottom:1px solid var(--border)}}
td{{padding:9px 10px;border-bottom:1px solid var(--border)}}
tr:last-child td{{border-bottom:none}}
tr:hover td{{background:var(--bg3)}}
.bar-bg{{flex:1;height:6px;background:var(--bg4);border-radius:3px;position:relative;min-width:60px}}
.bar-f{{position:absolute;top:0;height:6px;border-radius:3px;background:var(--blue2)}}
.bar-m{{position:absolute;top:-2px;width:3px;height:10px;border-radius:1px;background:var(--blue)}}
.rb{{height:6px;border-radius:3px}}
.info{{background:var(--bg3);border-left:3px solid var(--blue2);border-radius:0 8px 8px 0;padding:10px 14px;font-size:12px;color:var(--text3);line-height:1.6;margin-top:12px}}
.info strong{{color:var(--text2)}}
.warn{{background:var(--amber3);border:1px solid var(--amber2);border-radius:8px;padding:10px 14px;font-size:12px;color:var(--amber);margin-bottom:16px}}
.upd{{font-size:11px;color:var(--text3);text-align:right;margin-top:20px}}
.opt-row{{display:grid;grid-template-columns:130px 1fr 90px;gap:10px;align-items:center;margin-bottom:6px}}
</style>
</head>
<body>
<div class="wrap">
<div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:4px;flex-wrap:wrap;gap:8px">
  <div>
    <h1>USD/JPY カオス理論アンサンブル予測</h1>
    <div class="sub">レジーム検出 + 予測分布 + 自動最適化アンサンブル | データ: {payload['last_date']}</div>
  </div>
  <div id="rbadge" class="badge r{payload['current_regime']}"><span class="dot"></span><span id="rlabel"></span></div>
</div>
<div class="warn">⚠ 研究・学習目的のモデルです。実際の取引判断には使用しないでください。</div>
<div class="metrics" id="mg"></div>
<div class="tabs">
  <button class="tab on" onclick="sw(0)">予測分布</button>
  <button class="tab" onclick="sw(1)">バックテスト</button>
  <button class="tab" onclick="sw(2)">レジーム履歴</button>
  <button class="tab" onclick="sw(3)">モデル詳細</button>
</div>
<div class="pn on" id="p0">
  <div class="card">
    <div class="ct">5日間予測（直近20日 + 予測帯）</div>
    <div class="cw h280"><canvas id="c0"></canvas></div>
  </div>
  <div class="card">
    <div class="ct">日別予測詳細</div>
    <table id="t0"></table>
  </div>
  <div class="info"><strong>読み方</strong>：青帯は80%信頼区間。現レジーム（<span id="rl2"></span>）に応じて帯幅が自動調整されます。</div>
</div>
<div class="pn" id="p1">
  <div class="card">
    <div class="ct">直近{payload['bt_n']}日バックテスト（自動最適化重みで計算）</div>
    <div id="btr"></div>
  </div>
  <div class="card">
    <div class="ct">残差時系列（直近60日）</div>
    <div class="cw h240"><canvas id="c1"></canvas></div>
    <div class="info"><strong>自動重み最適化</strong>：毎日バックテストを実行し、3モデルの重みをRMSE最小化グリッドサーチで自動更新します。市場環境の変化（レジーム遷移）に追従して重みが変わります。</div>
  </div>
</div>
<div class="pn" id="p2">
  <div class="card">
    <div class="ct">直近60日 レジームタイムライン</div>
    <div id="rtl" style="display:flex;gap:2px;height:28px;border-radius:6px;overflow:hidden;margin-bottom:8px"></div>
    <div style="display:flex;gap:16px;font-size:11px;color:var(--text3)">
      <span>🟢 低ボラ安定</span><span>🟡 中ボラ</span><span>🔴 高ボラ不安定</span>
    </div>
  </div>
  <div class="card">
    <div class="ct">ボラティリティ + 価格</div>
    <div class="cw h240"><canvas id="c2"></canvas></div>
    <div class="info"><strong>レジーム閾値</strong>：低ボラ &lt; {payload['vol_thresholds'][0]*100:.1f}% ≤ 中ボラ &lt; {payload['vol_thresholds'][1]*100:.1f}% ≤ 高ボラ（年率換算）</div>
  </div>
</div>
<div class="pn" id="p3">
  <div class="card">
    <div class="ct">自動最適化されたアンサンブル重み</div>
    <div id="wrows"></div>
    <div class="info" style="margin-top:16px">
      <strong>重みの自動調整ロジック</strong>：直近{payload['bt_n']}日の予測誤差を0.1刻みのグリッドサーチで最小化。レジームが変わると最適重みも変化します。今日の最適解 → カオス: {payload['weights']['chaos']*100:.0f}% / AR: {payload['weights']['ar']*100:.0f}% / モメンタム: {payload['weights']['momentum']*100:.0f}%
    </div>
  </div>
  <div class="card">
    <div class="ct">モデル別 1日先予測</div>
    <div class="cw h200"><canvas id="c3"></canvas></div>
  </div>
</div>
<div class="upd">自動更新: 毎週月〜金 日本時間 翌1時 | GitHub Actions + yfinance</div>
</div>
<script>
const D={payload_json};
const RL=['低ボラ安定レジーム','中ボラ レジーム','高ボラ 不安定レジーム'];
const RC=['r0','r1','r2'];
const RCOL=['#3ddc84','#f0a830','#ff6b6b'];
document.getElementById('rlabel').textContent=RL[D.current_regime];
document.getElementById('rl2').textContent=RL[D.current_regime];
// metrics
const mg=document.getElementById('mg');
const mk=(label,val,sub,vc)=>`<div class="m"><div class="ml">${{label}}</div><div class="mv" style="color:var(--${{vc}})">${{val}}</div><div class="ms">${{sub}}</div></div>`;
mg.innerHTML=
  mk('現在レート','¥'+D.last_price.toFixed(2),D.last_date,'blue')+
  mk('1日先予測（中央値）','¥'+D.ens[0].mean.toFixed(2),'['+D.ens[0].p10.toFixed(2)+', '+D.ens[0].p90.toFixed(2)+']','green')+
  mk('ボラティリティ（年率）',(D.current_vol*100).toFixed(2)+'%','現在レジーム: '+RL[D.current_regime],'amber')+
  mk('AR 方向的中率',(D.bt.ar.da*100).toFixed(0)+'%','直近'+D.bt_n+'日BT','purple')+
  mk('アンサンブル RMSE',D.bt.ensemble.rmse.toFixed(4),'自動最適化後','blue');
// tab
const tabs=document.querySelectorAll('.tab');
const panels=document.querySelectorAll('.pn');
function sw(i){{tabs.forEach((t,j)=>t.classList.toggle('on',i===j));panels.forEach((p,j)=>p.classList.toggle('on',i===j));}}
// --- chart 0: prediction ---
const lp=D.last20_prices, ld=D.last20_dates;
const pLabels=[...ld.map(d=>d.slice(5)),'+1','+2','+3','+4','+5'];
const nL=lp.length;
const actData=[...lp,...Array(5).fill(null)];
const ensM=[...Array(nL-1).fill(null),D.last_price,...D.ens.map(p=>p.mean)];
const ensP10=[...Array(nL-1).fill(null),D.last_price,...D.ens.map(p=>p.p10)];
const ensP90=[...Array(nL-1).fill(null),D.last_price,...D.ens.map(p=>p.p90)];
const arM=[...Array(nL-1).fill(null),D.last_price,...D.ar_preds];
const chM=[...Array(nL-1).fill(null),D.last_price,...D.chaos_preds.map(p=>p.mean)];
const ymin=Math.min(...lp,...D.ens.map(p=>p.p10))-1;
const ymax=Math.max(...lp,...D.ens.map(p=>p.p90))+1;
new Chart(document.getElementById('c0'),{{
  type:'line',data:{{labels:pLabels,datasets:[
    {{data:ensP90,borderWidth:0,pointRadius:0,fill:'+1',backgroundColor:'rgba(74,158,255,0.13)',tension:.3,spanGaps:true}},
    {{data:ensP10,borderWidth:0,pointRadius:0,fill:false,tension:.3,spanGaps:true}},
    {{label:'アンサンブル',data:ensM,borderColor:'#4a9eff',borderWidth:2.5,pointRadius:3,fill:false,tension:.3,spanGaps:true}},
    {{label:'AR(5)',data:arM,borderColor:'rgba(240,168,48,.7)',borderWidth:1.5,pointRadius:2,borderDash:[3,2],fill:false,tension:.3,spanGaps:true}},
    {{label:'カオスk-NN',data:chM,borderColor:'rgba(160,112,255,.7)',borderWidth:1.5,pointRadius:2,borderDash:[4,3],fill:false,tension:.3,spanGaps:true}},
    {{label:'実績',data:actData,borderColor:'#3ddc84',borderWidth:2,pointRadius:3,fill:false,tension:.3}}
  ]}},
  options:{{responsive:true,maintainAspectRatio:false,
    plugins:{{legend:{{display:false}},tooltip:{{mode:'index',intersect:false,callbacks:{{label:ctx=>ctx.parsed.y!=null?ctx.dataset.label+': ¥'+ctx.parsed.y.toFixed(2):''}}}}}},
    scales:{{x:{{grid:{{color:'rgba(42,48,69,.8)'}},ticks:{{color:'#6a7090',maxRotation:45,autoSkip:false,maxTicksLimit:15}}}},y:{{grid:{{color:'rgba(42,48,69,.8)'}},ticks:{{color:'#6a7090',callback:v=>'¥'+v.toFixed(1)}},min:ymin,max:ymax}}}}
  }}
}});
// pred table
const days=['+1日','+2日','+3日','+4日','+5日'];
const pMin=ymin,pMax=ymax;
let th='<tr><th>日</th><th>中央値</th><th>80%信頼区間</th><th>カオス</th><th>AR(5)</th><th>モメンタム</th></tr>';
D.ens.forEach((p,i)=>{{
  const pct=(p.mean-pMin)/(pMax-pMin)*100;
  const bw=(p.p90-p.p10)/(pMax-pMin)*100;
  const d=p.mean>D.last_price?'↑':p.mean<D.last_price?'↓':'→';
  const dc=p.mean>D.last_price?'green':p.mean<D.last_price?'red':'amber';
  th+=`<tr><td>${{days[i]}}</td><td style="color:var(--${{dc}});font-weight:600">${{d}} ${{p.mean.toFixed(2)}}</td>
    <td><div style="display:flex;align-items:center;gap:6px;font-size:12px">
      <span style="color:var(--text3);min-width:44px">${{p.p10.toFixed(1)}}</span>
      <div class="bar-bg"><div class="bar-f" style="left:${{Math.max(0,pct-bw/2)}}%;width:${{bw}}%"></div><div class="bar-m" style="left:${{pct}}%"></div></div>
      <span style="color:var(--text3);min-width:44px;text-align:right">${{p.p90.toFixed(1)}}</span>
    </div></td>
    <td style="color:var(--purple)">${{p.chaos.toFixed(2)}}</td>
    <td style="color:var(--amber)">${{p.ar.toFixed(2)}}</td>
    <td style="color:var(--text2)">${{p.momentum.toFixed(2)}}</td></tr>`;
}});
document.getElementById('t0').innerHTML=th;
// backtest rows
const btModels=[
  {{k:'ar',name:'AR(5)',note:'通常最強',c:'#f0a830'}},
  {{k:'ensemble',name:'アンサンブル',note:'自動最適化',c:'#3ddc84'}},
  {{k:'momentum',name:'モメンタム',note:'トレンド追従',c:'#4a9eff'}},
  {{k:'chaos',name:'カオス k-NN',note:'位相空間',c:'#a070ff'}},
];
const maxR=Math.max(...btModels.map(m=>D.bt[m.k]?.rmse||0))*1.1;
let bth='<div style="display:grid;grid-template-columns:130px 1fr 80px 80px 80px;gap:10px;padding:8px 0;border-bottom:1px solid var(--border)"><span style="font-size:11px;color:var(--text3)">モデル</span><span></span><span style="font-size:11px;color:var(--text3);text-align:right">RMSE</span><span style="font-size:11px;color:var(--text3);text-align:right">MAE</span><span style="font-size:11px;color:var(--text3);text-align:right">方向的中率</span></div>';
btModels.forEach(m=>{{
  const b=D.bt[m.k]||{{}};
  const bw=b.rmse?Math.min(100,b.rmse/maxR*100):0;
  bth+=`<div style="display:grid;grid-template-columns:130px 1fr 80px 80px 80px;gap:10px;align-items:center;padding:12px 0;border-bottom:1px solid var(--border)">
    <div style="color:${{m.c}};font-size:13px;font-weight:500">${{m.name}}<br><span style="font-size:11px;color:var(--text3)">${{m.note}}</span></div>
    <div><div style="height:8px;background:var(--bg4);border-radius:4px"><div class="rb" style="width:${{bw}}%;background:${{m.c}}"></div></div></div>
    <div style="text-align:right;font-size:13px;color:${{m.c}}">${{b.rmse!=null?b.rmse.toFixed(4):'—'}}</div>
    <div style="text-align:right;font-size:13px;color:${{m.c}}">${{b.mae!=null?b.mae.toFixed(4):'—'}}</div>
    <div style="text-align:right;font-size:13px;color:${{m.c}}">${{b.da!=null?(b.da*100).toFixed(0)+'%':'—'}}</div>
  </div>`;
}});
document.getElementById('btr').innerHTML=bth;
// residual chart
const rh=D.regime_history.slice(-60);
const rLabels=rh.map(r=>r.date.slice(5));
const resEns=D.errors_ens_60||[];
new Chart(document.getElementById('c1'),{{
  type:'bar',data:{{labels:rLabels,datasets:[
    {{label:'アンサンブル残差',data:resEns,backgroundColor:resEns.map(v=>v>=0?'rgba(74,158,255,.6)':'rgba(255,107,107,.5)'),borderWidth:0,borderRadius:2}}
  ]}},
  options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{display:false}}}},
    scales:{{x:{{grid:{{color:'rgba(42,48,69,.5)'}},ticks:{{color:'#6a7090',maxTicksLimit:10,maxRotation:45}}}},y:{{grid:{{color:'rgba(42,48,69,.5)'}},ticks:{{color:'#6a7090'}},title:{{display:true,text:'残差（円）',color:'#6a7090'}}}}}}
  }}
}});
// regime timeline
const rtl=document.getElementById('rtl');
const regCols=['#3ddc84','#f0a830','#ff6b6b'];
rh.forEach(r=>{{
  const s=document.createElement('div');
  s.style.cssText=`flex:1;background:${{regCols[r.regime]}};opacity:.7`;
  s.title=`${{r.date}} ¥${{r.price}} | ${{RL[r.regime]}}`;
  rtl.appendChild(s);
}});
// regime chart
new Chart(document.getElementById('c2'),{{
  type:'line',data:{{labels:rh.map(r=>r.date.slice(5)),datasets:[
    {{label:'価格',data:rh.map(r=>r.price),borderColor:'#4a9eff',borderWidth:2,pointRadius:1,fill:false,tension:.3,yAxisID:'y'}},
    {{label:'ボラ年率',data:rh.map(r=>parseFloat((r.vol*100).toFixed(2))),borderColor:'#f0a830',borderWidth:1.5,pointRadius:1,borderDash:[3,2],fill:false,tension:.3,yAxisID:'y2'}}
  ]}},
  options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{display:false}}}},
    scales:{{
      x:{{grid:{{color:'rgba(42,48,69,.5)'}},ticks:{{color:'#6a7090',maxTicksLimit:10,maxRotation:45}}}},
      y:{{grid:{{color:'rgba(42,48,69,.5)'}},ticks:{{color:'#4a9eff',callback:v=>'¥'+vType}},position:'left'}},
      y2:{{grid:{{display:false}},ticks:{{color:'#f0a830',callback:v=>v+'%'}},position:'right',min:0}}
    }}
  }}
}});
// weight rows
const wk=[['chaos','カオス k-NN','purple'],['ar','AR(5)','amber'],['momentum','モメンタム','blue']];
let wh='';
wk.forEach(([k,name,c])=>{{
  const pct=Math.round(D.weights[k]*100);
  wh+=`<div class="opt-row" style="margin-bottom:10px">
    <div style="font-size:13px;color:var(--${{c}})">${{name}}</div>
    <div style="height:8px;background:var(--bg4);border-radius:4px"><div class="rb" style="width:${{pct}}%;background:var(--${{c}}2)"></div></div>
    <div style="text-align:right;font-size:14px;font-weight:600;color:var(--${{c}})">${{pct}}%</div>
  </div>`;
}});
document.getElementById('wrows').innerHTML=wh;
// model compare
new Chart(document.getElementById('c3'),{{
  type:'bar',data:{{
    labels:['カオス k-NN','AR(5)','モメンタム','アンサンブル'],
    datasets:[{{data:[D.chaos_preds[0].mean,D.ar_preds[0],D.mom_preds[0],D.ens[0].mean],
      backgroundColor:['rgba(160,112,255,.7)','rgba(240,168,48,.7)','rgba(74,158,255,.7)','rgba(61,220,132,.8)'],
      borderColor:['#a070ff','#f0a830','#4a9eff','#3ddc84'],borderWidth:1.5,borderRadius:4}}]
  }},
  options:{{responsive:true,maintainAspectRatio:false,
    plugins:{{legend:{{display:false}},tooltip:{{callbacks:{{label:ctx=>'¥'+ctx.parsed.y.toFixed(2)}}}}}},
    scales:{{x:{{grid:{{color:'rgba(42,48,69,.8)'}},ticks:{{color:'#6a7090'}}}},y:{{grid:{{color:'rgba(42,48,69,.8)'}},ticks:{{color:'#6a7090',callback:v=>'¥'+v.toFixed(1)}}}}}}
  }}
}});
</script>
</body>
</html>"""

# ============================================================
# 8. メイン処理
# ============================================================
def main():
    print("=== USD/JPY カオス予測モデル 更新開始 ===")
    print("実行時刻:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    # データ取得
    dates, prices = fetch_usdjpy()
    N = len(prices)

    # 特徴量
    roll_vol, momentum, regimes, vq33, vq67 = calc_features(prices)
    current_regime = int(regimes[-1])
    current_vol    = float(roll_vol[-1])
    print(f"現在レジーム: {current_regime} | ボラ: {current_vol*100:.2f}%年率")

    # バックテスト最適化（毎回実行）
    print("バックテスト最適化中...")
    BT_N = 200
    weights, bt_stats = backtest_optimize(prices, momentum, regimes, n_bt=BT_N)
    print(f"最適重み: chaos={weights['chaos']:.1f} ar={weights['ar']:.1f} mom={weights['momentum']:.1f}")
    print(f"RMSE: chaos={bt_stats['chaos']['rmse']:.4f} ar={bt_stats['ar']['rmse']:.4f} ens={bt_stats['ensemble']['rmse']:.4f}")

    # 予測
    chaos_preds = predict_chaos(prices, momentum, regimes)
    ar_preds    = predict_ar(prices)
    mom_preds   = predict_momentum(prices)
    ens_preds = ensemble_predict(chaos_preds, ar_preds, mom_preds, weights, current_regime)

    # 残差（直近60日） (修正箇所: predict_chaos から数値を正しく取得)
    errors_ens_60 = []
    for i in range(60, 0, -1):
        idx = N - i
        if idx < (DIM-1)*TAU + 50:
            continue
        cp_dict = predict_chaos(prices[:idx], momentum[:idx], regimes[:idx], steps=1)
        cp = cp_dict[0]["mean"] if cp_dict else prices[idx-1]
        
        ap = predict_ar(prices[:idx], steps=1)[0]
        mp = predict_momentum(prices[:idx], steps=1)[0]
        ens_e = weights["chaos"]*cp + weights["ar"]*ap + weights["momentum"]*mp
        errors_ens_60.append(round(prices[idx] - ens_e, 4))

    # レジーム履歴（直近60日）
    regime_history = [
        {"date": dates[i], "price": float(prices[i]),
          "regime": int(regimes[i]), "vol": float(roll_vol[i])}
        for i in range(N-60, N)
    ]

    # ペイロード組み立て
    payload = {
        "last_price":     float(prices[-1]),
        "last_date":      dates[-1],
        "current_regime": current_regime,
        "current_vol":    current_vol,
        "vol_thresholds": [float(vq33), float(vq67)],
        "weights":        weights,
        "bt_n":           BT_N,
        "bt":             bt_stats,
        "ens":            ens_preds,
        "chaos_preds":    chaos_preds,
        "ar_preds":       [round(v,4) for v in ar_preds],
        "mom_preds":      [round(v,4) for v in mom_preds],
        "last20_dates":   dates[-20:],
        "last20_prices":  [float(v) for v in prices[-20:]],
        "regime_history": regime_history,
        "errors_ens_60":  errors_ens_60,
        "generated_at":   datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    # HTML 生成
    os.makedirs("docs", exist_ok=True)
    html = generate_html(payload)
    with open("docs/index.html", "w", encoding="utf-8") as f:
        f.write(html)
    print(f"docs/index.html 生成完了 ({len(html):,} bytes)")
    print("=== 完了 ===")

if __name__ == "__main__":
    main()

"""
ORBIT — Regime-Adaptive Analog Ensemble (Attractor Co., Ltd.)
USD/JPY アンサンブル予測モデル v3 — 方法論改善版

v2からの主な改善(すべて日次自動更新の運用前提を維持):
  1. look-ahead bias の完全排除
     - レジーム判定のボラ中央値: 全期間 → 過去252日ローリング中央値
     - 埋め込み座標の正規化: 全期間平均/分散 → 過去252日ローリングzスコア
  2. ナイーブ(ランダムウォーク)ベンチマークを4番目のモデルとして追加
     - 全統計をナイーブ比で表示、Diebold-Mariano検定でスキルの有意性を検証
  3. 重み最適化の walk-forward 化
     - 各評価日ごとに「その日より前の150日」の誤差だけで重みを選択
     - 表示される性能は完全アウトオブサンプル(直近250日)
  4. 検証モデルと本番モデルの完全統一(同一関数を使用)
  5. 予測区間を conformal 方式で校正
     - OOS残差の経験分位点から P10-P90 を構成、カバレッジを実測表示
  6. AR(5) を対数リターンで推定(価格水準の単位根問題を回避)
  7. スプレッド込み簡易PnLバックテスト
  8. 埋め込み遅延τを自己相関ゼロ交差から日次推定
  9. 上昇確率をOOS残差分布ベースに変更(15近傍の投票 → 250サンプル)

依存: numpy yfinance
"""

import numpy as np
import json, os, math
from datetime import datetime, timezone

# ============================================================
# 設定
# ============================================================
DIM        = 3      # 遅延埋め込み次元(+ momentum, vol で計5次元)
K_NN       = 15     # k-NN近傍数
W_NORM     = 252    # 因果正規化のローリング窓
W_VOLMED   = 252    # レジーム判定ボラ中央値のローリング窓
SEL_N      = 150    # 重み選択窓(walk-forward)
EVAL_N     = 250    # OOS評価窓
HORIZONS   = 5      # 予測ホライズン(日)
PIP_JPY      = 0.01                       # USD/JPY の 1pip
SPREAD_PIPS  = 0.2                        # 実質往復コスト(pips)
COST_ONEWAY  = SPREAD_PIPS * PIP_JPY / 2  # 片道コスト(円/USD) = 0.001
NOTIONAL_YEN = 1_000_000                  # PnL想定元本(円)
PNL_THRESH   = 0.02   # 予測変化がこの閾値(円)未満なら見送り
MIN_PRICE, MAX_PRICE = 50.0, 400.0  # データ健全性チェック

REGIME_NAMES = ["Low Vol Range", "High Vol Range",
                "Bull Trend", "Bear Trend", "Unstable"]

# ============================================================
# 1. データ取得
# ============================================================
def fetch_usdjpy():
    try:
        import yfinance as yf
        df = yf.download("USDJPY=X", period="10y", interval="1d",
                         progress=False, auto_adjust=True)
        df = df.dropna()
        if len(df) < 800:
            raise ValueError("データ不足")
        closes = df["Close"].values.flatten().astype(float)
        dates  = [d.strftime("%Y/%m/%d") for d in df.index]
        _sanity_check(closes)
        print(f"[OK] yfinance: {len(closes)} 件 ({dates[0]} ~ {dates[-1]})")
        return dates, closes, "yfinance"
    except Exception as e:
        print(f"[WARN] yfinance 失敗: {e}")
        d, c = fetch_usdjpy_stooq()
        return d, c, "stooq"

def fetch_usdjpy_stooq():
    import urllib.request, csv, io
    url = "https://stooq.com/q/d/l/?s=usdjpy&i=d"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as r:
        raw = r.read().decode("utf-8")
    rows = [r for r in csv.reader(io.StringIO(raw))
            if len(r) >= 5 and r[4] not in ("", "null", "Close")]
    rows.sort(key=lambda r: r[0])
    dates  = [r[0].replace("-", "/") for r in rows]
    closes = np.array([float(r[4]) for r in rows])
    _sanity_check(closes)
    print(f"[OK] stooq: {len(closes)} 件 ({dates[0]} ~ {dates[-1]})")
    return dates, closes

def _sanity_check(closes):
    """価格レンジと日次変化の健全性チェック(ソース混在・異常値の検出)"""
    if np.any(closes < MIN_PRICE) or np.any(closes > MAX_PRICE):
        raise ValueError("価格が想定レンジ外")
    jumps = np.abs(np.diff(np.log(closes)))
    if np.any(jumps > 0.10):
        raise ValueError(f"日次10%超の異常ジャンプを検出 (max={jumps.max():.3f})")

# ============================================================
# 2. 特徴量(すべて因果: 時点tの値は t 以前のデータのみ使用)
# ============================================================
def rolling_vol(prices, w=20):
    ret = np.diff(np.log(prices))
    rv = np.array([np.std(ret[max(0, i - w):i]) * np.sqrt(252)
                   for i in range(1, len(ret) + 1)])
    return np.concatenate([[rv[0]], rv])

def calc_adx(prices, period=14):
    """終値ベース簡易ADX(トレーリング窓のみ使用)"""
    N = len(prices)
    tr   = np.abs(np.diff(prices))
    tr   = np.concatenate([[tr[0]], tr])
    up   = np.concatenate([[0], np.where(np.diff(prices) > 0, np.diff(prices), 0)])
    down = np.concatenate([[0], np.where(-np.diff(prices) > 0, -np.diff(prices), 0)])
    atr  = np.array([np.mean(tr[max(0, i - period):i + 1]) for i in range(N)])
    pdi  = np.array([np.mean(up[max(0, i - period):i + 1]) for i in range(N)]) / (atr + 1e-8) * 100
    mdi  = np.array([np.mean(down[max(0, i - period):i + 1]) for i in range(N)]) / (atr + 1e-8) * 100
    dx   = np.abs(pdi - mdi) / (pdi + mdi + 1e-8) * 100
    adx  = np.array([np.mean(dx[max(0, i - period):i + 1]) for i in range(N)])
    return adx, pdi, mdi

def calc_hurst(prices, window=120):
    """R/S法 Hurst指数(ローリング、因果)"""
    N = len(prices)
    out = np.full(N, 0.5)
    ns_list = [10, 20, 40, 80]
    for i in range(window, N):
        seg = prices[i - window:i]
        ret = np.diff(np.log(seg))
        ns, rs = [], []
        for n in ns_list:
            if n > len(ret) // 2:
                continue
            chunks = len(ret) // n
            rvs = []
            for c in range(chunks):
                s = ret[c * n:(c + 1) * n]
                dev = np.cumsum(s - s.mean())
                S = np.std(s, ddof=1)
                if S > 0:
                    rvs.append((np.max(dev) - np.min(dev)) / S)
            if rvs:
                ns.append(np.log(n))
                rs.append(np.log(np.mean(rvs)))
        if len(ns) >= 2:
            out[i] = float(np.clip(np.polyfit(ns, rs, 1)[0], 0.1, 0.9))
    return out

def calc_momentum(prices, w_short=20, w_long=60):
    N = len(prices)
    ma_s = np.array([np.mean(prices[max(0, i - w_short):i + 1]) for i in range(N)])
    ma_l = np.array([np.mean(prices[max(0, i - w_long):i + 1]) for i in range(N)])
    return (ma_s - ma_l) / (ma_l + 1e-8) * 100

def rolling_zscore(x, w=W_NORM, min_p=60):
    """因果ローリングzスコア: 各時点の値をその時点までの直近w日で正規化"""
    n = len(x)
    cs  = np.concatenate([[0.0], np.cumsum(x)])
    cs2 = np.concatenate([[0.0], np.cumsum(x * x)])
    out = np.zeros(n)
    for i in range(n):
        lo = max(0, i - w + 1)
        cnt = i - lo + 1
        if cnt < min_p:
            lo, cnt = 0, i + 1
        m = (cs[i + 1] - cs[lo]) / cnt
        v = (cs2[i + 1] - cs2[lo]) / cnt - m * m
        out[i] = (x[i] - m) / math.sqrt(max(v, 1e-12))
    return out

def rolling_median(x, w=W_VOLMED, min_p=60):
    """因果ローリング中央値"""
    n = len(x)
    out = np.empty(n)
    for i in range(n):
        lo = max(0, i - w + 1)
        if i + 1 < min_p:
            lo = 0
        out[i] = np.median(x[lo:i + 1])
    return out

def classify_regime(rv, adx, pdi, mdi, hurst):
    """5分類レジーム。ボラ閾値は因果ローリング中央値(v2は全期間percentileでリーク)"""
    N = len(rv)
    vol_med = rolling_median(rv)
    regimes = np.zeros(N, dtype=int)
    for i in range(N):
        trending = adx[i] > 25 and hurst[i] > 0.55
        high_vol = rv[i] > vol_med[i] * 1.5
        if trending and pdi[i] > mdi[i]:
            regimes[i] = 2
        elif trending and mdi[i] > pdi[i]:
            regimes[i] = 3
        elif high_vol and not trending:
            regimes[i] = 4
        elif rv[i] < vol_med[i]:
            regimes[i] = 0
        else:
            regimes[i] = 1
    return regimes, vol_med

def calc_transition_prob(regimes, lookback=252, n=5):
    mat = np.zeros((n, n))
    recent = regimes[-lookback:]
    for i in range(len(recent) - 1):
        mat[recent[i], recent[i + 1]] += 1
    rs = mat.sum(axis=1, keepdims=True)
    return mat / (rs + 1e-8)

# ============================================================
# 3. 位相空間再構成(因果)
# ============================================================
def estimate_tau(x, max_lag=60, lo=5, hi=40, default=20):
    """自己相関の最初のゼロ交差からτを推定(直近1000日)"""
    s = np.asarray(x[-1000:], dtype=float)
    s = s - s.mean()
    denom = float(np.dot(s, s))
    if denom <= 0:
        return default
    for lag in range(1, min(max_lag, len(s) - 1)):
        ac = float(np.dot(s[:-lag], s[lag:])) / denom
        if ac <= 0:
            return int(np.clip(lag, lo, hi))
    return default

def build_embedding(pn, mn, vn, tau, dim=DIM):
    """
    5次元拡張状態ベクトル: [pn(t), pn(t-τ), pn(t-2τ), mn(t), vn(t)]
    Xe[t] は価格インデックス offset+t に対応(過去方向の遅延なので完全に因果)
    """
    offset = (dim - 1) * tau
    idxs = np.arange(offset, len(pn))
    cols = [pn[idxs - j * tau] for j in range(dim)] + [mn[idxs], vn[idxs]]
    return np.column_stack(cols), offset

# ============================================================
# 4. 予測モデル(検証・本番で同一関数を使用)
# ============================================================
def chaos_forecast(Xe, prices, regimes, offset, idx, horizons=HORIZONS,
                   k=K_NN, theiler=None):
    """
    位相空間k-NN。時点idxまでの情報のみ使用。
    theiler: 時間的に近接した(窓が重複する)ベクトルの除外幅
    予測は近傍の「相対変化率」を現在価格に適用する
    (絶対水準の平均だと価格水準が異なる過去近傍で破綻するため)
    返り値: 各ホライズンの予測値リスト
    """
    t_e = idx - offset
    if t_e < k + 50:
        return [float(prices[idx])] * horizons
    if theiler is None:
        theiler = offset  # 埋め込み窓の幅ぶん除外
    cur = Xe[t_e]
    cand_e = Xe[:t_e]
    dists = np.linalg.norm(cand_e - cur, axis=1)
    pen = np.where(regimes[offset:offset + t_e] == regimes[idx], 1.0, 2.0)
    adj = dists * pen
    adj[max(0, t_e - theiler):] = np.inf
    order = np.argsort(adj)
    out = []
    for h in range(1, horizons + 1):
        # 近傍jの h日先 prices[offset+j+h] が時点idxで既知である近傍のみ
        sel = [j for j in order[:k * 4] if j + h <= t_e and np.isfinite(adj[j])][:k]
        if not sel:
            out.append(float(prices[idx]))
            continue
        sel = np.array(sel)
        rel = prices[offset + sel + h] / prices[offset + sel]  # 近傍のh日先相対変化
        ws = 1.0 / (adj[sel] + 1e-8)
        ws /= ws.sum()
        out.append(float(prices[idx] * np.dot(rel, ws)))
    return out

def ar_forecast(prices, idx, p=5, steps=HORIZONS, lookback=300):
    """AR(p) を対数リターンで推定(v2の価格水準ARは単位根近傍で不安定)"""
    seg = prices[max(0, idx - lookback):idx + 1]
    r = np.diff(np.log(seg))
    if len(r) < p + 30:
        return [float(prices[idx])] * steps
    X = np.array([r[i:i + p] for i in range(len(r) - p)])
    A = np.column_stack([X, np.ones(len(X))])
    coef, _, _, _ = np.linalg.lstsq(A, r[p:], rcond=None)
    hist = list(r[-p:])
    out, last = [], float(prices[idx])
    for _ in range(steps):
        rh = float(np.dot(coef, hist[-p:] + [1.0]))
        rh = float(np.clip(rh, -0.05, 0.05))
        last *= math.exp(rh)
        out.append(last)
        hist.append(rh)
    return out

def momentum_forecast(prices, idx, steps=HORIZONS):
    """短期・長期ドリフト + 平均回帰の合成"""
    if idx < 210:
        return [float(prices[idx])] * steps
    st = (prices[idx] - prices[idx - 3]) / 3
    lt = (prices[idx] - prices[idx - 10]) / 10
    drift = 0.5 * st + 0.5 * lt + (np.mean(prices[idx - 199:idx + 1]) - prices[idx]) * 0.01
    out, last = [], float(prices[idx])
    for s in range(1, steps + 1):
        last += drift * (0.9 ** s)
        out.append(float(last))
    return out

def naive_forecast(prices, idx, steps=HORIZONS):
    """ランダムウォークベンチマーク: 明日=今日"""
    return [float(prices[idx])] * steps

MODEL_KEYS = ["chaos", "ar", "momentum", "naive"]

# ============================================================
# 5. walk-forward 検証エンジン
# ============================================================
def _weight_combos(n_models=4, step=10):
    combos = []
    for a in range(step + 1):
        for b in range(step + 1 - a):
            for c in range(step + 1 - a - b):
                d = step - a - b - c
                combos.append((a / step, b / step, c / step, d / step))
    return np.array(combos)

COMBOS = _weight_combos()

def walk_forward(prices, Xe, regimes, offset):
    """
    直近 (EVAL_N + SEL_N + HORIZONS) 日について、各日 idx 時点までの情報のみで
    4モデルの1〜5日先予測を生成し、誤差行列を構築する。
    返り値: preds[m] (HORIZONS, N), errs[m] (HORIZONS, N)  ※ NaN埋め
    """
    N = len(prices)
    hist_n = EVAL_N + SEL_N + HORIZONS + 5
    start = max(offset + 400, N - hist_n)
    preds = {m: np.full((HORIZONS, N), np.nan) for m in MODEL_KEYS}
    errs  = {m: np.full((HORIZONS, N), np.nan) for m in MODEL_KEYS}
    for idx in range(start, N):
        fc = {
            "chaos":    chaos_forecast(Xe, prices, regimes, offset, idx),
            "ar":       ar_forecast(prices, idx),
            "momentum": momentum_forecast(prices, idx),
            "naive":    naive_forecast(prices, idx),
        }
        for m in MODEL_KEYS:
            for h in range(HORIZONS):
                preds[m][h, idx] = fc[m][h]
                if idx + h + 1 < N:
                    errs[m][h, idx] = prices[idx + h + 1] - fc[m][h]
    return preds, errs, start

def select_weights(errs, idx, sel_n=SEL_N):
    """idxより前のsel_n日の1日先誤差のみで重みをグリッドサーチ(RMSE最小)"""
    win = np.array([errs[m][0, idx - sel_n:idx] for m in MODEL_KEYS])
    valid = ~np.isnan(win).any(axis=0)
    win = win[:, valid]
    if win.shape[1] < 30:
        return np.array([0.0, 0.0, 0.0, 1.0]), None
    comb_err = COMBOS @ win
    rmse = np.sqrt((comb_err ** 2).mean(axis=1))
    i = int(np.argmin(rmse))
    return COMBOS[i], float(rmse[i])

def evaluate_oos(prices, preds, errs, start):
    """
    walk-forward重み選択つきOOS評価。
    各評価日の重みは「その日より前の150日」の誤差のみから決定される。
    """
    N = len(prices)
    eval_idxs = [i for i in range(max(start + SEL_N, N - EVAL_N - HORIZONS), N)
                 if not np.isnan(errs["naive"][0, i])]
    ens_pred = np.full((HORIZONS, N), np.nan)
    ens_err  = np.full((HORIZONS, N), np.nan)
    w_hist = []
    for idx in eval_idxs:
        w, _ = select_weights(errs, idx)
        w_hist.append({"idx": idx, "w": w.tolist()})
        for h in range(HORIZONS):
            p = sum(w[mi] * preds[m][h, idx] for mi, m in enumerate(MODEL_KEYS))
            ens_pred[h, idx] = p
            if idx + h + 1 < N:
                ens_err[h, idx] = prices[idx + h + 1] - p
    return ens_pred, ens_err, eval_idxs, w_hist

def _stats_at(err_row, pred_row, prices, idxs):
    """1ホライズン分の OOS 統計(da は予測変化と実変化の方向一致率)"""
    es, hits = [], []
    for i in idxs:
        if np.isnan(err_row[i]):
            continue
        es.append(err_row[i])
        pred_chg = pred_row[i] - prices[i]
        act_chg  = (pred_row[i] + err_row[i]) - prices[i]
        if abs(pred_chg) > 1e-9:
            hits.append(1 if pred_chg * act_chg > 0 else 0)
    es = np.array(es)
    if len(es) == 0:
        return None
    return {
        "rmse": float(np.sqrt(np.mean(es ** 2))),
        "mae":  float(np.mean(np.abs(es))),
        "da":   float(np.mean(hits)) if hits else None,
        "n":    int(len(es)),
    }

def dm_test(e_model, e_naive):
    """Diebold-Mariano検定(二乗誤差、h=1)。返り値: (統計量, p値)"""
    d = e_model ** 2 - e_naive ** 2
    d = d[~np.isnan(d)]
    n = len(d)
    if n < 30:
        return None, None
    dbar = d.mean()
    var = ((d - dbar) ** 2).sum() / n
    if var <= 0:
        return None, None
    stat = dbar / math.sqrt(var / n)
    stat *= math.sqrt(max(n - 1, 1) / n)  # HLN小標本補正(h=1)
    p = math.erfc(abs(stat) / math.sqrt(2))
    return float(stat), float(p)

# ============================================================
# 6. conformal 予測区間 / カバレッジ / 上昇確率
# ============================================================
def conformal_quantiles(ens_err, eval_idxs):
    """ホライズン別にOOS残差の経験分位点を取得"""
    qs = {}
    for h in range(HORIZONS):
        es = np.array([ens_err[h, i] for i in eval_idxs if not np.isnan(ens_err[h, i])])
        if len(es) < 50:
            qs[h] = None
            continue
        qs[h] = {p: float(np.percentile(es, p)) for p in (10, 25, 50, 75, 90)}
        qs[h]["_errs"] = es
    return qs

def coverage_check(ens_err, eval_idxs, holdout=60):
    """前半残差で分位点を作り、後半60日でカバレッジを実測(分位点の自己評価を回避)"""
    out = {}
    for h in range(HORIZONS):
        es = np.array([ens_err[h, i] for i in eval_idxs if not np.isnan(ens_err[h, i])])
        if len(es) < holdout + 60:
            out[f"d{h+1}"] = None
            continue
        fit, test = es[:-holdout], es[-holdout:]
        q10, q90 = np.percentile(fit, 10), np.percentile(fit, 90)
        q25, q75 = np.percentile(fit, 25), np.percentile(fit, 75)
        out[f"d{h+1}"] = {
            "cov80": round(float(np.mean((test >= q10) & (test <= q90))), 4),
            "cov50": round(float(np.mean((test >= q25) & (test <= q75))), 4),
        }
    return out

# ============================================================
# 7. スプレッド込みPnLバックテスト
# ============================================================
def pnl_backtest(prices, ens_pred, eval_idxs):
    """
    1日先予測の符号でポジション。想定元本100万円の円建てPnL。
    保有USD数量 = 100万円 / その日の価格。コストはポジション変更時に発生。
    """
    N = len(prices)
    pos_prev, pnls, trades, hits, recs = 0, [], 0, [], []
    for idx in eval_idxs:
        if idx + 1 >= N or np.isnan(ens_pred[0, idx]):
            continue
        edge = ens_pred[0, idx] - prices[idx]
        pos = 0 if abs(edge) < PNL_THRESH else (1 if edge > 0 else -1)
        units = NOTIONAL_YEN / prices[idx]  # 保有USD数量
        dp = prices[idx + 1] - prices[idx]
        cost = COST_ONEWAY * abs(pos - pos_prev) * units
        pnl = pos * units * dp - cost
        pnls.append(pnl)
        if pos != 0:
            trades += 1
            hits.append(1 if pos * dp > 0 else 0)
        recs.append({"i": idx, "pnl": round(float(pnl), 1)})
        pos_prev = pos
    pnls = np.array(pnls)
    if len(pnls) == 0:
        return None, []
    cum = np.cumsum(pnls)
    sharpe = float(pnls.mean() / (pnls.std() + 1e-12) * math.sqrt(252))
    summary = {
        "total":     round(float(cum[-1])),
        "sharpe":    round(sharpe, 3),
        "hit":       round(float(np.mean(hits)), 4) if hits else None,
        "n_days":    int(len(pnls)),
        "n_trades":  int(trades),
        "max_dd":    round(float(np.max(np.maximum.accumulate(cum) - cum))),
        "cost_pips": SPREAD_PIPS,
        "notional":  NOTIONAL_YEN,
        "thresh":    PNL_THRESH,
    }
    return summary, recs

# ============================================================
# 8. HTML生成(トークン置換方式)
# ============================================================
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ORBIT v3 — Attractor Co., Ltd.</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#050810;--bg2:#0d1117;--bg3:#141b27;--bg4:#1c2535;
  --border:#1e2d45;--border2:#2a3f5f;
  --text:#cdd9f0;--text2:#8899bb;--text3:#556688;
  --cyan:#00d4ff;--green:#00ff88;--red:#ff4466;--amber:#ffaa00;
  --purple:#aa66ff;--blue:#4488ff;
  --green3:#002211;--green2:#009955;--blue3:#001133;--blue2:#2255bb;
  --amber3:#221100;--amber2:#996600;--red3:#220011;--red2:#992233;
  --purple3:#110022;--purple2:#6633bb;
  --r:6px;
}
body{background:var(--bg);color:var(--text);font-family:'SF Mono','Consolas','Courier New',monospace;font-size:13px;line-height:1.5;min-height:100vh}
.wrap{max-width:1200px;margin:0 auto;padding:16px}
.hdr{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:20px;padding-bottom:16px;border-bottom:1px solid var(--border2)}
.hdr-l h1{font-size:15px;font-weight:400;color:var(--cyan);letter-spacing:2px;text-transform:uppercase}
.hdr-l .sub{font-size:11px;color:var(--text3);margin-top:4px;letter-spacing:1px}
.hdr-r{text-align:right;font-size:11px;color:var(--text3)}
.hdr-price{font-size:28px;font-weight:300;color:var(--cyan);letter-spacing:2px}
.rbadge{display:inline-flex;align-items:center;gap:6px;padding:4px 12px;border-radius:2px;font-size:11px;letter-spacing:1px;text-transform:uppercase;font-weight:500;border:1px solid;margin-top:6px}
.r0{background:var(--green3);color:var(--green);border-color:var(--green2)}
.r1{background:var(--blue3);color:var(--blue);border-color:var(--blue2)}
.r2{background:var(--amber3);color:var(--amber);border-color:var(--amber2)}
.r3{background:var(--red3);color:var(--red);border-color:var(--red2)}
.r4{background:var(--purple3);color:var(--purple);border-color:var(--purple2)}
.rdot{width:6px;height:6px;border-radius:50%;background:currentColor;animation:blink 2s infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.3}}
.mstrip{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:1px;background:var(--border);border:1px solid var(--border);border-radius:var(--r);overflow:hidden;margin-bottom:20px}
.mc{background:var(--bg2);padding:12px 14px}
.mc-l{font-size:10px;color:var(--text3);letter-spacing:1px;text-transform:uppercase;margin-bottom:5px}
.mc-v{font-size:18px;font-weight:300;letter-spacing:1px}
.mc-s{font-size:10px;color:var(--text3);margin-top:2px}
.tabs{display:flex;border-bottom:1px solid var(--border2);margin-bottom:20px;overflow-x:auto}
.tab{padding:8px 18px;font-size:11px;background:none;border:none;cursor:pointer;color:var(--text3);border-bottom:2px solid transparent;margin-bottom:-1px;white-space:nowrap;letter-spacing:1px;text-transform:uppercase}
.tab:hover{color:var(--text2)}
.tab.on{color:var(--cyan);border-bottom-color:var(--cyan)}
.pn{display:none}.pn.on{display:block}
.card{background:var(--bg2);border:1px solid var(--border);border-radius:var(--r);padding:18px;margin-bottom:14px}
.card-t{font-size:10px;color:var(--text3);letter-spacing:2px;text-transform:uppercase;margin-bottom:14px;display:flex;align-items:center;gap:8px}
.card-t::before{content:'';display:inline-block;width:3px;height:12px;background:var(--cyan);border-radius:1px}
.cw{position:relative;width:100%}
.h260{height:260px}.h240{height:240px}.h200{height:200px}.h180{height:180px}
.g2{display:grid;grid-template-columns:1fr 1fr;gap:14px}
@media(max-width:700px){.g2{grid-template-columns:1fr}}
table{width:100%;border-collapse:collapse;font-size:12px}
th{text-align:left;padding:7px 10px;font-size:10px;color:var(--text3);letter-spacing:1px;text-transform:uppercase;border-bottom:1px solid var(--border2)}
td{padding:9px 10px;border-bottom:1px solid var(--border);font-variant-numeric:tabular-nums}
tr:last-child td{border-bottom:none}
tr:hover td{background:var(--bg3)}
.prob-bar{height:4px;border-radius:2px;background:var(--bg4);overflow:hidden;margin-top:4px}
.rtl{display:flex;gap:2px;height:20px;border-radius:3px;overflow:hidden;margin-bottom:8px}
.rt-seg{flex:1;cursor:default;transition:opacity .15s}
.rt-seg:hover{opacity:.7}
.warn{background:var(--amber3);border:1px solid var(--amber2);border-radius:var(--r);padding:8px 12px;font-size:11px;color:var(--amber);margin-bottom:16px;letter-spacing:.5px}
.note{font-size:11px;color:var(--text3);margin-top:12px;line-height:1.7}
.foot{font-size:10px;color:var(--text3);text-align:right;margin-top:20px;letter-spacing:.5px}
body::after{content:'';position:fixed;inset:0;background:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,0,0,.03) 2px,rgba(0,0,0,.03) 4px);pointer-events:none;z-index:9999}
</style>
</head>
<body>
<div class="wrap">

<div class="hdr">
  <div class="hdr-l">
    <h1>ORBIT &nbsp;&middot;&nbsp; Regime-Adaptive Analog Ensemble</h1>
    <div class="sub">Causal Features &middot; Walk-Forward OOS &middot; Conformal Intervals &middot; vs Random Walk</div>
    <div class="sub" style="color:var(--cyan);opacity:.7;margin-top:2px">Attractor Co., Ltd.</div>
  </div>
  <div class="hdr-r">
    <div class="hdr-price" id="hprice">&mdash;</div>
    <div id="hdate" style="margin-top:4px;font-size:11px;color:var(--text3)"></div>
    <div id="rbadge" class="rbadge r0" style="margin-top:6px"><span class="rdot"></span><span id="rlabel"></span></div>
  </div>
</div>

<div class="warn">&#9888; 研究・学習目的のモデルです。実際の取引判断には使用しないでください。全性能指標は完全アウトオブサンプル(walk-forward)です。</div>

<div class="mstrip" id="mstrip"></div>

<div class="tabs">
  <button class="tab on" onclick="sw(0)">Forecast</button>
  <button class="tab" onclick="sw(1)">Regime</button>
  <button class="tab" onclick="sw(2)">Backtest (OOS)</button>
  <button class="tab" onclick="sw(3)">PnL</button>
  <button class="tab" onclick="sw(4)">Optimization</button>
</div>

<!-- TAB 0: FORECAST -->
<div class="pn on" id="p0">
  <div class="card">
    <div class="card-t">Price Forecast &mdash; 5-Day Conformal Interval (calibrated on OOS residuals)</div>
    <div class="cw h260"><canvas id="c0"></canvas></div>
  </div>
  <div class="g2">
    <div class="card">
      <div class="card-t">Daily Forecast Detail</div>
      <table id="t0"></table>
      <div class="note">区間はOOS残差の経験分位点で校正。上昇確率もOOS残差分布から算出。</div>
    </div>
    <div class="card">
      <div class="card-t">Regime Transition Probability</div>
      <div id="tmat"></div>
      <div style="margin-top:14px">
        <div class="card-t" style="margin-top:0">Up / Down Probability</div>
        <div id="updown"></div>
      </div>
    </div>
  </div>
</div>

<!-- TAB 1: REGIME -->
<div class="pn" id="p1">
  <div class="card">
    <div class="card-t">Regime Timeline &mdash; Last 60 Days (causal classification)</div>
    <div class="rtl" id="rtl"></div>
    <div style="display:flex;flex-wrap:wrap;gap:14px;font-size:10px;color:var(--text3);margin-top:6px" id="rlegend"></div>
  </div>
  <div class="card">
    <div class="card-t">Price &middot; Volatility &middot; ADX</div>
    <div class="cw h260"><canvas id="c1"></canvas></div>
  </div>
  <div class="g2">
    <div class="card">
      <div class="card-t">Hurst Exponent (last 60d)</div>
      <div class="cw h180"><canvas id="c1h"></canvas></div>
    </div>
    <div class="card">
      <div class="card-t">Regime Distribution (60d)</div>
      <div class="cw h180"><canvas id="c1r"></canvas></div>
    </div>
  </div>
</div>

<!-- TAB 2: BACKTEST -->
<div class="pn" id="p2">
  <div class="card">
    <div class="card-t">Model Performance vs Random Walk &mdash; 1-Day OOS (last __EVAL_N__d, walk-forward)</div>
    <table id="tbt"></table>
    <div class="note">
      DM p値: Diebold-Mariano検定(対ナイーブ、二乗誤差)。p&lt;0.05でナイーブとの差が有意。
      RMSE比&lt;1 がランダムウォーク超え。DA=方向一致率(予測変化が閾値未満の日は除外)。
    </div>
  </div>
  <div class="g2">
    <div class="card">
      <div class="card-t">Ensemble Multi-Step OOS</div>
      <table id="tms"></table>
    </div>
    <div class="card">
      <div class="card-t">Interval Calibration (holdout 60d)</div>
      <table id="tcov"></table>
      <div class="note">cov80はP10&ndash;P90に実績が入った割合(理想80%)、cov50はP25&ndash;P75(理想50%)。</div>
    </div>
  </div>
  <div class="card">
    <div class="card-t">Ensemble OOS Residuals (last 60d)</div>
    <div class="cw h200"><canvas id="c2b"></canvas></div>
  </div>
</div>

<!-- TAB 3: PNL -->
<div class="pn" id="p3">
  <div class="card">
    <div class="card-t">Cost-Adjusted PnL Simulation &mdash; OOS (1-day signal, &yen;1,000,000 notional)</div>
    <div class="mstrip" id="pnlstrip" style="margin-bottom:14px"></div>
    <div class="cw h240"><canvas id="c3p"></canvas></div>
    <div class="note" id="pnlnote"></div>
  </div>
</div>

<!-- TAB 4: OPTIMIZATION -->
<div class="pn" id="p4">
  <div class="card">
    <div class="card-t">Current Ensemble Weights (selected from prior __SEL_N__d errors only)</div>
    <div id="wrows"></div>
    <div class="note">
      重みはグリッドサーチ(0.1刻み)だが、選択に使う誤差窓と性能評価期間は完全分離(walk-forward)。
      ナイーブも重み候補に含むため、他モデルに優位性がない局面では自動的にランダムウォークへ収束する。
    </div>
  </div>
  <div class="card">
    <div class="card-t">Weight Evolution &mdash; Last 60 Days</div>
    <div class="cw h200"><canvas id="c4w"></canvas></div>
  </div>
  <div class="card">
    <div class="card-t">Configuration</div>
    <table id="tcfg"></table>
  </div>
</div>

<div class="foot">
  &copy; Attractor Co., Ltd. &nbsp;|&nbsp;
  Generated: __GENERATED_AT__ &nbsp;|&nbsp; Source: __SOURCE__ &nbsp;|&nbsp;
  Auto-update: Mon&ndash;Fri 09:00 JST &nbsp;|&nbsp; GitHub Actions
</div>

</div>

<script>
const D=__DATA__;
const RN=__REGIME_NAMES__;
const RC=['#3ddc84','#4a9eff','#f0a830','#ff6b6b','#a070ff'];

document.getElementById('hprice').textContent='¥'+D.last_price.toFixed(2);
document.getElementById('hdate').textContent=D.last_date+' Close';
const rb=document.getElementById('rbadge');
rb.className='rbadge r'+D.current_regime;
document.getElementById('rlabel').textContent=RN[D.current_regime];

const ratio=D.oos.ensemble.rmse/D.oos.naive.rmse;
const mData=[
  {l:'1D Forecast',v:'¥'+D.ens[0].mean.toFixed(2),s:'['+D.ens[0].p10.toFixed(2)+', '+D.ens[0].p90.toFixed(2)+']',c:'cyan'},
  {l:'Up Probability',v:(D.ens[0].up_prob*100).toFixed(0)+'%',s:'OOS residual based',c:D.ens[0].up_prob>0.5?'green':'red'},
  {l:'RMSE vs Naive',v:ratio.toFixed(3),s:ratio<1?'beats random walk':'no edge vs RW',c:ratio<1?'green':'red'},
  {l:'OOS Dir. Acc.',v:D.oos.ensemble.da!=null?(D.oos.ensemble.da*100).toFixed(0)+'%':'—',s:'1d, walk-forward',c:'blue'},
  {l:'P10–P90 Coverage',v:D.coverage.d1?(D.coverage.d1.cov80*100).toFixed(0)+'%':'—',s:'target 80%',c:'purple'},
  {l:'Volatility (Ann.)',v:(D.current_vol*100).toFixed(2)+'%',s:'Hurst '+D.current_hurst.toFixed(2),c:'amber'},
];
document.getElementById('mstrip').innerHTML=mData.map(m=>
  `<div class="mc"><div class="mc-l">${m.l}</div><div class="mc-v" style="color:var(--${m.c})">${m.v}</div><div class="mc-s">${m.s}</div></div>`
).join('');

const tabs=document.querySelectorAll('.tab');
const panels=document.querySelectorAll('.pn');
function sw(i){tabs.forEach((t,j)=>t.classList.toggle('on',i===j));panels.forEach((p,j)=>p.classList.toggle('on',i===j));}

// ---- TAB 0 ----
const lp=D.last20_prices, ld=D.last20_dates, nL=lp.length;
const pLabels=[...ld.map(d=>d.slice(5)),'+1','+2','+3','+4','+5'];
const nPad=Array(nL-1).fill(null);
const actData=[...lp,...Array(5).fill(null)];
const ensM=[...nPad,D.last_price,...D.ens.map(p=>p.mean)];
const p10=[...nPad,D.last_price,...D.ens.map(p=>p.p10)];
const p25=[...nPad,D.last_price,...D.ens.map(p=>p.p25)];
const p75=[...nPad,D.last_price,...D.ens.map(p=>p.p75)];
const p90=[...nPad,D.last_price,...D.ens.map(p=>p.p90)];
const arM=[...nPad,D.last_price,...D.ar_preds];
const allVals=[...lp,...D.ens.map(p=>p.p10),...D.ens.map(p=>p.p90)].filter(v=>v!=null);
const ymin=Math.min(...allVals)-0.5, ymax=Math.max(...allVals)+0.5;

new Chart(document.getElementById('c0'),{
  type:'line',
  data:{labels:pLabels,datasets:[
    {data:p90,borderWidth:0,pointRadius:0,fill:'+1',backgroundColor:'rgba(0,212,255,0.06)',spanGaps:true},
    {data:p75,borderWidth:0,pointRadius:0,fill:'+1',backgroundColor:'rgba(0,212,255,0.10)',spanGaps:true},
    {data:p25,borderWidth:0,pointRadius:0,fill:'+1',backgroundColor:'rgba(0,212,255,0.06)',spanGaps:true},
    {data:p10,borderWidth:0,pointRadius:0,fill:false,spanGaps:true},
    {label:'Ensemble',data:ensM,borderColor:'#00d4ff',borderWidth:2,pointRadius:3,fill:false,tension:.3,spanGaps:true},
    {label:'AR(5)',data:arM,borderColor:'rgba(255,170,0,.6)',borderWidth:1.5,pointRadius:2,borderDash:[3,2],fill:false,tension:.3,spanGaps:true},
    {label:'Actual',data:actData,borderColor:'#00ff88',borderWidth:2,pointRadius:2,fill:false,tension:.2},
  ]},
  options:{responsive:true,maintainAspectRatio:false,
    plugins:{legend:{display:false},tooltip:{mode:'index',intersect:false,
      callbacks:{label:ctx=>ctx.parsed.y!=null&&ctx.dataset.label?ctx.dataset.label+': ¥'+ctx.parsed.y.toFixed(2):''}}},
    scales:{
      x:{grid:{color:'rgba(30,45,69,.6)'},ticks:{color:'#556688',maxRotation:45,autoSkip:false,maxTicksLimit:15}},
      y:{grid:{color:'rgba(30,45,69,.6)'},ticks:{color:'#556688',callback:v=>'¥'+v.toFixed(1)},min:ymin,max:ymax}
    }
  }
});

let th='<tr><th>Day</th><th>Median</th><th>P10–P90</th><th>Up%</th><th>NaiveΔ</th></tr>';
D.ens.forEach((p,i)=>{
  const up=p.up_prob>=0.5;
  const arrow=p.mean>D.last_price?'↑':p.mean<D.last_price?'↓':'—';
  const ac=p.mean>D.last_price?'green':p.mean<D.last_price?'red':'amber';
  const probColor=up?'var(--green)':'var(--red)';
  th+=`<tr>
    <td style="color:var(--text3);font-size:11px">+${i+1}d</td>
    <td style="color:var(--${ac});font-weight:500">${arrow} ${p.mean.toFixed(2)}</td>
    <td style="font-size:11px;color:var(--text3)">${p.p10.toFixed(2)} – ${p.p90.toFixed(2)}</td>
    <td>
      <div style="font-size:12px;color:${probColor}">${(p.up_prob*100).toFixed(0)}%</div>
      <div class="prob-bar"><div style="height:100%;width:${(p.up_prob*100).toFixed(0)}%;background:${probColor}"></div></div>
    </td>
    <td style="color:var(--text2);font-size:11px">${(p.mean-D.last_price>=0?'+':'')+(p.mean-D.last_price).toFixed(2)}</td>
  </tr>`;
});
document.getElementById('t0').innerHTML=th;

const trans=D.transition_prob[D.current_regime];
let tmh='';
trans.forEach((p,j)=>{
  if(p<0.01) return;
  tmh+=`<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;font-size:11px">
    <div style="width:120px;color:var(--text3);font-size:10px">${RN[j]}</div>
    <div style="flex:1;height:6px;background:var(--bg4);border-radius:3px;overflow:hidden"><div style="height:100%;width:${(p*100).toFixed(0)}%;background:${RC[j]};border-radius:3px"></div></div>
    <div style="width:36px;text-align:right;color:${RC[j]}">${(p*100).toFixed(0)}%</div>
  </div>`;
});
document.getElementById('tmat').innerHTML=tmh;

let udh='';
D.ens.forEach((p,i)=>{
  const up=p.up_prob;
  udh+=`<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;font-size:11px">
    <span style="color:var(--text3);width:24px">+${i+1}</span>
    <div style="flex:1;height:6px;background:var(--bg4);border-radius:3px;overflow:hidden">
      <div style="height:100%;width:${(up*100).toFixed(0)}%;background:var(--green);border-radius:3px"></div>
    </div>
    <span style="color:var(--green);width:36px">${(up*100).toFixed(0)}%↑</span>
    <span style="color:var(--red);width:36px">${((1-up)*100).toFixed(0)}%↓</span>
  </div>`;
});
document.getElementById('updown').innerHTML=udh;

// ---- TAB 1 ----
const rh=D.regime_history;
const rtl=document.getElementById('rtl');
rh.forEach(r=>{
  const s=document.createElement('div');
  s.className='rt-seg';
  s.style.background=RC[r.regime];
  s.style.opacity='.75';
  s.title=`${r.date} ¥${r.price.toFixed(2)} | ${RN[r.regime]}`;
  rtl.appendChild(s);
});
document.getElementById('rlegend').innerHTML=RN.map((n,i)=>
  `<span style="display:flex;align-items:center;gap:5px"><span style="width:10px;height:10px;border-radius:2px;background:${RC[i]};display:inline-block"></span>${n}</span>`
).join('');

new Chart(document.getElementById('c1'),{
  type:'line',
  data:{labels:rh.map(r=>r.date.slice(5)),datasets:[
    {label:'Price',data:rh.map(r=>r.price),borderColor:'#00d4ff',borderWidth:1.5,pointRadius:0,fill:false,tension:.3,yAxisID:'y'},
    {label:'Vol%',data:rh.map(r=>+(r.vol*100).toFixed(2)),borderColor:'rgba(255,170,0,.7)',borderWidth:1,pointRadius:0,borderDash:[3,2],fill:false,tension:.3,yAxisID:'y2'},
    {label:'ADX',data:rh.map(r=>r.adx!=null?+r.adx.toFixed(1):null),borderColor:'rgba(170,102,255,.7)',borderWidth:1,pointRadius:0,borderDash:[2,3],fill:false,tension:.3,yAxisID:'y2',spanGaps:true},
  ]},
  options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},
    scales:{
      x:{grid:{color:'rgba(30,45,69,.5)'},ticks:{color:'#556688',maxTicksLimit:10,maxRotation:45}},
      y:{grid:{color:'rgba(30,45,69,.5)'},ticks:{color:'#00d4ff',callback:v=>'¥'+v.toFixed(0)},position:'left'},
      y2:{grid:{display:false},ticks:{color:'rgba(255,170,0,.7)',callback:v=>v.toFixed(0)},position:'right',min:0,max:50}
    }
  }
});

new Chart(document.getElementById('c1h'),{
  type:'line',
  data:{labels:rh.map(r=>r.date.slice(5)),datasets:[
    {label:'Hurst',data:rh.map(r=>r.hurst!=null?+r.hurst.toFixed(3):null),borderColor:'#aa66ff',borderWidth:1.5,pointRadius:0,fill:true,backgroundColor:'rgba(170,102,255,.1)',tension:.3,spanGaps:true},
  ]},
  options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},
    scales:{
      x:{display:false},
      y:{grid:{color:'rgba(30,45,69,.5)'},ticks:{color:'#aa66ff'},min:0.3,max:0.8,
        title:{display:true,text:'Hurst (>0.55=Trend)',color:'#556688',font:{size:10}}}
    }
  }
});

const regimeCounts=Array(5).fill(0);
rh.forEach(r=>regimeCounts[r.regime]++);
new Chart(document.getElementById('c1r'),{
  type:'doughnut',
  data:{labels:RN,datasets:[{data:regimeCounts,backgroundColor:RC.map(c=>c+'bb'),borderColor:RC,borderWidth:1}]},
  options:{responsive:true,maintainAspectRatio:false,
    plugins:{legend:{position:'right',labels:{color:'#8899bb',font:{size:10},boxWidth:10}}}}
});

// ---- TAB 2 ----
const MK=[['naive','Naive (RW)','#8899bb'],['ensemble','Ensemble','#00ff88'],['chaos','Chaos k-NN','#aa66ff'],['ar','AR(5) ret','#ffaa00'],['momentum','Momentum','#4488ff']];
let bth='<tr><th>Model</th><th>RMSE</th><th>vs RW</th><th>MAE</th><th>DA</th><th>DM p</th></tr>';
MK.forEach(([k,n,c])=>{
  const s=D.oos[k]; if(!s) return;
  const rr=s.rmse/D.oos.naive.rmse;
  const rc=k==='naive'?'var(--text2)':(rr<1?'var(--green)':'var(--red)');
  const dm=D.dm[k];
  bth+=`<tr>
    <td style="color:${c}">${n}</td>
    <td>${s.rmse.toFixed(4)}</td>
    <td style="color:${rc}">${k==='naive'?'1.000':rr.toFixed(3)}</td>
    <td>${s.mae.toFixed(4)}</td>
    <td>${s.da!=null?(s.da*100).toFixed(1)+'%':'—'}</td>
    <td style="color:${dm&&dm.p<0.05?'var(--green)':'var(--text3)'}">${dm?dm.p.toFixed(3):'—'}</td>
  </tr>`;
});
document.getElementById('tbt').innerHTML=bth;

let msh='<tr><th>Day</th><th>DA</th><th>MAE</th><th>RMSE</th><th>vs RW</th></tr>';
for(let s=1;s<=5;s++){
  const b=D.multistep['d'+s], nv=D.multistep_naive['d'+s];
  if(!b){continue;}
  const rr=nv?b.rmse/nv.rmse:null;
  const daC=b.da==null?'var(--text3)':b.da>0.55?'var(--green)':b.da>0.45?'var(--amber)':'var(--red)';
  msh+=`<tr>
    <td style="color:var(--text3)">+${s}d</td>
    <td style="color:${daC}">${b.da!=null?(b.da*100).toFixed(1)+'%':'—'}</td>
    <td>${b.mae.toFixed(4)}</td>
    <td>${b.rmse.toFixed(4)}</td>
    <td style="color:${rr!=null&&rr<1?'var(--green)':'var(--red)'}">${rr!=null?rr.toFixed(3):'—'}</td>
  </tr>`;
}
document.getElementById('tms').innerHTML=msh;

let cvh='<tr><th>Day</th><th>P10–P90 (80%)</th><th>P25–P75 (50%)</th></tr>';
for(let s=1;s<=5;s++){
  const c=D.coverage['d'+s];
  if(!c){continue;}
  const ok80=Math.abs(c.cov80-0.8)<=0.1, ok50=Math.abs(c.cov50-0.5)<=0.1;
  cvh+=`<tr>
    <td style="color:var(--text3)">+${s}d</td>
    <td style="color:${ok80?'var(--green)':'var(--amber)'}">${(c.cov80*100).toFixed(0)}%</td>
    <td style="color:${ok50?'var(--green)':'var(--amber)'}">${(c.cov50*100).toFixed(0)}%</td>
  </tr>`;
}
document.getElementById('tcov').innerHTML=cvh;

const res=D.errors_ens_60||[];
new Chart(document.getElementById('c2b'),{
  type:'bar',
  data:{labels:res.map(r=>r.date.slice(5)),datasets:[
    {data:res.map(r=>r.e),backgroundColor:res.map(r=>r.e>=0?'rgba(0,255,136,.5)':'rgba(255,68,102,.4)'),borderWidth:0,borderRadius:2}
  ]},
  options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},
    scales:{x:{grid:{color:'rgba(30,45,69,.3)'},ticks:{color:'#556688',maxTicksLimit:8,maxRotation:45}},
      y:{grid:{color:'rgba(30,45,69,.3)'},ticks:{color:'#8899bb'},title:{display:true,text:'Error (JPY)',color:'#556688',font:{size:10}}}}
  }
});

// ---- TAB 3: PNL ----
const P=D.pnl;
if(P){
  // 円表記: 符号 + ¥ + 3桁区切り (例: +¥12,345 / -¥3,210)
  const fmtY=(v,sign)=>(sign?(v>=0?'+':'-'):(v<0?'-':''))+'¥'+Math.round(Math.abs(v)).toLocaleString('ja-JP');
  const pData=[
    {l:'Total PnL',v:fmtY(P.total,true),s:'per ¥1,000,000 notional',c:P.total>=0?'green':'red'},
    {l:'Sharpe (Ann.)',v:P.sharpe.toFixed(2),s:'cost-adjusted',c:P.sharpe>0?'green':'red'},
    {l:'Hit Rate',v:P.hit!=null?(P.hit*100).toFixed(1)+'%':'—',s:P.n_trades+' trades / '+P.n_days+' days',c:'blue'},
    {l:'Max Drawdown',v:fmtY(P.max_dd,false),s:'cumulative (JPY)',c:'amber'},
  ];
  document.getElementById('pnlstrip').innerHTML=pData.map(m=>
    `<div class="mc"><div class="mc-l">${m.l}</div><div class="mc-v" style="color:var(--${m.c})">${m.v}</div><div class="mc-s">${m.s}</div></div>`
  ).join('');
  let cum=0;
  const cumData=D.pnl_series.map(r=>{cum+=r.pnl;return Math.round(cum);});
  new Chart(document.getElementById('c3p'),{
    type:'line',
    data:{labels:D.pnl_series.map(r=>r.date.slice(5)),datasets:[
      {data:cumData,borderColor:'#00ff88',borderWidth:1.5,pointRadius:0,fill:true,backgroundColor:'rgba(0,255,136,.06)',tension:.1}
    ]},
    options:{responsive:true,maintainAspectRatio:false,
      plugins:{legend:{display:false},tooltip:{callbacks:{label:ctx=>'累積PnL: '+fmtY(ctx.parsed.y,true)}}},
      scales:{x:{grid:{color:'rgba(30,45,69,.3)'},ticks:{color:'#556688',maxTicksLimit:10}},
        y:{grid:{color:'rgba(30,45,69,.3)'},ticks:{color:'#8899bb',callback:v=>fmtY(v,false)},
          title:{display:true,text:'Cumulative PnL (JPY)',color:'#556688',font:{size:10}}}}
    }
  });
  document.getElementById('pnlnote').textContent=
    `前提: 想定元本¥1,000,000(保有USD数量=元本÷当日価格)、往復スプレッド${P.cost_pips}pips、`+
    `予測変化が${P.thresh}円未満の日は見送り。PnLはすべて円建て。スリッページ・スワップ(金利差)は未考慮の簡易検証。`;
}

// ---- TAB 4 ----
const wk=[['chaos','Chaos k-NN','purple'],['ar','AR(5) ret','amber'],['momentum','Momentum','blue'],['naive','Naive (RW)','cyan']];
let wh='';
wk.forEach(([k,n,c])=>{
  const pct=Math.round(D.weights[k]*100);
  const rmse=D.oos[k]?D.oos[k].rmse:null;
  wh+=`<div style="display:grid;grid-template-columns:120px 1fr 50px 110px;gap:10px;align-items:center;margin-bottom:12px">
    <div style="font-size:11px;color:var(--${c})">${n}</div>
    <div style="height:6px;background:var(--bg4);border-radius:3px;overflow:hidden">
      <div style="height:100%;width:${pct}%;background:var(--${c});opacity:.6;border-radius:3px"></div>
    </div>
    <div style="text-align:right;font-size:14px;font-weight:300;color:var(--${c})">${pct}%</div>
    <div style="text-align:right;font-size:10px;color:var(--text3)">OOS RMSE ${rmse!=null?rmse.toFixed(4):'—'}</div>
  </div>`;
});
document.getElementById('wrows').innerHTML=wh;

const whist=D.weights_history;
new Chart(document.getElementById('c4w'),{
  type:'line',
  data:{labels:whist.map(r=>r.date.slice(5)),datasets:[
    {label:'Chaos',data:whist.map(r=>r.w[0]),borderColor:'#aa66ff',backgroundColor:'rgba(170,102,255,.3)',fill:true,pointRadius:0,borderWidth:1},
    {label:'AR',data:whist.map(r=>r.w[1]),borderColor:'#ffaa00',backgroundColor:'rgba(255,170,0,.3)',fill:true,pointRadius:0,borderWidth:1},
    {label:'Momentum',data:whist.map(r=>r.w[2]),borderColor:'#4488ff',backgroundColor:'rgba(68,136,255,.3)',fill:true,pointRadius:0,borderWidth:1},
    {label:'Naive',data:whist.map(r=>r.w[3]),borderColor:'#8899bb',backgroundColor:'rgba(136,153,187,.3)',fill:true,pointRadius:0,borderWidth:1},
  ]},
  options:{responsive:true,maintainAspectRatio:false,
    plugins:{legend:{labels:{color:'#8899bb',font:{size:10},boxWidth:10}}},
    scales:{x:{grid:{display:false},ticks:{color:'#556688',maxTicksLimit:10}},
      y:{stacked:true,min:0,max:1,grid:{color:'rgba(30,45,69,.3)'},ticks:{color:'#8899bb'}}}
  }
});

const cfg=D.config;
document.getElementById('tcfg').innerHTML=
  '<tr><th>Parameter</th><th>Value</th></tr>'+
  Object.entries(cfg).map(([k,v])=>`<tr><td style="color:var(--text3)">${k}</td><td>${v}</td></tr>`).join('');
</script>
</body>
</html>"""

def generate_html(D, source):
    html = HTML_TEMPLATE
    html = html.replace("__DATA__", json.dumps(D, ensure_ascii=False, separators=(',', ':')))
    html = html.replace("__REGIME_NAMES__", json.dumps(REGIME_NAMES))
    html = html.replace("__GENERATED_AT__", D["generated_at"])
    html = html.replace("__SOURCE__", source)
    html = html.replace("__EVAL_N__", str(EVAL_N))
    html = html.replace("__SEL_N__", str(SEL_N))
    return html

# ============================================================
# 9. メイン
# ============================================================
def main():
    print("=== USD/JPY Ensemble v3 (causal / walk-forward) ===")
    print("Time:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    dates, prices, source = fetch_usdjpy()
    N = len(prices)

    print("特徴量計算中(全て因果)...")
    rv  = rolling_vol(prices)
    adx, pdi, mdi = calc_adx(prices)
    hurst = calc_hurst(prices)
    mom   = calc_momentum(prices)
    regimes, vol_med = classify_regime(rv, adx, pdi, mdi, hurst)
    trans = calc_transition_prob(regimes)

    pn = rolling_zscore(prices)
    mn = rolling_zscore(mom)
    vn = rolling_zscore(rv)
    tau = estimate_tau(pn)
    Xe, offset = build_embedding(pn, mn, vn, tau)
    print(f"τ={tau} (ACFゼロ交差) | 埋め込み次元=5 | offset={offset}")

    cur_r = int(regimes[-1])
    print(f"Regime: {REGIME_NAMES[cur_r]} | Vol: {rv[-1]*100:.2f}% | "
          f"Hurst: {hurst[-1]:.3f} | ADX: {adx[-1]:.1f}")

    print(f"walk-forwardバックテスト中({EVAL_N + SEL_N + HORIZONS}日)...")
    preds, errs, start = walk_forward(prices, Xe, regimes, offset)
    ens_pred, ens_err, eval_idxs, w_hist = evaluate_oos(prices, preds, errs, start)
    print(f"OOS評価日数: {len(eval_idxs)}")

    # OOS統計(1日先) + DM検定
    oos, dm = {}, {}
    for m in MODEL_KEYS:
        oos[m] = _stats_at(errs[m][0], preds[m][0], prices, eval_idxs)
    oos["ensemble"] = _stats_at(ens_err[0], ens_pred[0], prices, eval_idxs)
    e_naive = np.array([errs["naive"][0, i] for i in eval_idxs])
    for m in ["chaos", "ar", "momentum"]:
        e_m = np.array([errs[m][0, i] for i in eval_idxs])
        s, p = dm_test(e_m, e_naive)
        dm[m] = {"stat": round(s, 3), "p": round(p, 4)} if s is not None else None
    e_ens = np.array([ens_err[0, i] for i in eval_idxs])
    s, p = dm_test(e_ens, e_naive)
    dm["ensemble"] = {"stat": round(s, 3), "p": round(p, 4)} if s is not None else None
    for m in ["naive", "ensemble", "chaos", "ar", "momentum"]:
        st = oos[m]
        if st:
            da_s = f"{st['da']*100:.1f}%" if st["da"] is not None else "—"
            dm_s = f" DM_p={dm[m]['p']:.3f}" if m in dm and dm[m] else ""
            print(f"  {m:9s}: RMSE={st['rmse']:.4f} MAE={st['mae']:.4f} DA={da_s}{dm_s}")

    # 多段階OOS(アンサンブル本体とナイーブ)
    multistep, multistep_naive = {}, {}
    for h in range(HORIZONS):
        st = _stats_at(ens_err[h], ens_pred[h], prices, eval_idxs)
        sn = _stats_at(errs["naive"][h], preds["naive"][h], prices, eval_idxs)
        if st:
            multistep[f"d{h+1}"] = {k: round(v, 4) if isinstance(v, float) else v
                                    for k, v in st.items()}
        if sn:
            multistep_naive[f"d{h+1}"] = {"rmse": round(sn["rmse"], 4)}

    # conformal区間とカバレッジ
    cq = conformal_quantiles(ens_err, eval_idxs)
    coverage = coverage_check(ens_err, eval_idxs)

    # PnL
    pnl_summary, pnl_recs = pnl_backtest(prices, ens_pred, eval_idxs)
    if pnl_summary:
        print(f"PnL(OOS, 元本100万円, cost込): total={pnl_summary['total']:+,}円 "
              f"Sharpe={pnl_summary['sharpe']:.2f} hit={pnl_summary['hit']}")

    # 本番予測(今日)
    idx = N - 1
    w_today, _ = select_weights(errs, idx)
    weights = {m: round(float(w_today[i]), 2) for i, m in enumerate(MODEL_KEYS)}
    print(f"Weights(直近{SEL_N}日で選択): {weights}")

    fc = {
        "chaos":    chaos_forecast(Xe, prices, regimes, offset, idx),
        "ar":       ar_forecast(prices, idx),
        "momentum": momentum_forecast(prices, idx),
        "naive":    naive_forecast(prices, idx),
    }
    last = float(prices[-1])
    ens_today = []
    for h in range(HORIZONS):
        mean = sum(w_today[i] * fc[m][h] for i, m in enumerate(MODEL_KEYS))
        q = cq.get(h)
        if q:
            es = q["_errs"]
            up_prob = float(np.mean(mean + es > last))
            row = {"mean": round(mean, 4),
                   "p10": round(mean + q[10], 4), "p25": round(mean + q[25], 4),
                   "p50": round(mean + q[50], 4), "p75": round(mean + q[75], 4),
                   "p90": round(mean + q[90], 4),
                   "up_prob": round(up_prob, 4)}
        else:
            row = {"mean": round(mean, 4), "p10": None, "p25": None,
                   "p50": None, "p75": None, "p90": None, "up_prob": 0.5}
        for i, m in enumerate(MODEL_KEYS):
            row[m] = round(fc[m][h], 4)
        ens_today.append(row)

    # 残差チャート用(直近60評価日)
    errors_60 = [{"date": dates[i], "e": round(float(ens_err[0, i]), 4)}
                 for i in eval_idxs[-60:] if not np.isnan(ens_err[0, i])]

    # 重み履歴(直近60評価日)
    weights_history = [{"date": dates[r["idx"]], "w": [round(x, 2) for x in r["w"]]}
                       for r in w_hist[-60:]]

    # PnL系列(日付付与)
    pnl_series = [{"date": dates[r["i"]], "pnl": r["pnl"]} for r in pnl_recs]

    regime_history = [
        {"date": dates[i], "price": round(float(prices[i]), 4),
         "regime": int(regimes[i]), "vol": round(float(rv[i]), 6),
         "adx": round(float(adx[i]), 2), "hurst": round(float(hurst[i]), 4)}
        for i in range(N - 60, N)
    ]

    def _clean(d):
        return {k: v for k, v in d.items() if k != "_errs"} if d else None

    payload = {
        "last_price":      round(last, 4),
        "last_date":       dates[-1],
        "current_regime":  cur_r,
        "current_vol":     round(float(rv[-1]), 6),
        "current_hurst":   round(float(hurst[-1]), 4),
        "current_adx":     round(float(adx[-1]), 2),
        "weights":         weights,
        "oos":             {k: ({kk: (round(vv, 4) if isinstance(vv, float) else vv)
                                 for kk, vv in v.items()} if v else None)
                            for k, v in oos.items()},
        "dm":              dm,
        "multistep":       multistep,
        "multistep_naive": multistep_naive,
        "coverage":        coverage,
        "pnl":             pnl_summary,
        "pnl_series":      pnl_series,
        "ens":             ens_today,
        "ar_preds":        [round(v, 4) for v in fc["ar"]],
        "transition_prob": [[round(float(p), 4) for p in row] for row in trans],
        "last20_dates":    dates[-20:],
        "last20_prices":   [round(float(v), 4) for v in prices[-20:]],
        "regime_history":  regime_history,
        "errors_ens_60":   errors_60,
        "weights_history": weights_history,
        "config": {
            "tau (ACF zero-cross)": tau,
            "embedding dim": "3 delay + momentum + vol (5d)",
            "k-NN": K_NN,
            "normalization": f"causal rolling z-score ({W_NORM}d)",
            "regime vol threshold": f"causal rolling median ({W_VOLMED}d)",
            "weight selection window": f"{SEL_N}d (walk-forward)",
            "OOS evaluation window": f"{EVAL_N}d",
            "intervals": "conformal (OOS residual quantiles)",
            "PnL notional": "¥1,000,000",
            "cost assumption": f"{SPREAD_PIPS} pips round-trip",
        },
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    os.makedirs("docs", exist_ok=True)
    html = generate_html(payload, source)
    with open("docs/index.html", "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[OK] docs/index.html {len(html):,} bytes")
    print("=== Done ===")

if __name__ == "__main__":
    main()

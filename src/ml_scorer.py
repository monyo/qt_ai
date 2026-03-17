"""
ML 選股評分器

功能：
  1. 訓練 XGBoost 模型（首次使用時自動訓練，結果快取）
  2. 對 ADD 候選股票計算 ML%（打敗 SPY 的歷史勝率）
  3. SHAP 解釋：列出推升/拖累這支股票評分的前幾個原因

使用方式：
    from src.ml_scorer import MLScorer
    scorer = MLScorer()
    scorer.ensure_trained()   # 首次 ~3-5 分鐘；之後 <1 秒讀快取
    results = scorer.score(["NVDA", "META", "PLTR"])
    # results["NVDA"] = {
    #     "prob": 0.71,
    #     "shap_top": [("科技板塊21日強勢", +0.12, "↑"),
    #                  ("252日動能強",       +0.08, "↑"),
    #                  ("油價上漲壓力",       -0.04, "↓")]
    # }

注意：ML% 是「打敗 SPY 的歷史勝率」，不是「股價上漲機率」。
      基準比例 ~48%（約一半的股票在任意一個月打敗 SPY）。
"""
import os
import pickle
import warnings
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

MODEL_CACHE  = "data/_ml_model.pkl"
FEATURE_CACHE = "data/_ml_features.pkl"
SECTOR_CACHE  = "data/_ml_sector_map.pkl"
ETF_CACHE     = "data/_ml_sector_etf_prices.pkl"
VIX_CACHE     = "data/_ml_vix.pkl"
OIL_CACHE     = "data/_ml_oil.pkl"
PRICE_CACHE   = "data/_protection_bt_prices.pkl"

SECTOR_ETF_MAP = {
    "Technology":             "XLK", "Information Technology": "XLK",
    "Financial Services":     "XLF", "Financials":             "XLF",
    "Healthcare":             "XLV", "Health Care":            "XLV",
    "Energy":                 "XLE", "Utilities":              "XLU",
    "Industrials":            "XLI", "Basic Materials":        "XLB",
    "Materials":              "XLB", "Real Estate":            "XLRE",
    "Consumer Cyclical":      "XLY", "Consumer Discretionary": "XLY",
    "Consumer Defensive":     "XLP", "Consumer Staples":       "XLP",
    "Communication Services": "XLC",
}
SECTOR_ETFS = sorted(set(SECTOR_ETF_MAP.values()))

FEATURE_COLS = [
    "mom_5d", "mom_21d", "mom_63d", "mom_126d", "mom_252d",
    "vol_21d", "vol_63d", "near52", "dist_ma50", "dist_ma200",
    "bounce_pct", "from_high_pct",
    "breadth", "vix_level", "vix_ma_ratio",
    "oil_ret_21d", "oil_ret_63d", "oil_ma_ratio", "oil_vs_spy_21d",
    "sector_rel_21d", "sector_rel_63d", "sector_rel_126d", "sector_streak",
    "month_sin", "month_cos",
    "sec_XLB", "sec_XLC", "sec_XLE", "sec_XLF", "sec_XLI",
    "sec_XLK", "sec_XLP", "sec_XLRE", "sec_XLU", "sec_XLV", "sec_XLY",
]

# 人類可讀的特徵描述（用於 SHAP 顯示）
FEATURE_LABELS = {
    "mom_5d":          "5日短線動能",
    "mom_21d":         "21日動能",
    "mom_63d":         "63日動能",
    "mom_126d":        "126日動能",
    "mom_252d":        "252日長期動能",
    "vol_21d":         "短期波動率",
    "vol_63d":         "中期波動率",
    "near52":          "接近52週高點",
    "dist_ma50":       "距50日均線",
    "dist_ma200":      "距200日均線",
    "bounce_pct":      "40日反彈幅度",
    "from_high_pct":   "距40日高點",
    "breadth":         "市場廣度",
    "vix_level":       "VIX恐慌水平",
    "vix_ma_ratio":    "VIX趨勢",
    "oil_ret_21d":     "油價21日漲跌",
    "oil_ret_63d":     "油價63日漲跌",
    "oil_ma_ratio":    "油價趨勢",
    "oil_vs_spy_21d":  "油價vs股市",
    "sector_rel_21d":  "板塊21日相對強弱",
    "sector_rel_63d":  "板塊63日相對強弱",
    "sector_rel_126d": "板塊6月相對強弱",
    "sector_streak":   "板塊連強月數",
    "month_sin":       "季節性（sin）",
    "month_cos":       "季節性（cos）",
    "sec_XLB":         "原物料板塊",
    "sec_XLC":         "通訊板塊",
    "sec_XLE":         "能源板塊",
    "sec_XLF":         "金融板塊",
    "sec_XLI":         "工業板塊",
    "sec_XLK":         "科技板塊",
    "sec_XLP":         "必需消費板塊",
    "sec_XLRE":        "房地產板塊",
    "sec_XLU":         "公用事業板塊",
    "sec_XLV":         "醫療板塊",
    "sec_XLY":         "非必需消費板塊",
}


class MLScorer:
    """XGBoost 模型包裝器，支援訓練、評分、SHAP 解釋"""

    def __init__(self, model_cache=MODEL_CACHE):
        self.model_cache = model_cache
        self.model    = None
        self.scaler   = None
        self.medians  = None
        self._ready   = False

    # ──────────────────────────────────────────────────────────────────
    # 公開 API
    # ──────────────────────────────────────────────────────────────────

    def ensure_trained(self) -> bool:
        """
        確保模型已就緒。快取存在直接讀取，否則自動訓練。
        Returns True if ready, False if training failed.
        """
        if self._ready:
            return True
        if os.path.exists(self.model_cache):
            return self._load()
        return self._train()

    def score(self, symbols: list, breadth: float = 0.6) -> dict:
        """
        對候選股票計算 ML%（打敗 SPY 的歷史勝率）+ SHAP 解釋。

        Parameters
        ----------
        symbols  : 股票代碼清單（新倉 + 金字塔候選）
        breadth  : 市場廣度值（0–1），由 breadth_monitor 提供；預設 0.6

        Returns
        -------
        { symbol: {"prob": float, "shap_top": [(label, shap_val, arrow), ...]} }
        """
        if not self._ready:
            return {}
        if not symbols:
            return {}

        try:
            X, valid_syms = self._build_inference_features(symbols, breadth)
            if X is None or len(valid_syms) == 0:
                return {}

            X_filled  = X.fillna(self.medians)
            X_scaled  = self.scaler.transform(X_filled)
            probs     = self.model.predict_proba(X_scaled)[:, 1]

            # SHAP 解釋
            shap_map = self._compute_shap(X_scaled, valid_syms)

            results = {}
            for i, sym in enumerate(valid_syms):
                shap_vals = shap_map.get(sym, [])
                results[sym] = {
                    "prob":     float(probs[i]),
                    "shap_top": shap_vals,
                }
            return results

        except Exception as e:
            return {}

    # ──────────────────────────────────────────────────────────────────
    # 訓練
    # ──────────────────────────────────────────────────────────────────

    def _train(self) -> bool:
        """從快取建立完整特徵矩陣並訓練模型"""
        required = [FEATURE_CACHE, SECTOR_CACHE, ETF_CACHE, PRICE_CACHE]
        for p in required:
            if not os.path.exists(p):
                print(f"  [MLScorer] 缺少訓練快取：{p}，跳過 ML 評分")
                return False

        try:
            from sklearn.preprocessing import StandardScaler
            try:
                from xgboost import XGBClassifier
            except ImportError:
                print("  [MLScorer] 未安裝 xgboost，跳過 ML 評分")
                return False

            print("  [MLScorer] 首次建立 ML 模型（約 3-5 分鐘）...")

            # 1. 載入基礎特徵（BASE 13 + y + date + symbol）
            df = pd.read_pickle(FEATURE_CACHE)

            # 2. 載入輔助資料
            prices = pd.read_pickle(PRICE_CACHE)
            prices.index = pd.to_datetime(prices.index)
            if prices.index.tz is not None:
                prices.index = prices.index.tz_convert(None)
            trading_days = prices.index.sort_values()

            sector_map = pd.read_pickle(SECTOR_CACHE)
            if isinstance(sector_map, pd.Series):
                sector_map = sector_map.to_dict()

            etf_prices = pd.read_pickle(ETF_CACHE)
            etf_prices.index = (pd.to_datetime(etf_prices.index).tz_localize(None)
                                 if etf_prices.index.tz is None
                                 else etf_prices.index.tz_convert(None))
            etf_prices = etf_prices.reindex(trading_days)

            # VIX
            if os.path.exists(VIX_CACHE):
                vix = pd.read_pickle(VIX_CACHE)
            else:
                vix = yf.Ticker("^VIX").history(start="2020-01-01", end="2026-03-01",
                                                  auto_adjust=True)["Close"]
                vix.index = pd.to_datetime(vix.index).tz_localize(None)
                vix.to_pickle(VIX_CACHE)
            vix = vix.reindex(trading_days, method="ffill")
            vix_ma63 = vix.rolling(63, min_periods=30).mean()

            # 石油
            if os.path.exists(OIL_CACHE):
                oil = pd.read_pickle(OIL_CACHE)
            else:
                oil = yf.Ticker("CL=F").history(start="2020-01-01", end="2026-03-01",
                                                  auto_adjust=True)["Close"]
                oil.index = pd.to_datetime(oil.index).tz_localize(None)
                oil.to_pickle(OIL_CACHE)
            oil = oil.reindex(trading_days, method="ffill")
            oil_ma200   = oil.rolling(200, min_periods=100).mean()
            spy_etf     = etf_prices.get("SPY", pd.Series(dtype=float))
            oil_ret_21s = (oil / oil.shift(21)  - 1) * 100
            oil_ret_63s = (oil / oil.shift(63)  - 1) * 100
            oil_ma_rs   = oil / oil_ma200
            oil_spy_21s = oil_ret_21s - (spy_etf / spy_etf.shift(21) - 1) * 100

            def tidx(date_str):
                return int(trading_days.searchsorted(pd.Timestamp(date_str)))

            date_ti_map = {
                str(ref.date()): tidx(str(ref.date()))
                for ref in pd.bdate_range("2021-01-01", "2025-09-01", freq="BMS")
            }

            # 3. 新增季節特徵
            months = pd.to_datetime(df["date"]).dt.month
            df["month_sin"] = np.sin(2 * np.pi * months / 12)
            df["month_cos"] = np.cos(2 * np.pi * months / 12)

            # 4. 板塊 one-hot
            df["_etf"] = df["symbol"].map(
                lambda s: SECTOR_ETF_MAP.get(sector_map.get(s, ""), None)
            )
            for etf in SECTOR_ETFS:
                df[f"sec_{etf}"] = (df["_etf"] == etf).astype(float)

            # 5. 板塊相對強弱 + streak
            rel_cache = {}
            for win in [21, 63, 126]:
                rel_cache[win] = {}
                for etf in SECTOR_ETFS:
                    if etf not in etf_prices.columns:
                        continue
                    rel = etf_prices[etf] / spy_etf
                    rel_cache[win][etf] = (rel / rel.shift(win) - 1) * 100

            rebal_all = pd.bdate_range("2020-12-01", "2025-09-01", freq="BMS")
            etf_streak_map = {}
            for etf in SECTOR_ETFS:
                if etf not in etf_prices.columns:
                    continue
                ep = etf_prices[etf]
                monthly = {}
                prev_ref = None
                for ref in rebal_all:
                    ti = tidx(str(ref.date()))
                    if prev_ref is None:
                        prev_ref = ref; continue
                    tp = tidx(str(prev_ref.date()))
                    sp_n, sp_p = spy_etf.iloc[ti], spy_etf.iloc[tp]
                    ep_n, ep_p = ep.iloc[ti], ep.iloc[tp]
                    if all(pd.notna(x) and x > 0 for x in [ep_n, ep_p, sp_n, sp_p]):
                        monthly[str(ref.date())] = (ep_n/ep_p - 1) - (sp_n/sp_p - 1)
                    prev_ref = ref
                ms = pd.Series(monthly)
                dates = sorted(ms.index)
                streak = {}
                for i, d in enumerate(dates):
                    cnt = 0
                    for j in range(i-1, max(i-13, -1), -1):
                        if ms[dates[j]] > 0:
                            cnt += 1
                        else:
                            break
                    streak[d] = cnt
                etf_streak_map[etf] = pd.Series(streak)

            for feat_name, win_map in [
                ("sector_rel_21d",  rel_cache[21]),
                ("sector_rel_63d",  rel_cache[63]),
                ("sector_rel_126d", rel_cache[126]),
            ]:
                col_vals = np.full(len(df), np.nan)
                for etf in SECTOR_ETFS:
                    mask = df["_etf"] == etf
                    if not mask.any() or etf not in win_map:
                        continue
                    series = win_map[etf]
                    for idx in df.index[mask]:
                        ti = date_ti_map.get(df.at[idx, "date"])
                        if ti is not None and ti < len(series):
                            v = series.iloc[ti]
                            if pd.notna(v):
                                col_vals[df.index.get_loc(idx)] = float(v)
                df[feat_name] = col_vals

            col_streak = np.zeros(len(df), dtype=float)
            for etf, series in etf_streak_map.items():
                mask = df["_etf"] == etf
                for idx in df.index[mask]:
                    v = series.get(df.at[idx, "date"], np.nan)
                    col_streak[df.index.get_loc(idx)] = v
            df["sector_streak"] = col_streak

            # 6. VIX + 石油特徵
            vix_lv = np.full(len(df), np.nan)
            vix_mr = np.full(len(df), np.nan)
            oil_r21 = np.full(len(df), np.nan)
            oil_r63 = np.full(len(df), np.nan)
            oil_mar = np.full(len(df), np.nan)
            oil_spy = np.full(len(df), np.nan)

            for i, (idx, row) in enumerate(df.iterrows()):
                ti = date_ti_map.get(row["date"])
                if ti is None:
                    continue
                def safe(s):
                    v = s.iloc[ti] if ti < len(s) else np.nan
                    return float(v) if pd.notna(v) else np.nan
                vl = safe(vix);  vm = safe(vix_ma63)
                vix_lv[i]  = vl
                vix_mr[i]  = vl / vm if pd.notna(vl) and pd.notna(vm) and vm > 0 else np.nan
                oil_r21[i] = safe(oil_ret_21s)
                oil_r63[i] = safe(oil_ret_63s)
                oil_mar[i] = safe(oil_ma_rs)
                oil_spy[i] = safe(oil_spy_21s)

            df["vix_level"]      = vix_lv
            df["vix_ma_ratio"]   = vix_mr
            df["oil_ret_21d"]    = oil_r21
            df["oil_ret_63d"]    = oil_r63
            df["oil_ma_ratio"]   = oil_mar
            df["oil_vs_spy_21d"] = oil_spy

            df = df.drop(columns=["_etf"])

            # 7. 訓練
            X_all = df[FEATURE_COLS].copy()
            y_all = df["y"].values
            medians = X_all.median()
            X_filled = X_all.fillna(medians)

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_filled)

            model = XGBClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.04,
                subsample=0.8, colsample_bytree=0.8,
                use_label_encoder=False, eval_metric="logloss",
                random_state=42, verbosity=0,
            )
            model.fit(X_scaled, y_all)

            # 8. 儲存
            payload = {
                "model":       model,
                "scaler":      scaler,
                "medians":     medians,
                "feature_cols": FEATURE_COLS,
            }
            os.makedirs(os.path.dirname(self.model_cache) or ".", exist_ok=True)
            with open(self.model_cache, "wb") as f:
                pickle.dump(payload, f)

            self.model    = model
            self.scaler   = scaler
            self.medians  = medians
            self._ready   = True
            print(f"  [MLScorer] 訓練完成，已快取至 {self.model_cache}")
            return True

        except Exception as e:
            print(f"  [MLScorer] 訓練失敗：{e}")
            return False

    def _load(self) -> bool:
        try:
            with open(self.model_cache, "rb") as f:
                payload = pickle.load(f)
            self.model    = payload["model"]
            self.scaler   = payload["scaler"]
            self.medians  = payload["medians"]
            self._ready   = True
            return True
        except Exception as e:
            print(f"  [MLScorer] 載入模型失敗：{e}")
            return False

    # ──────────────────────────────────────────────────────────────────
    # 推理特徵計算（live 資料）
    # ──────────────────────────────────────────────────────────────────

    def _build_inference_features(self, symbols, breadth):
        """下載 symbols 過去 300 天報價 + 市場環境，計算特徵矩陣"""
        try:
            sector_map = {}
            if os.path.exists(SECTOR_CACHE):
                sm = pd.read_pickle(SECTOR_CACHE)
                sector_map = sm.to_dict() if isinstance(sm, pd.Series) else sm

            # ── 下載股票 + 市場 ETF 資料 ──────────────────────────────
            fetch_syms = list(set(symbols) | set(SECTOR_ETFS) | {"SPY", "^VIX", "CL=F"})
            raw = yf.download(
                fetch_syms, period="400d",
                auto_adjust=True, progress=False,
            )
            if isinstance(raw.columns, pd.MultiIndex):
                close = raw["Close"]
            else:
                close = raw

            close.index = pd.to_datetime(close.index).tz_localize(None)
            close = close.ffill()

            if len(close) < 63:
                return None, []

            spy_series = close.get("SPY", pd.Series(dtype=float))
            vix_series = close.get("^VIX", pd.Series(dtype=float)).dropna()
            oil_series = close.get("CL=F", pd.Series(dtype=float)).dropna()
            if oil_series.empty:
                oil_series = pd.Series(dtype=float)

            # ── 計算市場環境（全市場共用）────────────────────────────
            vix_level = float(vix_series.iloc[-1])   if len(vix_series) > 0 else np.nan
            vix_ma63  = float(vix_series.rolling(63, min_periods=20).mean().iloc[-1]) \
                        if len(vix_series) >= 20 else np.nan
            vix_ratio = vix_level / vix_ma63 \
                        if pd.notna(vix_level) and pd.notna(vix_ma63) and vix_ma63 > 0 else np.nan

            def ret_n(series, n):
                if len(series) <= n:
                    return np.nan
                v0 = float(series.iloc[-1]); vn = float(series.iloc[-(n+1)])
                return (v0 / vn - 1) * 100 if vn > 0 else np.nan

            oil_ret_21 = ret_n(oil_series, 21)
            oil_ret_63 = ret_n(oil_series, 63)
            oil_ma200_v = float(oil_series.rolling(200, min_periods=60).mean().iloc[-1]) \
                         if len(oil_series) >= 60 else np.nan
            oil_ma_ratio = float(oil_series.iloc[-1]) / oil_ma200_v \
                           if pd.notna(oil_ma200_v) and oil_ma200_v > 0 else np.nan
            oil_vs_spy   = (oil_ret_21 or 0) - (ret_n(spy_series, 21) or 0) \
                           if pd.notna(oil_ret_21) else np.nan

            # 季節性
            today_month  = pd.Timestamp.today().month
            month_sin = float(np.sin(2 * np.pi * today_month / 12))
            month_cos = float(np.cos(2 * np.pi * today_month / 12))

            # ── 板塊相對強弱 + streak ──────────────────────────────
            sector_rels = {}   # etf -> {21: val, 63: val, 126: val}
            sector_streaks = {}  # etf -> streak
            for etf in SECTOR_ETFS:
                if etf not in close.columns:
                    continue
                ep = close[etf].dropna()
                if len(ep) < 21:
                    continue
                rel = ep / spy_series.reindex(ep.index).ffill()
                sector_rels[etf] = {}
                for win in [21, 63, 126]:
                    sector_rels[etf][win] = ret_n(rel, win)
                # streak：每月相對報酬（用月底日期模擬）
                monthly_idx = ep.resample("ME").last().index
                monthly_ep  = ep.reindex(monthly_idx, method="ffill")
                monthly_sp  = spy_series.reindex(monthly_idx, method="ffill")
                if len(monthly_ep) >= 2:
                    ep_ret  = monthly_ep.pct_change().dropna()
                    sp_ret  = monthly_sp.pct_change().dropna()
                    rel_ret = ep_ret - sp_ret
                    cnt = 0
                    for v in reversed(rel_ret.values[:-1]):   # 不含當月
                        if pd.notna(v) and v > 0:
                            cnt += 1
                        else:
                            break
                    sector_streaks[etf] = float(min(cnt, 12))
                else:
                    sector_streaks[etf] = 0.0

            # ── 逐股計算特徵 ──────────────────────────────────────────
            rows = []
            valid_syms = []
            for sym in symbols:
                if sym not in close.columns:
                    continue
                p = close[sym].dropna()
                if len(p) < 63:
                    continue

                p0 = float(p.iloc[-1])
                if p0 <= 0:
                    continue

                def m_n(n):
                    return (p0 / float(p.iloc[-(n+1)]) - 1) * 100 \
                           if len(p) > n else np.nan

                def log_vol(n, min_obs=15):
                    if len(p) < n:
                        return np.nan
                    seg = np.log(p.iloc[-n:] / p.iloc[-n:].shift(1)).dropna()
                    return float(seg.std() * np.sqrt(252)) if len(seg) >= min_obs else np.nan

                # 動能
                mom_5   = m_n(5)
                mom_21  = m_n(21)
                mom_63  = m_n(63)
                mom_126 = m_n(126)
                mom_252 = m_n(252)

                # 波動率
                vol_21 = log_vol(21)
                vol_63 = log_vol(63)

                # 技術位置
                max_252 = float(p.iloc[-252:].max()) if len(p) >= 252 else float(p.max())
                near52  = p0 / max_252 if max_252 > 0 else np.nan

                ma50  = float(p.iloc[-50:].mean())  if len(p) >= 50  else np.nan
                ma200 = float(p.iloc[-200:].mean()) if len(p) >= 200 else np.nan
                dist_ma50  = (p0 / ma50  - 1) * 100 if ma50  and ma50  > 0 else np.nan
                dist_ma200 = (p0 / ma200 - 1) * 100 if ma200 and ma200 > 0 else np.nan

                low_40  = float(p.iloc[-40:].min())  if len(p) >= 40 else float(p.min())
                high_40 = float(p.iloc[-40:].max())  if len(p) >= 40 else float(p.max())
                bounce    = (p0 / low_40  - 1) * 100 if low_40  > 0 else np.nan
                from_high = (p0 / high_40 - 1) * 100 if high_40 > 0 else np.nan

                # 板塊特徵
                etf = SECTOR_ETF_MAP.get(sector_map.get(sym, ""), None)
                sr21  = sector_rels.get(etf, {}).get(21,  np.nan) if etf else np.nan
                sr63  = sector_rels.get(etf, {}).get(63,  np.nan) if etf else np.nan
                sr126 = sector_rels.get(etf, {}).get(126, np.nan) if etf else np.nan
                streak = sector_streaks.get(etf, np.nan)           if etf else np.nan

                # sector one-hot
                sec_vals = {f"sec_{e}": 1.0 if e == etf else 0.0 for e in SECTOR_ETFS}

                row = {
                    "mom_5d": mom_5, "mom_21d": mom_21, "mom_63d": mom_63,
                    "mom_126d": mom_126, "mom_252d": mom_252,
                    "vol_21d": vol_21, "vol_63d": vol_63,
                    "near52": near52, "dist_ma50": dist_ma50, "dist_ma200": dist_ma200,
                    "bounce_pct": bounce, "from_high_pct": from_high,
                    "breadth": breadth,
                    "vix_level": vix_level, "vix_ma_ratio": vix_ratio,
                    "oil_ret_21d": oil_ret_21, "oil_ret_63d": oil_ret_63,
                    "oil_ma_ratio": oil_ma_ratio, "oil_vs_spy_21d": oil_vs_spy,
                    "sector_rel_21d": sr21, "sector_rel_63d": sr63,
                    "sector_rel_126d": sr126, "sector_streak": streak,
                    "month_sin": month_sin, "month_cos": month_cos,
                }
                row.update(sec_vals)
                rows.append(row)
                valid_syms.append(sym)

            if not rows:
                return None, []

            X = pd.DataFrame(rows, index=valid_syms)[FEATURE_COLS]
            return X, valid_syms

        except Exception as e:
            print(f"  [MLScorer] 特徵計算失敗：{e}")
            return None, []

    # ──────────────────────────────────────────────────────────────────
    # SHAP 解釋
    # ──────────────────────────────────────────────────────────────────

    def _compute_shap(self, X_scaled: np.ndarray, valid_syms: list) -> dict:
        """回傳 {sym: [(label, shap_val, arrow), ...]}，按絕對值排序前 3"""
        try:
            import shap
            explainer = shap.TreeExplainer(self.model)
            shap_vals = explainer.shap_values(X_scaled)   # shape: (n, n_features)

            result = {}
            for i, sym in enumerate(valid_syms):
                vals = shap_vals[i]                         # (n_features,)
                # 合併 month_sin/cos → 一個「季節性」
                season_idx = [FEATURE_COLS.index(c) for c in ("month_sin", "month_cos")
                              if c in FEATURE_COLS]
                season_val = sum(vals[j] for j in season_idx)

                items = []
                for j, feat in enumerate(FEATURE_COLS):
                    if feat in ("month_sin", "month_cos"):
                        continue
                    label = FEATURE_LABELS.get(feat, feat)
                    sv    = float(vals[j])
                    items.append((label, sv))
                # 加回季節性
                items.append(("季節性", float(season_val)))

                items.sort(key=lambda x: abs(x[1]), reverse=True)
                top = [(label, sv, "↑" if sv > 0 else "↓")
                       for label, sv in items[:3] if abs(sv) > 0.005]
                result[sym] = top
            return result
        except Exception:
            return {}

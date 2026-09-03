"""
SPY Directional Forecast Backtester (fin-only vs fin+weather, Tree vs MLP)
===========================================================================
Runs every (signal_horizon, pnl_horizon) pair in 1..20 for all 4 strategies
(Tree/MLP x fin-only/fin+weather) = 1,600 backtests.

  signal_horizon : days ahead the model's forecast target looks
                   (y_H = Close[t+H]/Close[t] - 1)
  pnl_horizon    : trading days a position is held once the dead-band fires
                   (no re-signaling mid-trade)

Training is done once per (model, features, signal_horizon) = 80 fits.
Each fit's predictions are reused across all 20 pnl_horizons.

Outputs:
  grid_search_results.csv  -- all 1,600 rows
  grid_heatmaps.png        -- Sharpe heatmaps (signal x pnl) for each model
  equity_curves.png        -- best Sharpe config per model vs Buy & Hold
"""

import os

# Must be set before NumPy/sklearn load, otherwise Adam/BLAS can return
# different weights on every run and the "best" horizons jump around.
os.environ["PYTHONHASHSEED"] = "42"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import logging
import random
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

warnings.filterwarnings("ignore")
random.seed(42)
np.random.seed(42)
SEED = 42

# =============================================================================
# CONFIG
# =============================================================================
TRAIN_START = "2000-01-01"
TRAIN_END   = "2018-12-31"
TEST_START  = "2019-01-01"
TEST_END    = "2026-07-30"

TICKER = "SPY"
WEATHER_LAT = 40.7128   # NYC -- proxy for "NYSE weather"
WEATHER_LON = -74.0060

THRESHOLDS = [
    0.0000,
    0.0005,
    0.0010,
    0.0015,
    0.0020,
    0.0025,
    0.0030,
    0.0040,
    0.0050,
]
ALLOW_SHORT_OPTIONS = [False, True]
COST_BPS          = 0.6
SNAPSHOT_SLIP_BPS = 0.0     # extra per-side cost vs official close (near-close fill)

RETRAIN   = "frozen"        # "frozen" or "rolling"
ROLL_FREQ = "MS"
ANN       = 252

SIGNAL_HORIZONS = list(range(1, 21))
PNL_HORIZONS    = list(range(1, 21))
MODELS          = [("tree", "Tree"), ("mlp", "MLP")]
FEATURE_SETS    = ["fin-only", "fin+weather"]

OUTDIR = os.path.dirname(os.path.abspath(__file__))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger("backtest")


# =============================================================================
# DATA LOADING
# =============================================================================
def _naive_dates(idx) -> pd.DatetimeIndex:
    """Timezone-naive midnight dates so train/test cuts do not drift between runs."""
    idx = pd.DatetimeIndex(pd.to_datetime(idx))
    if idx.tz is not None:
        idx = idx.tz_convert("America/New_York").tz_localize(None)
    return idx.normalize()


def _clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.index = _naive_dates(df.index)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.index.name = "date"
    return df


def load_financial_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Daily OHLCV for `ticker`. Requires internet + `pip install yfinance`."""
    import yfinance as yf
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)
    return _clean_frame(df[["open", "high", "low", "close", "volume"]])


def load_weather_data(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    """Daily weather from Open-Meteo's free historical archive (no API key)."""
    import requests
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,"
                 "windspeed_10m_max,cloudcover_mean",
        "timezone": "America/New_York",
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    daily = r.json()["daily"]
    wdf = pd.DataFrame(daily)
    wdf["date"] = pd.to_datetime(wdf["time"])
    wdf = wdf.drop(columns=["time"]).set_index("date")
    return _clean_frame(wdf)


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================
def build_financial_features(df: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    ret1 = df["close"].pct_change()

    f["ret_1d"] = ret1
    f["ret_5d"] = df["close"].pct_change(5)
    f["ret_10d"] = df["close"].pct_change(10)
    f["ret_20d"] = df["close"].pct_change(20)
    f["vol_10d"] = ret1.rolling(10).std()
    f["vol_20d"] = ret1.rolling(20).std()
    f["sma_10_ratio"] = df["close"] / df["close"].rolling(10).mean() - 1
    f["sma_50_ratio"] = df["close"] / df["close"].rolling(50).mean() - 1

    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    f["rsi_14"] = 100 - 100 / (1 + rs)

    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    f["macd"] = ema12 - ema26
    f["macd_signal"] = f["macd"].ewm(span=9, adjust=False).mean()

    f["volume_z"] = (df["volume"] - df["volume"].rolling(20).mean()) / df["volume"].rolling(20).std()
    return f


def build_weather_features(wdf: pd.DataFrame) -> pd.DataFrame:
    w = pd.DataFrame(index=wdf.index)
    w["temp_max"] = wdf["temperature_2m_max"]
    w["temp_min"] = wdf["temperature_2m_min"]
    w["temp_range"] = w["temp_max"] - w["temp_min"]
    w["precip"] = wdf["precipitation_sum"]
    w["wind_max"] = wdf["windspeed_10m_max"]
    w["cloud_mean"] = wdf["cloudcover_mean"]
    for c in ["temp_max", "precip", "cloud_mean"]:
        w[f"{c}_anom"] = w[c] - w[c].rolling(30, min_periods=5).mean()
    return w


def forward_return(close: pd.Series, horizon: int) -> pd.Series:
    return close.shift(-horizon) / close - 1


# =============================================================================
# MODELING
# =============================================================================
class SeededMLPRegressor(BaseEstimator, RegressorMixin):
    """Tiny 32-16 ReLU MLP trained with full-batch gradient descent.

    sklearn's MLPRegressor (Adam/LBFGS) is not bit-stable on Windows: the same
    seed still yields different weights because BLAS matrix multiplies can run
    on several CPU cores in a different order. This estimator uses float64,
    a seeded NumPy RNG, one thread, and the same update every epoch.
    """

    def __init__(self, hidden=(32, 16), lr=0.05, epochs=400, l2=1e-3, seed=SEED):
        self.hidden = hidden
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.seed = seed

    def fit(self, X, y):
        X = np.ascontiguousarray(X, dtype=np.float64)
        y = np.ascontiguousarray(y, dtype=np.float64).reshape(-1, 1)
        rng = np.random.RandomState(self.seed)
        dims = [X.shape[1], *self.hidden, 1]
        weights, biases = [], []
        for din, dout in zip(dims[:-1], dims[1:]):
            weights.append(rng.normal(0.0, np.sqrt(2.0 / din), size=(din, dout)))
            biases.append(np.zeros((1, dout), dtype=np.float64))

        n = X.shape[0]
        with threadpool_limits(limits=1):
            for _ in range(self.epochs):
                acts = [X]
                h = X
                for i, (w, b) in enumerate(zip(weights, biases)):
                    z = h @ w + b
                    h = z if i == len(weights) - 1 else np.maximum(z, 0.0)
                    acts.append(h)
                grad = (acts[-1] - y) / n
                for i in range(len(weights) - 1, -1, -1):
                    if i < len(weights) - 1:
                        grad = grad * (acts[i + 1] > 0.0)
                    dw = acts[i].T @ grad + self.l2 * weights[i]
                    db = grad.sum(axis=0, keepdims=True)
                    if i > 0:
                        grad = grad @ weights[i].T
                    weights[i] -= self.lr * dw
                    biases[i] -= self.lr * db

        self.weights_ = weights
        self.biases_ = biases
        return self

    def predict(self, X):
        h = np.ascontiguousarray(X, dtype=np.float64)
        with threadpool_limits(limits=1):
            for i, (w, b) in enumerate(zip(self.weights_, self.biases_)):
                h = h @ w + b
                if i < len(self.weights_) - 1:
                    h = np.maximum(h, 0.0)
        return h.ravel()


MODEL_FACTORY = {
    "tree": lambda: Pipeline([
        ("scaler", StandardScaler()),
        ("model", DecisionTreeRegressor(
            max_depth=5, min_samples_leaf=50, splitter="best", random_state=SEED,
        )),
    ]),
    "mlp": lambda: Pipeline([
        ("scaler", StandardScaler()),
        ("model", SeededMLPRegressor(hidden=(32, 16), lr=0.05, epochs=400, l2=1e-3, seed=SEED)),
    ]),
}


def train_predict(features: pd.DataFrame, close: pd.Series, horizon: int, model_kind: str,
                   train_start, train_end, test_start, test_end,
                   retrain="frozen", roll_freq="MS") -> pd.Series:
    X = features.copy().sort_index()
    yv = forward_return(close, horizon).reindex(X.index)
    x_ok = X.notna().all(axis=1)

    train_start = pd.Timestamp(train_start)
    train_end = pd.Timestamp(train_end)
    test_start = pd.Timestamp(test_start)
    test_end = pd.Timestamp(test_end)

    # Labels are required for training only -- test-set predictions keep rows
    # where X is valid even if y is unknown (last `horizon` days of the sample).
    train_mask = x_ok & yv.notna() & (X.index >= train_start) & (X.index <= train_end)
    test_mask = x_ok & (X.index >= test_start) & (X.index <= test_end)

    X_arr = np.ascontiguousarray(X.to_numpy(dtype=np.float64))
    y_arr = yv.to_numpy(dtype=np.float64)
    preds = pd.Series(index=X.index[test_mask], dtype=float)

    if retrain == "frozen":
        model = MODEL_FACTORY[model_kind]()
        model.fit(X_arr[train_mask.to_numpy()], y_arr[train_mask.to_numpy()])
        preds[:] = model.predict(X_arr[test_mask.to_numpy()])

    elif retrain == "rolling":
        test_idx = X.index[test_mask]
        period_starts = pd.Series(test_idx, index=test_idx).resample(roll_freq).first().dropna()
        boundaries = list(period_starts.values) + [test_idx[-1] + pd.Timedelta(days=1)]
        for i in range(len(boundaries) - 1):
            chunk_start, chunk_end = boundaries[i], boundaries[i + 1]
            hist_mask = x_ok & yv.notna() & (X.index < chunk_start)
            if hist_mask.sum() < 100:
                continue
            model = MODEL_FACTORY[model_kind]()
            model.fit(X_arr[hist_mask.to_numpy()], y_arr[hist_mask.to_numpy()])
            chunk_mask = test_mask & (X.index >= chunk_start) & (X.index < chunk_end)
            if chunk_mask.sum() == 0:
                continue
            preds.loc[X.index[chunk_mask]] = model.predict(X_arr[chunk_mask.to_numpy()])
    else:
        raise ValueError(f"unknown RETRAIN={retrain!r}")

    return preds.dropna()


def signal_from_prediction(pred: pd.Series, threshold: float, allow_short: bool) -> pd.Series:
    sig = pd.Series(0, index=pred.index)
    sig[pred > threshold] = 1
    if allow_short:
        sig[pred < -threshold] = -1
    return sig


# =============================================================================
# BACKTEST ENGINE -- fixed pnl_horizon holding period, no mid-trade re-signals
# =============================================================================
def run_backtest_fixed_horizon(signal: pd.Series, price_close: pd.Series, pnl_horizon: int,
                                cost_bps: float = 0.0, slip_bps: float = 0.0) -> dict:
    idx = price_close.index
    n = len(idx)
    pos_of = {d: i for i, d in enumerate(idx)}

    sig_arr = np.zeros(n)
    for d, v in signal.items():
        loc = pos_of.get(d)
        if loc is not None:
            sig_arr[loc] = v

    position = np.zeros(n)
    trade_id = np.full(n, np.nan)
    tid = 0
    i = 0
    while i < n:
        s = sig_arr[i]
        if s != 0:
            entry = i + 1
            if entry >= n:
                break
            exit_ = min(entry + pnl_horizon - 1, n - 1)
            tid += 1
            position[entry:exit_ + 1] = s
            trade_id[entry:exit_ + 1] = tid
            i = exit_ + 1
        else:
            i += 1

    position = pd.Series(position, index=idx)
    trade_id = pd.Series(trade_id, index=idx)

    daily_ret = price_close.pct_change().fillna(0)
    turnover = position.diff().abs().fillna(position.abs())
    trade_cost = turnover * (cost_bps + slip_bps) / 1e4

    strat_ret = position * daily_ret - trade_cost
    equity = (1 + strat_ret).cumprod()

    return {"strat_ret": strat_ret, "equity": equity, "position": position, "trade_ids": trade_id}


# =============================================================================
# METRICS
# =============================================================================
def perf_metrics(strat_ret: pd.Series, equity: pd.Series, ann: int = 252) -> dict:
    n = len(strat_ret)
    total_return = equity.iloc[-1] - 1
    years = n / ann
    ann_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else np.nan
    ann_vol = strat_ret.std() * np.sqrt(ann)
    sharpe = (strat_ret.mean() * ann) / ann_vol if ann_vol > 0 else np.nan

    downside = strat_ret[strat_ret < 0]
    downside_vol = downside.std() * np.sqrt(ann) if len(downside) > 1 else np.nan
    sortino = (strat_ret.mean() * ann) / downside_vol if downside_vol and downside_vol > 0 else np.nan

    roll_max = equity.cummax()
    max_dd = (equity / roll_max - 1).min()

    active = strat_ret[strat_ret != 0]
    hit_rate = (active > 0).mean() if len(active) else np.nan

    


    return {
        "Total Return": total_return,
        "Annualized": ann_return,
        "Ann Vol": ann_vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Max Drawdown": max_dd,
        "Hit Rate": hit_rate,
        "Days": n,
    }


def activity_metrics(position: pd.Series, trade_ids: pd.Series, ann: int = 252) -> dict:
    td = trade_ids.dropna()
    trades = int(td.nunique())
    hold_lengths = td.groupby(td).size()

    flat = (position == 0)
    flat_ids = (~flat).cumsum().where(flat).dropna()
    flat_lengths = flat_ids.groupby(flat_ids).size()

    return {
        "Trades": trades,
        "Avg Hold (d)": hold_lengths.mean() if trades else np.nan,
        "Med Hold (d)": hold_lengths.median() if trades else np.nan,
        "Max Hold (d)": hold_lengths.max() if trades else np.nan,
        "Avg Flat (d)": flat_lengths.mean() if len(flat_lengths) else np.nan,
        "Avg Exposure": (position != 0).mean(),
        "Turnover/yr": position.diff().abs().sum() / (len(position) / ann),
    }


# =============================================================================
# GRID SEARCH
# =============================================================================
def load_and_prepare(load_fin, load_weather):
    log.info("Loading financial data...")
    fin = load_fin(TICKER, TRAIN_START, TEST_END)
    log.info("Loading weather data...")
    weather = load_weather(WEATHER_LAT, WEATHER_LON, TRAIN_START, TEST_END)

    fin_feats = build_financial_features(fin)
    weather_feats = build_weather_features(weather)
    close = fin["close"]
    close_test = close[(close.index >= TEST_START) & (close.index <= TEST_END)]
    feature_sets = {
        "fin-only": fin_feats,
        "fin+weather": fin_feats.join(weather_feats, how="left"),
    }
    return close, close_test, feature_sets


def run_grid_search(load_fin=load_financial_data, load_weather=load_weather_data,
                    signal_horizons=None, pnl_horizons=None,
                    thresholds=None, allow_short_options=None):

    signal_horizons = signal_horizons or SIGNAL_HORIZONS
    pnl_horizons = pnl_horizons or PNL_HORIZONS
    thresholds = thresholds or THRESHOLDS
    allow_short_options = (
        allow_short_options
        if allow_short_options is not None
        else ALLOW_SHORT_OPTIONS
    )

    n_total = (
        len(MODELS)
        * len(FEATURE_SETS)
        * len(signal_horizons)
        * len(pnl_horizons)
        * len(thresholds)
        * len(allow_short_options)
    )

    log.info(
        "Grid search: %d models x %d feature sets x %d signal horizons "
        "x %d pnl horizons x %d thresholds x %d short settings = %d simulations",
        len(MODELS),
        len(FEATURE_SETS),
        len(signal_horizons),
        len(pnl_horizons),
        len(thresholds),
        len(allow_short_options),
        n_total,
    )

    close, close_test, feature_sets = load_and_prepare(load_fin, load_weather)

    rows = []

    # Training is still only once per model/features/signal_horizon.
    n_fits = len(MODELS) * len(FEATURE_SETS) * len(signal_horizons)
    fit_count = 0

    for model_kind, model_name in MODELS:
        for feat_label in FEATURE_SETS:
            feats = feature_sets[feat_label]

            for sh in signal_horizons:

                fit_count += 1

                log.info(
                    "[fit %d/%d] %s (%s) signal_horizon=%dd",
                    fit_count,
                    n_fits,
                    model_name,
                    feat_label,
                    sh,
                )

                # Train once
                pred = train_predict(
                    feats,
                    close,
                    sh,
                    model_kind,
                    TRAIN_START,
                    TRAIN_END,
                    TEST_START,
                    TEST_END,
                    RETRAIN,
                    ROLL_FREQ,
                )

                # Test every threshold and long/short combination
                for threshold in thresholds:
                    for allow_short in allow_short_options:

                        sig = signal_from_prediction(
                            pred,
                            threshold,
                            allow_short
                        )

                        # Test every holding period
                        for ph in pnl_horizons:

                            bt = run_backtest_fixed_horizon(
                                sig,
                                close_test,
                                ph,
                                COST_BPS,
                                SNAPSHOT_SLIP_BPS,
                            )

                            rows.append({
                                "model": model_name,
                                "features": feat_label,
                                "signal_horizon": sh,
                                "pnl_horizon": ph,
                                "threshold": threshold,
                                "allow_short": allow_short,

                                **perf_metrics(
                                    bt["strat_ret"],
                                    bt["equity"],
                                    ANN
                                ),

                                **activity_metrics(
                                    bt["position"],
                                    bt["trade_ids"],
                                    ANN
                                ),
                            })

    grid_df = pd.DataFrame(rows)

    grid_df["Score"] = (
    grid_df["Total Return"]
    * (0.3372 / grid_df["Max Drawdown"].abs())
    )

    grid_df["Score"] = grid_df["Score"].replace(
    [np.inf, -np.inf],
    np.nan
    )

    log.info(
        "Grid search complete: %d rows",
        len(grid_df)
    )

    return grid_df, close, close_test, feature_sets


def best_configs(grid_df: pd.DataFrame, top_n: int = 1) -> pd.DataFrame:
    ranked = grid_df.dropna(subset=["Score"]).sort_values(
        ["Score", "Total Return", "Sharpe", "signal_horizon", "pnl_horizon"],
        ascending=[False, False, False, True, True],
        kind="mergesort",
    )

    return (
        ranked
        .groupby(["model", "features"], group_keys=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def equity_for_best(best: pd.DataFrame, close: pd.Series, close_test: pd.Series,
                     feature_sets: dict) -> dict:
    """Re-run the 4 winning (sig, pnl) configs so we can plot their equity curves."""
    kind_of = {"Tree": "tree", "MLP": "mlp"}
    curves = {}
    for _, row in best.iterrows():
        kind = kind_of[row["model"]]
        name = (f'{row["model"]} ({row["features"]}, '
                f'sig={int(row["signal_horizon"])}d, pnl={int(row["pnl_horizon"])}d)')
        log.info("Equity curve for best %s...", name)
        pred = train_predict(
            feature_sets[row["features"]], close, int(row["signal_horizon"]), kind,
            TRAIN_START, TRAIN_END, TEST_START, TEST_END, RETRAIN, ROLL_FREQ,
        )
        threshold = float(row["threshold"])
        allow_short = bool(row["allow_short"])

        sig = signal_from_prediction(pred, threshold, allow_short)
        bt = run_backtest_fixed_horizon(
            sig, close_test, int(row["pnl_horizon"]), COST_BPS, SNAPSHOT_SLIP_BPS,
        )
        curves[name] = bt["equity"]

    bh_ret = close_test.pct_change().fillna(0)
    curves["Buy & Hold"] = (1 + bh_ret).cumprod()
    return curves


def plot_equity_curves(equity_curves: dict, outdir: str, filename: str = "equity_curves.png") -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 6))
    for name, eq in equity_curves.items():
        is_bh = name == "Buy & Hold"
        ax.plot(eq.index, (eq - 1) * 100, label=name,
                linewidth=1.8 if is_bh else 1.2,
                linestyle="--" if is_bh else "-",
                color="black" if is_bh else None)

    ax.set_title("Best Total Return / Exposure config per model -- Cumulative Return")
    ax.set_ylabel("Cumulative Return (%)")
    ax.set_xlabel("Date")
    ax.axhline(0, color="grey", linewidth=0.6, alpha=0.6)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    path = os.path.join(outdir, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_grid_heatmaps(
    grid_df: pd.DataFrame,
    outdir: str,
    metric: str = "Score",
    filename: str = "grid_heatmaps.png"
) -> str:

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Make sure Score exists
    if "Score" not in grid_df.columns:
        grid_df = grid_df.copy()

        grid_df["Score"] = (
            grid_df["Total Return"]
            * (0.3372 / grid_df["Max Drawdown"].abs())
        )

        grid_df["Score"] = grid_df["Score"].replace(
            [np.inf, -np.inf],
            np.nan
        )

    combos = (
        grid_df[["model", "features"]]
        .drop_duplicates()
        .sort_values(["model", "features"])
        .values
        .tolist()
    )

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes = axes.ravel()

    # Global scale for all heatmaps
    vmin = grid_df[metric].min()
    vmax = grid_df[metric].max()

    for ax, (model_name, feat_label) in zip(axes, combos):

        # Select this model + feature combination
        sub = grid_df[
            (grid_df["model"] == model_name) &
            (grid_df["features"] == feat_label)
        ].copy()

        # There are now multiple rows for every
        # (signal_horizon, pnl_horizon) because threshold
        # and allow_short are also being tested.
        #
        # Keep the BEST Score for each horizon combination.
        sub = (
            sub.groupby(
                ["pnl_horizon", "signal_horizon"],
                as_index=False
            )[metric]
            .max()
        )

        # Now there is exactly ONE value per
        # (pnl_horizon, signal_horizon), so pivot works.
        pivot = sub.pivot(
            index="pnl_horizon",
            columns="signal_horizon",
            values=metric
        )

        pivot = pivot.sort_index(ascending=False)

        im = ax.imshow(
            pivot.values,
            aspect="auto",
            cmap="RdYlGn",
            vmin=vmin,
            vmax=vmax
        )

        ax.set_title(f"{model_name} ({feat_label})")
        ax.set_xlabel("signal_horizon (d)")
        ax.set_ylabel("pnl_horizon (d)")

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(
            pivot.columns,
            fontsize=6,
            rotation=90
        )

        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(
            pivot.index,
            fontsize=6
        )

        fig.colorbar(
            im,
            ax=ax,
            fraction=0.046,
            pad=0.04,
            label=metric
        )

    fig.suptitle(
        f"Grid Search -- {metric} by (signal_horizon, pnl_horizon)",
        fontsize=13
    )

    fig.tight_layout()

    path = os.path.join(outdir, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)

    return path


def buy_and_hold_row(close_test: pd.Series) -> dict:
    bh_ret = close_test.pct_change().fillna(0)
    bh_equity = (1 + bh_ret).cumprod()
    perf = perf_metrics(bh_ret, bh_equity, ANN)
    return {
        "model": "Buy & Hold",
        "features": "",
        "signal_horizon": np.nan,
        "pnl_horizon": np.nan,
        **perf,
        "Trades": np.nan,
        "Avg Hold (d)": np.nan,
        "Med Hold (d)": np.nan,
        "Max Hold (d)": np.nan,
        "Avg Flat (d)": np.nan,
        "Avg Exposure": 1.0,
        "Turnover/yr": np.nan,
    }


def format_best(best: pd.DataFrame) -> pd.DataFrame:
    out = best.copy()

    for c in [
        "Total Return",
        "Annualized",
        "Ann Vol",
        "Max Drawdown",
        "Hit Rate",
        "Avg Exposure",
    ]:
        if c in out.columns:
            out[c] = (out[c] * 100).round(2).astype(str) + "%"

    if "threshold" in out.columns:
        out["threshold"] = out["threshold"].round(4)

    if "Score" in out.columns:
        out["Score"] = out["Score"].round(3)

    for c in ["Sharpe", "Sortino"]:
        if c in out.columns:
            out[c] = out[c].round(2)

    for c in ["signal_horizon", "pnl_horizon", "Trades"]:
        if c in out.columns:
            out[c] = out[c].apply(
                lambda v: "" if pd.isna(v) else int(v)
            )

    out = out.rename(columns={"Avg Exposure": "Exposure"})

    cols = [
        "model",
        "features",
        "signal_horizon",
        "pnl_horizon",
        "threshold",
        "allow_short",
        "Total Return",
        "Max Drawdown",
        "Exposure",
        "Score",
        "Sharpe",
        "Sortino",
        "Hit Rate",
        "Trades",
    ]

    return out[[c for c in cols if c in out.columns]]


def main():
    grid_df, close, close_test, feature_sets = run_grid_search()
    grid_path = os.path.join(OUTDIR, "grid_search_results.csv")
    grid_df.to_csv(grid_path, index=False)
    log.info("Full grid (%d rows) written to %s", len(grid_df), grid_path)

    winners = best_configs(grid_df, top_n=1)
    table = pd.concat([winners, pd.DataFrame([buy_and_hold_row(close_test)])], ignore_index=True)
    print("\n[BEST CONFIG PER MODEL, BY CUSTOM RETURN / DRAWDOWN / EXPOSURE SCORE]")
    print("-" * 100)
    print(format_best(table).to_string(index=False))

    heatmap_path = plot_grid_heatmaps(grid_df, OUTDIR, metric="Score")
    log.info("Heatmap written to %s", heatmap_path)

    curves = equity_for_best(winners, close, close_test, feature_sets)
    plot_path = plot_equity_curves(curves, OUTDIR)
    log.info("Done. Results in %s (%s, %s)", OUTDIR, os.path.basename(heatmap_path),
             os.path.basename(plot_path))


if __name__ == "__main__":
    main()

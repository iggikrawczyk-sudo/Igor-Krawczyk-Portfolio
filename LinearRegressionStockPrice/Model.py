

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

file = r"C:\Users\iggik\Desktop\Python Finance project CV\matrix\top_40_tech_latest_quarter.csv"

df_train = pd.read_csv(
    file,
    sep=";",
    decimal=","
)

tickers = ["2303.TW", "2454.TW", "3711.TW", "6758.T", "AAPL", "ABNB", "ACLS",
    "ADBE", "ADI", "ADSK", "AMAT", "AMD", "ARM", "AVGO", "BABA", "CAT",
    "CDNS", "CHKP", "COHR", "CRM", "CRWD", "CSCO", "DASH", "DDOG", "DE",
    "DELL", "DSY.PA", "DTE.DE", "EBAY", "EMR", "ENTG", "ERIC-B.ST", "ESTC",
    "ETN", "ETSY", "FORM", "FTNT", "GE", "GEN", "GFS", "GLW", "GOOG", "GPN",
    "GRMN", "HON", "HPE", "HPQ", "HUBS", "IBM", "IFX.DE", "INTU", "JBL",
    "JD", "KEYS", "KLAC", "LOGI", "LRCX", "LULU", "MA", "MCHP", "MDB",
    "META", "MMM", "MPWR", "MRVL", "MSFT", "MU", "NFLX", "NOKIA.HE", "NOW",
    "NTAP", "NTES", "NVDA", "NXPI", "OKTA", "ON", "ONTO", "ORCL", "PANW",
    "PATH", "PDD", "PINS"
]


file2 = r"C:\Users\iggik\Desktop\Python Finance project CV\matrix\teststocks.csv"

df_check = pd.read_csv(
    file2,
    sep=";",
    decimal=","
)

tickers_check = [
    "TXN", "UBER", "UMC", "V", "VRT", "WDAY", "PLTR", "PYPL",
    "QCOM", "ROK", "ROKU", "RTX", "SAP", "SHOP", "SIE.DE", "SMCI",
    "SNPS", "SONY", "SPOT", "TDY", "TER", "TSLA", "TSM", "TT"
]


features = [
    "Gross Profit",
    "Growth",
    "Total Debt",
    "Current Assets",
    "Current Liabilities"
]

# --- Log-transform the skewed dollar-scale features ---
# "Growth" is likely already a ratio/percentage, so leave it as-is.
# Dollar-scale features get log1p (handles zeros safely, unlike plain log).
dollar_features = ["Gross Profit", "Total Debt", "Current Assets", "Current Liabilities"]

def prep_features(df):
    df_prepped = df[features].copy()
    for col in dollar_features:
        # Some companies can have negative Gross Profit or zero debt;
        # log1p(abs(x)) with sign preserved keeps things well-behaved.
        df_prepped[col] = np.sign(df_prepped[col]) * np.log1p(np.abs(df_prepped[col]))
    return df_prepped

X_train = prep_features(df_train)
X_check = prep_features(df_check)

# --- Log-transform the target (Market Cap is heavy-tailed) ---
y_train_raw = df_train["Market Cap"]
y_check_raw = df_check["Market Cap"]

y_train_log = np.log(y_train_raw)

# --- Model: scale features, then Ridge (handles correlated features better than plain OLS) ---
model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))

model.fit(X_train, y_train_log)

y_pred_log = model.predict(X_check)
y_pred = np.exp(y_pred_log)  # back to raw Market Cap scale

print("Actual prices:")
print(y_check_raw.to_numpy())

print("\nPredicted prices:")
print(y_pred)

print("\nR² (on log scale):", r2_score(y_train_log, model.predict(X_train)))
print("R² (check set, raw scale):", r2_score(y_check_raw, y_pred))
print("RMSE (check set, raw scale):", np.sqrt(mean_squared_error(y_check_raw, y_pred)))

# Ridge coefficients are on the scaled feature space, so they're comparable
# to each other but not directly in "dollars per dollar" terms.
ridge_step = model.named_steps["ridge"]
print("\nStandardized coefficients (log-target, scaled features):")
for feature, coefficient in zip(features, ridge_step.coef_):
    print(feature, ":", coefficient)

print("\nIntercept:", ridge_step.intercept_)

# --- Plot: log-log scale with ticker labels ---
actual = y_check_raw.to_numpy()
predicted = y_pred

plt.figure(figsize=(8, 8))
plt.scatter(actual, predicted)

min_value = max(min(actual.min(), predicted.min()), 1e9)
max_value = max(actual.max(), predicted.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--",
    label="Perfect Prediction"
)

for ticker, x, y in zip(tickers_check, actual, predicted):
    plt.annotate(
        ticker,
        (x, y),
        textcoords="offset points",
        xytext=(5, 5),
        fontsize=9
    )

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Actual Market Cap")
plt.ylabel("Predicted Market Cap")
plt.title("Linear Regression: Actual vs Predicted Market Cap (log-log)")

plt.legend()
plt.grid(True, which="both")
plt.tight_layout()
plt.show()

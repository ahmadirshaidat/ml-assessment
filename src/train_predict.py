"""Fit final model on all labeled data (Jan-Oct 2025) and predict validation + December chart rows.

Run: python train_predict.py
Outputs: validation_predictions.csv, data/december_chart_inputs.csv (filled), outputs/*.json
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import lightgbm as lgb

from features import build, date_signals
from common import PARAMS, SEED, clean_mask

tr = pd.read_csv("data/train-test.csv", parse_dates=["date"])
va = pd.read_csv("data/validation.csv", parse_dates=["date"])
tmpl = pd.read_csv("data/validation-predictions-template.csv")
dec = pd.read_csv("data/december-chart-inputs.csv", parse_dates=["date"])

signals = date_signals(tr, va)
X_tr = build(tr, signals)
y = np.log(tr["posted_rate"].values)
COLS = [c for c in X_tr.columns if c not in ("dom",)]  # keep doy: acts as "latest regime" proxy (see report)

keep = clean_mask(X_tr, y)
print(f"training rows: {keep.sum()} / {len(keep)} (dropped {(~keep).sum()} extreme-label outliers)")

# seed-bagging for stability
models = []
for s in range(5):
    p = {**PARAMS, "random_state": SEED + s}
    models.append(lgb.LGBMRegressor(**p).fit(X_tr[keep][COLS], y[keep]))


def predict(X):
    return np.exp(np.mean([m.predict(X[COLS]) for m in models], axis=0))


# ---- validation predictions
X_va = build(va, signals)
pred = predict(X_va)
out = tmpl[["load_id"]].merge(pd.DataFrame({"load_id": va["load_id"], "predicted_rate": np.round(pred, 2)}),
                              on="load_id", how="left")
assert out["predicted_rate"].notna().all() and len(out) == 12_000
out.to_csv("deliverables/validation_predictions.csv", index=False)
print("validation predictions:", out.predicted_rate.describe().round(1).to_dict())

# ---- fixed December chart: Lexington -> Fort Wayne, 360 mi, Dry Van, 32,000 lb
geo = pd.concat([
    tr[["pickup", "pickup_lat", "pickup_lon"]].rename(columns=lambda c: c.replace("pickup", "city")),
    tr[["delivery", "delivery_lat", "delivery_lon"]].rename(columns=lambda c: c.replace("delivery", "city")),
]).drop_duplicates("city").set_index("city")
d = dec.copy()
d["pickup_lat"], d["pickup_lon"] = geo.loc["Lexington", ["city_lat", "city_lon"]].values
d["delivery_lat"], d["delivery_lon"] = geo.loc["Fort Wayne", ["city_lat", "city_lon"]].values
# the chart file has no market_index/quote_signal: use that date's mean from validation features
d["market_index"] = np.nan
d["quote_signal"] = signals.reindex(d["date"])["qs_day"].values
X_dec = build(d, signals)
dec_pred = predict(X_dec)
dec["predicted_rate"] = np.round(dec_pred, 2)
dec["date"] = dec["date"].dt.strftime("%Y-%m-%d")
dec.to_csv("deliverables/december-chart-inputs.csv", index=False)
print(dec[["date", "predicted_rate"]].describe().round(1).T)

# sanity: sensitivity of the chart to the doy feature
alt_cols = [c for c in COLS if c != "doy"]
alt = np.exp(np.mean([lgb.LGBMRegressor(**{**PARAMS, "random_state": SEED + s}).fit(X_tr[keep][alt_cols], y[keep])
                      .predict(X_dec[alt_cols]) for s in range(2)], axis=0))
print("Dec mean with doy: %.1f | without doy: %.1f" % (dec_pred.mean(), alt.mean()))

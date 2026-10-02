"""Validation experiments. Run: python validate.py

Split logic: the real target (validation.csv) is Nov-Dec 2025 = strictly AFTER training data
(Jan-Oct 2025), so random K-fold would leak market regime/time and overstate accuracy.
We therefore use forward-chaining (rolling-origin) folds, plus a city-holdout check because
8 validation cities are unseen in training.
"""
from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from common import PARAMS, SEED, clean_mask
from features import build, date_signals

tr = pd.read_csv("data/train-test.csv", parse_dates=["date"])
va = pd.read_csv("data/validation.csv", parse_dates=["date"])
signals = date_signals(tr, va)
X_all = build(tr, signals)
y_all = np.log(tr["posted_rate"].values)



def metrics(y_true_rate, pred_rate):
    err = pred_rate - y_true_rate
    return dict(MAE=float(np.mean(np.abs(err))), RMSE=float(np.sqrt(np.mean(err ** 2))),
                MAPE=float(np.mean(np.abs(err) / y_true_rate) * 100),
                MedAPE=float(np.median(np.abs(err) / y_true_rate) * 100),
                R2=float(1 - np.sum(err ** 2) / np.sum((y_true_rate - y_true_rate.mean()) ** 2)))




def fit_predict(Xtr, ytr, Xte, cols, drop_outliers=True, params=PARAMS):
    if drop_outliers:
        m = clean_mask(Xtr, ytr)
        Xtr, ytr = Xtr[m], ytr[m]
    mdl = lgb.LGBMRegressor(**params)
    mdl.fit(Xtr[cols], ytr)
    return np.exp(mdl.predict(Xte[cols]))


ALL = list(X_all.columns)
NO_DOY = [c for c in ALL if c not in ("doy", "dom")]
CHOSEN = [c for c in ALL if c != "dom"]  # keeps doy as "latest regime" proxy
BASIC = ["log_dist", "equip", "weight"]  # naive reference model

# forward-chaining folds: train on months < k, test on months k..k+1  (mimics 2-month-ahead forecast)
months = tr["date"].dt.month.values
folds = [(list(range(1, k)), [k, k + 1]) for k in (5, 7, 9)]

experiments = {
    "baseline (dist+equip+weight)": (BASIC, True),
    "all features, no outlier drop": (NO_DOY, False),
    "no doy/dom": (NO_DOY, True),
    "with doy+dom": (ALL, True),
    "with doy only (CHOSEN)": (CHOSEN, True),
}
results = {}
for name, (cols, drop) in experiments.items():
    per = []
    for tr_m, te_m in folds:
        a, b = np.isin(months, tr_m), np.isin(months, te_m)
        p = fit_predict(X_all[a], y_all[a], X_all[b], cols, drop)
        per.append(metrics(tr["posted_rate"].values[b], p))
    results[name] = {k: float(np.mean([f[k] for f in per])) for k in per[0]}
    results[name]["per_fold_MAPE"] = [round(f["MAPE"], 2) for f in per]
    print(f"{name:34s}", {k: (round(v, 3) if not isinstance(v, list) else v) for k, v in results[name].items()})

# robust metric: evaluate also on 'clean' test rows (labels that are not extreme outliers)
a, b = months <= 8, months >= 9
p = fit_predict(X_all[a], y_all[a], X_all[b], CHOSEN, True)
yt = tr["posted_rate"].values[b]
cm = clean_mask(X_all[b], y_all[b])
print("last-2-month holdout  all rows   :", {k: round(v, 3) for k, v in metrics(yt, p).items()})
print("last-2-month holdout  clean rows :", {k: round(v, 3) for k, v in metrics(yt[cm], p[cm]).items()})
results["holdout_sep_oct_all"] = metrics(yt, p)
results["holdout_sep_oct_clean"] = metrics(yt[cm], p[cm])

# cold-start check: hold out 8 random cities entirely (any load touching them) and train on the rest
rng = np.random.default_rng(SEED)
cities = np.array(sorted(set(tr.pickup) | set(tr.delivery)))
held = set(rng.choice(cities, 8, replace=False))
touch = tr.pickup.isin(held) | tr.delivery.isin(held)
train_m = ~touch & (months <= 8)
test_m = touch & (months >= 9)
p = fit_predict(X_all[train_m], y_all[train_m], X_all[test_m], CHOSEN, True)
cs = metrics(tr["posted_rate"].values[test_m], p)
print("cold-start cities (unseen) :", {k: round(v, 3) for k, v in cs.items()}, "n=", int(test_m.sum()))
results["cold_start_cities"] = cs

json.dump(results, open("deliverables/validation_results.json", "w"), indent=2)

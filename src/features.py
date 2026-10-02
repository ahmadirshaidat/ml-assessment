"""Cleaning + feature engineering for the freight-rate challenge.

Data-quality findings that drive this file (see PLAN.md / report):
  * weight: ~0.6% missing, ~0.6% negative (sign flips), ~2.5% clipped at 47,500 lb
  * market_index: ~0.8% missing; it is a DATE-level signal + ~0.025 per-load noise
  * coordinates are distorted (e.g. San Francisco lat 35.2) but consistent per city
  * 8 cities in validation never appear in training -> no city-ID features;
    geography must come from coordinates so unseen cities still get sensible inputs
  * ~0.3% of labels are extreme outliers (rate/mile 0.3 or 9+), real noise sigma ~0.08 in log space
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EQUIP = {"Dry Van": 0, "Flatbed": 1, "Reefer": 2}
WEIGHT_CAP = 47_500.0


def _haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 3958.8 * 2 * np.arcsin(np.sqrt(a))


def date_signals(*frames: pd.DataFrame) -> pd.DataFrame:
    """Daily market_index / quote_signal means built from FEATURES only (no labels).

    market_index is a date-level signal, so averaging all loads on a date denoises it
    and lets us fill missing values and the December chart (whose input file has no
    market_index). Validation features are unlabeled, so using them here is not leakage.
    """
    allx = pd.concat([f[["date", "market_index", "quote_signal"]] for f in frames], ignore_index=True)
    allx["date"] = pd.to_datetime(allx["date"])
    g = allx.groupby("date").agg(mi_day=("market_index", "mean"), qs_day=("quote_signal", "mean"))
    g["mi_day_7d"] = g["mi_day"].rolling(7, min_periods=1, center=True).mean()
    return g


def build(df: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    X = pd.DataFrame(index=d.index)

    X["log_dist"] = np.log(d["distance"])
    X["equip"] = d["equipment"].map(EQUIP)

    w = d["weight"]
    X["weight_missing"] = w.isna().astype(int)
    X["weight_was_negative"] = (w < 0).astype(int)
    w = w.abs()
    X["weight_clipped"] = (w >= WEIGHT_CAP).astype(int)
    # impute with equipment-specific median (computed from the frame itself; weight has ~no signal beyond itself)
    X["weight"] = w.fillna(w.groupby(d["equipment"]).transform("median"))

    sig = signals.reindex(d["date"]).reset_index(drop=True)
    sig.index = d.index
    X["market_index"] = d["market_index"].fillna(sig["mi_day"])
    X["mi_day"] = sig["mi_day"]
    X["mi_day_7d"] = sig["mi_day_7d"]
    X["mi_missing"] = d["market_index"].isna().astype(int)
    X["quote_signal"] = d["quote_signal"]
    X["qs_day"] = sig["qs_day"]

    for p in ("pickup", "delivery"):
        X[f"{p}_lat"] = d[f"{p}_lat"]
        X[f"{p}_lon"] = d[f"{p}_lon"]
    X["geo_dist"] = _haversine(d.pickup_lat, d.pickup_lon, d.delivery_lat, d.delivery_lon)
    X["dlat"] = d.delivery_lat - d.pickup_lat
    X["dlon"] = d.delivery_lon - d.pickup_lon

    X["dow"] = d["date"].dt.dayofweek
    X["dom"] = d["date"].dt.day
    X["doy"] = d["date"].dt.dayofyear
    return X

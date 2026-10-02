"""Shared config/helpers used by validate.py and train_predict.py."""
import lightgbm as lgb
import numpy as np

SEED = 42
PARAMS = dict(objective="huber", alpha=0.3, learning_rate=0.03, n_estimators=1500, num_leaves=31,
              min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
              reg_lambda=1.0, random_state=SEED, verbose=-1, n_jobs=-1)


def clean_mask(X, y):
    """Keep rows whose log-rate residual vs a simple distance/equipment baseline is within 6 robust sigmas.
    Removes ~0.3-1% extreme labels (rate/mile ~0.3 or 9+) that are noise, not signal."""
    base = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.1, num_leaves=15, verbose=-1, random_state=SEED)
    base.fit(X[["log_dist", "equip"]], y)
    res = y - base.predict(X[["log_dist", "equip"]])
    mad = 1.4826 * np.median(np.abs(res - np.median(res)))
    return np.abs(res - np.median(res)) < 6 * mad

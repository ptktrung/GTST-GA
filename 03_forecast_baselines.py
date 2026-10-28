"""Naive forecasting baselines on the test split (Table 2).

Persistence: the travel time observed in the previous hour.
Historical average: mean travel time of the same arc over the training days (0-30) at the
same hour of day and the same day type (weekday or weekend). Two coarser groupings are
also reported for reference.
Test samples are the same as for GTST-Lite: days 38-44, every arc, hours with t >= 6.
"""
import numpy as np

from gtst_common import LAG, TEST_DAYS, TRAIN_DAYS, Network, save_json


def metrics(pred, actual):
    err = pred - actual
    return {"MAE_min": round(float(np.abs(err).mean()), 3),
            "RMSE_min": round(float(np.sqrt((err ** 2).mean())), 3),
            "MAPE_pct": round(float((np.abs(err) / actual).mean() * 100), 3)}


def main():
    net = Network()
    tt = net.travel_time.astype(np.float64)
    hours = np.arange(net.n_t)
    day, hod = hours // 24, hours % 24
    weekend = np.isin(day % 7, [5, 6])
    train = np.isin(day, list(TRAIN_DAYS))
    test_hours = [t for t in range(LAG, net.n_t) if day[t] in TEST_DAYS]
    actual = np.concatenate([tt[t] for t in test_hours])

    out = {"n_test": int(actual.size)}
    out["persistence"] = metrics(np.concatenate([tt[t - 1] for t in test_hours]), actual)
    groupings = {
        "historical_average_hour_daytype": lambda t: train & (hod == hod[t]) & (weekend == weekend[t]),
        "historical_average_hour_only": lambda t: train & (hod == hod[t]),
        "historical_average_hour_dayofweek": lambda t: train & (hod == hod[t]) & (day % 7 == day[t] % 7),
    }
    for name, mask in groupings.items():
        out[name] = metrics(np.concatenate([tt[mask(t)].mean(axis=0) for t in test_hours]), actual)
    for k, v in out.items():
        print(k, v)
    save_json(out, "forecast_baselines.json")


if __name__ == "__main__":
    main()

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(20261012)


@pytest.fixture(scope="session")
def clinical(rng):
    """One synthetic dataset with a column for every outcome type."""
    n = 240
    df = pd.DataFrame(
        {
            "age": rng.normal(55, 12, n),
            "group": rng.choice(["ctl", "trt"], n),
            "arm": rng.choice(["a", "b", "c"], n),
            "x": rng.normal(size=n),
            "exposure": rng.uniform(0.5, 2.0, n),
        }
    )
    lin = 0.03 * (df.age - 55) + 0.9 * (df.group == "trt") + 0.5 * df.x
    df["y"] = lin + rng.normal(size=n)
    df["y_skew"] = np.exp(rng.normal(size=n) * 0.8) + 0.5 * (df.group == "trt")
    df["ybin"] = (rng.random(n) < 1 / (1 + np.exp(-lin))).astype(int)
    df["ynom"] = rng.choice(["A", "B", "C"], n, p=[0.5, 0.3, 0.2])
    df["ycount"] = rng.poisson(np.exp(0.4 + 0.4 * (df.group == "trt") + 0.01 * (df.age - 55)) * df.exposure)
    df["ynb"] = rng.negative_binomial(2, 2 / (2 + np.exp(1 + 0.5 * (df.group == "trt"))))
    df["before"] = rng.normal(10, 2, n)
    df["after"] = df.before + rng.normal(0.6, 1.2, n)
    df["t1"], df["t2"], df["t3"] = (df.before + rng.normal(0, 1, n), df.before + rng.normal(0.4, 1, n), df.before + rng.normal(0.9, 1, n))
    df["b1"] = rng.integers(0, 2, n)
    df["b2"] = np.where(rng.random(n) < 0.75, df.b1, 1 - df.b1)
    df["b3"] = np.where(rng.random(n) < 0.65, df.b1, 1 - df.b1)
    df["r1"] = rng.integers(0, 4, n)
    df["r2"] = np.where(rng.random(n) < 0.7, df.r1, rng.integers(0, 4, n))
    df["r3"] = np.where(rng.random(n) < 0.6, df.r1, rng.integers(0, 4, n))
    df["o1"] = rng.integers(0, 3, n)
    df["o2"] = np.where(rng.random(n) < 0.7, df.o1, rng.integers(0, 3, n))
    df["time"] = rng.exponential(np.where(df.group == "trt", 15, 10))
    df["event"] = (rng.random(n) > 0.3).astype(int)
    return df

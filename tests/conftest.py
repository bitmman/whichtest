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
    # helper-test columns
    df["yord"] = pd.cut(df.x + 0.6 * (df.group == "trt") + rng.logistic(size=n), [-np.inf, -0.8, 0.6, np.inf], labels=[0, 1, 2]).astype(int)
    df["bmi"] = rng.normal(26, 4, n)
    df["y1"] = rng.normal(size=n) + 0.7 * (df.group == "trt") + 0.02 * df.age
    df["y2"] = 0.5 * df.y1 + rng.normal(size=n)
    df["y3"] = rng.normal(size=n) + 0.3 * df.x
    df["m1"] = rng.normal(100, 15, n)
    df["m2"] = df.m1 + 2 + rng.normal(0, 5, n)
    df["truth"] = (rng.random(n) < 0.4).astype(int)
    df["score_a"] = df.truth + rng.normal(size=n)
    df["score_b"] = 0.5 * df.truth + rng.normal(size=n)
    df["test_pos"] = (df.score_a > 0.5).astype(int)
    # judgment-test columns
    df["factor_b"] = rng.choice(["u", "v", "w"], n)
    df["site"] = rng.integers(0, 45, n)
    df["stage"] = rng.integers(1, 4, n)
    df["zcnt"] = np.where(rng.random(n) < 0.35, 0, rng.negative_binomial(2, 2 / (2 + np.exp(0.5 + 0.4 * (df.group == "trt")))))
    df["etype"] = np.where(df.event == 1, rng.choice([1, 2], n, p=[0.6, 0.4]), 0)
    for i in range(6):
        df[f"item{i}"] = rng.normal(size=n)
    f1, f2 = rng.normal(size=n), rng.normal(size=n)
    for i in range(3):
        df[f"item{i}"] += f1
        df[f"item{i + 3}"] += f2
    return df


@pytest.fixture(scope="session")
def longitudinal(rng):
    """60 subjects × 4 visits, long format, with every outcome type."""
    ns, nt = 60, 4
    d = pd.DataFrame({"id": np.repeat(range(ns), nt), "visit": np.tile(range(nt), ns)})
    d["group"] = np.repeat(rng.choice(["a", "b"], ns), nt)
    d["y"] = 1 + 0.5 * d.visit + 0.8 * (d.group == "b") + np.repeat(rng.normal(0, 1, ns), nt) + rng.normal(0, 0.7, len(d))
    d["ybin"] = (rng.random(len(d)) < 1 / (1 + np.exp(-(d.y - 2)))).astype(int)
    d["cnt"] = rng.poisson(np.exp(0.2 + 0.2 * d.visit))
    d["months"] = rng.uniform(1, 3, len(d))
    d["yord"] = pd.cut(d.y + rng.logistic(size=len(d)), [-np.inf, 1, 2.5, np.inf], labels=[0, 1, 2]).astype(int)
    d["ynom"] = rng.choice(["x", "y", "z"], len(d))
    return d


@pytest.fixture(scope="session")
def ratings(rng):
    d = pd.DataFrame({"patient": np.repeat(range(30), 3), "rater": np.tile(["r1", "r2", "r3"], 30)})
    d["score"] = np.repeat(rng.normal(50, 10, 30), 3) + rng.normal(0, 3, 90) + np.tile([0, 1, 2], 30)
    return d

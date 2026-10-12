"""Independent finite-sample oracles; no scipy/statsmodels test functions as references.

Expected probabilities use Fraction/combinatorics, not the pipeline cross-check.
Small no-tie samples deliberately keep rank tests on their exact branches.
"""
from fractions import Fraction
from itertools import combinations, product
from math import comb, exp, sqrt

import pandas as pd
import pytest

from whichtest import Spec, run


def binomial_probability(k, n, p, alternative):
    masses = [comb(n, j) * p**j * (1 - p)**(n - j) for j in range(n + 1)]
    if alternative == "less":
        return float(sum(masses[:k + 1]))
    if alternative == "greater":
        return float(sum(masses[k:]))
    return float(sum(q for q in masses if q <= masses[k]))


@pytest.mark.parametrize("alternative", ["two-sided", "less", "greater"])
@pytest.mark.parametrize("k,n,p", [(0, 8, Fraction(1, 2)), (3, 11, Fraction(1, 3)), (7, 9, Fraction(2, 5))])
def test_binomial_matches_enumerated_mass(k, n, p, alternative):
    r = run(Spec(test="binom_test", params={"k": k, "n": n, "p0": float(p)}, alternative=alternative))
    assert r.p == pytest.approx(binomial_probability(k, n, p, alternative), abs=1e-12)
    assert r.estimates[0].value == pytest.approx(k / n)


@pytest.mark.parametrize("alternative", ["two-sided", "less", "greater"])
def test_fisher_matches_enumerated_fixed_margins(alternative):
    # Fixed margins: rows 10, 14; columns 12, 12; top-left count = 1.
    masses = {a: Fraction(comb(12, a) * comb(12, 10 - a), comb(24, 10))
              for a in range(11)}
    selected = [q for a, q in masses.items()
                if (q <= masses[1] if alternative == "two-sided"
                    else a <= 1 if alternative == "less" else a >= 1)]
    r = run(Spec(test="two_by_two", params={"table": [[1, 9], [11, 3]]}, alternative=alternative))
    # Expected counts here are >=5, so force a sparse table instead below.
    # The primary dense-table directional contract is tested in contracts.py.
    assert r.extra["fisher"]["p"] == pytest.approx(float(sum(selected)), abs=1e-12)


@pytest.mark.parametrize("alternative", ["two-sided", "less", "greater"])
def test_sparse_fisher_primary_matches_enumeration(alternative):
    # Rows 4, 6; columns 5, 5: every expectation is below five.
    masses = {a: Fraction(comb(5, a) * comb(5, 4 - a), comb(10, 4)) for a in range(5)}
    selected = [q for a, q in masses.items()
                if (q <= masses[1] if alternative == "two-sided"
                    else a <= 1 if alternative == "less" else a >= 1)]
    r = run(Spec(test="two_by_two", params={"table": [[1, 3], [4, 2]]}, alternative=alternative))
    assert r.p == pytest.approx(float(sum(selected)), abs=1e-12)
    assert r.statistic == pytest.approx(Fraction(1, 6))


@pytest.mark.parametrize("b,c", [(1, 7), (4, 9), (10, 10)])
def test_exact_mcnemar_matches_discordant_sign_enumeration(b, c):
    r = run(Spec(test="mcnemar", params={"table": [[12, b], [c, 15]], "exact": True}))
    assert r.p == pytest.approx(binomial_probability(b, b + c, Fraction(1, 2), "two-sided"))
    assert r.estimates[1].value == pytest.approx(b / c)


@pytest.mark.parametrize("alternative", ["two-sided", "less", "greater"])
def test_mann_whitney_matches_label_assignment_enumeration(alternative):
    a, b = [1, 4, 6], [2, 3, 5, 7]
    observed = sum(x > y for x in a for y in b)
    population = a + b
    us = []
    for ids in combinations(range(7), 3):
        left = [population[i] for i in ids]
        right = [population[i] for i in range(7) if i not in ids]
        us.append(sum(x > y for x in left for y in right))
    lower = sum(u <= observed for u in us) / len(us)
    upper = sum(u >= observed for u in us) / len(us)
    expected = min(1, 2 * min(lower, upper)) if alternative == "two-sided" else lower if alternative == "less" else upper
    d = pd.DataFrame({"y": a + b, "g": ["a"] * 3 + ["b"] * 4})
    r = run(Spec(test="mann_whitney", variables={"outcome": "y", "group": "g"}, alternative=alternative), d)
    assert r.statistic == observed
    assert r.p == pytest.approx(expected, abs=1e-12)
    assert r.effect_size.value == pytest.approx(2 * observed / 12 - 1)


@pytest.mark.parametrize("alternative", ["two-sided", "less", "greater"])
def test_signed_rank_matches_all_sign_assignments(alternative):
    differences = [1, -2, 3, 4, -5, 6]
    wplus = 1 + 3 + 4 + 6
    ws = [sum(rank for rank, sign in zip(range(1, 7), signs) if sign)
          for signs in product([False, True], repeat=6)]
    lower = sum(w <= wplus for w in ws) / len(ws)
    upper = sum(w >= wplus for w in ws) / len(ws)
    p = min(1, 2 * min(lower, upper)) if alternative == "two-sided" else lower if alternative == "less" else upper
    r = run(Spec(test="one_sample_wilcoxon", variables={"outcome": "d"}, alternative=alternative), pd.DataFrame({"d": differences}))
    assert r.p == pytest.approx(p, abs=1e-12)
    assert r.statistic == (min(wplus, 21 - wplus) if alternative == "two-sided" else wplus)


@pytest.mark.parametrize("table", [[[12, 18, 30], [18, 22, 20]], [[0, 10, 20], [20, 10, 10]]])
def test_pearson_table_statistic_and_df2_tail_from_arithmetic(table):
    total = sum(map(sum, table))
    rows = list(map(sum, table))
    cols = [sum(row[j] for row in table) for j in range(3)]
    chi2 = sum((table[i][j] - rows[i]*cols[j]/total)**2 / (rows[i]*cols[j]/total)
               for i in range(2) for j in range(3))
    r = run(Spec(test="chi_sq_indep", params={"table": table}))
    assert r.statistic == pytest.approx(chi2)
    assert r.df == 2
    assert r.p == pytest.approx(exp(-chi2 / 2))  # chi-square(2) survival function
    assert r.effect_size.value == pytest.approx(sqrt(chi2 / total))
    # Both implementations must use the *same* original counts, including zeros.
    assert r.cross_check.agrees


def test_goodness_of_fit_df2_has_closed_form_tail():
    r = run(Spec(test="chi_gof", params={"observed": [10, 20, 30], "expected": [20, 20, 20]}))
    assert r.statistic == pytest.approx(10)
    assert r.p == pytest.approx(exp(-5))
    assert r.effect_size.value == pytest.approx(sqrt(1 / 6))


def test_friedman_known_complete_rank_sums():
    # All five subjects rank a < b < c. Rank sums 5,10,15 => Q=10, W=1.
    d = pd.DataFrame({"a": [1, 4, 2, 7, 3], "b": [2, 6, 3, 8, 5], "c": [4, 8, 6, 9, 7]})
    r = run(Spec(test="friedman", variables={"columns": ["a", "b", "c"]}), d)
    assert r.statistic == pytest.approx(10)
    assert r.p == pytest.approx(exp(-5))
    assert r.effect_size.value == pytest.approx(1)

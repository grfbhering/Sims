"""Two-good SimTrigo model.

Initial-period convention used here:
- Real margins m are calculated from the current-period information.
- The input `initial_inflation` is the inflation rate inherited from the
  period immediately preceding the simulation (p(0)). It is used to launch
  the first simulated period.
- At t=0, real margins are calculated from the initial market prices. The
  initial nominal margins are put on the nominal basis using the inherited
  inflation p_0: (1+n_0)=(1+m_0)(1+p_0). Domestic/import nominal margins are
  then exogenous/fixed; n1X and n2X remain endogenous from the export-price
  equations.
- P1X and P2X are exogenous export prices.

The aggregate price index uses the model's parameter weights x1 and x2.
The spectral radius is rho(A_t) = max_i |lambda_i(A_t)| and the maximum
profit rate is R_t = 1/rho(A_t) - 1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-12


def _as_path(x, T, shape_tail=()):
    arr = np.asarray(x, dtype=float)
    target = (T + 1,) + tuple(shape_tail)
    if arr.shape == target:
        return arr.copy()
    if arr.shape == shape_tail:
        return np.broadcast_to(arr, target).copy()
    if shape_tail == () and arr.ndim == 1 and arr.shape == (T + 1,):
        return arr.copy()
    raise ValueError(f"Expected shape {target} or {shape_tail}, got {arr.shape}.")


def _safe_div(num, den, name):
    if np.any(np.abs(den) < EPS):
        raise ZeroDivisionError(f"Zero/near-zero denominator in {name}.")
    return num / den


def _spectral_radius(A_t):
    eigenvalues = np.linalg.eigvals(A_t)
    return float(np.max(np.abs(eigenvalues)))


def _profit_rate_from_rho(rho):
    if abs(rho) < EPS:
        return np.inf
    return 1.0 / rho - 1.0


def _initial_real_margins(Pd, Pm, PX, W, e, l_i, cost):
    """Return real margins [d,m,X] for one sector at t=0."""
    return np.array([
        _safe_div(Pd - W * l_i, cost, "m_d_0") - 1.0,
        _safe_div(Pm - W * l_i, cost, "m_m_0") - 1.0,
        _safe_div(e * PX - W * l_i, cost, "m_X_0") - 1.0,
    ])


def simulate_two_good_model(
    T: int,
    A,
    l,
    X,
    W,
    e,
    phi,
    x1,
    x2,
    *,
    P1d0=None,
    P1m0=None,
    P2d0=None,
    P2m0=None,
    n1d=None,
    n1m=None,
    n2d=None,
    n2m=None,
    P1X=None,
    P2X=None,
    d11=None,
    d12=None,
    d21=None,
    d22=None,
    T1=None,
    T2=None,
    initial_inflation=0.0,
):
    """Simulate the two-good model for t=0,...,T.

    The input-cost coefficients d_ij determine the domestic/imported sourcing
    of each intermediate input. T_i is the sector-specific tax/price term.

    For t>=1 the production-price equations use input costs from t-1:

        P_i,t = (1+T_i,t) [ W_t l_i,t + C_i,t-1 (1+n_i,t) ],

    where C_i,t-1 combines domestic input prices with imported inputs priced
    at E_{t-1} P_j,t-1^X. The wage term W*l is NOT multiplied by (1+n).

    Real margins use current-period input costs and divide the market price
    by (1+T_i,t) before subtracting the wage bill, as in the supplied
    equations. At t=0, n_market(0)=m_market(0) initializes the nominal
    margins from the calculated real margins.
    """
    T = int(T)
    if T < 1:
        raise ValueError("T must be at least 1.")
    initial_inflation = float(initial_inflation)
    if initial_inflation <= -1:
        raise ValueError("initial_inflation must be greater than -1.")

    A = _as_path(A, T, (2, 2))
    l = _as_path(l, T, (2,))
    X = _as_path(X, T, (2,))
    W = _as_path(W, T)
    e = _as_path(e, T)
    phi = _as_path(phi, T, (2, 3))
    x1 = _as_path(x1, T)
    x2 = _as_path(x2, T)

    d11 = _as_path(1.0 if d11 is None else d11, T)
    d12 = _as_path(1.0 if d12 is None else d12, T)
    d21 = _as_path(1.0 if d21 is None else d21, T)
    d22 = _as_path(1.0 if d22 is None else d22, T)
    T1 = _as_path(0.0 if T1 is None else T1, T)
    T2 = _as_path(0.0 if T2 is None else T2, T)

    if np.any((d11 < 0) | (d11 > 1) | (d12 < 0) | (d12 > 1) | (d21 < 0) | (d21 > 1) | (d22 < 0) | (d22 > 1)):
        raise ValueError("Each d_ij must be between 0 and 1.")
    if np.any(T1 <= -1) or np.any(T2 <= -1):
        raise ValueError("T_i must be greater than -1.")

    initial_prices = {"P1d0": P1d0, "P1m0": P1m0, "P2d0": P2d0, "P2m0": P2m0}
    missing_initial = [k for k, v in initial_prices.items() if v is None]
    if missing_initial:
        raise ValueError(
            "Initial market prices are required to calculate the initial real margins: "
            + ", ".join(missing_initial)
        )

    required = {
        "n1d": n1d, "n1m": n1m, "n2d": n2d, "n2m": n2m,
        "P1X": P1X, "P2X": P2X,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(f"Missing required inputs: {', '.join(missing)}")

    n1d = _as_path(n1d, T)
    n1m = _as_path(n1m, T)
    n2d = _as_path(n2d, T)
    n2m = _as_path(n2m, T)
    P1X = _as_path(P1X, T)
    P2X = _as_path(P2X, T)

    if not np.allclose(x1 + x2, 1.0):
        raise ValueError("x1_t + x2_t must equal 1 for every t.")
    if not np.allclose(phi.sum(axis=2), 1.0):
        raise ValueError("Each sector's phi[d,m,X] shares must sum to 1 at every t.")

    names = [
        "P1", "P2", "P", "b1", "b2", "b", "w", "w_share",
        "p1", "p2", "p", "m1", "m2", "m",
        "n1", "n2", "n", "B", "R", "rho_A",
        "P1d", "P1m", "P1X", "P2d", "P2m", "P2X",
        "n1d", "n1m", "n1X", "n2d", "n2m", "n2X",
        "m1d", "m1m", "m1X", "m2d", "m2m", "m2X",
        "C1", "C2", "P1_cost", "P2_cost", "P1_from_markets", "P2_from_markets",
        "eq5_resid", "eq6_resid", "eq7_resid", "eq8_resid", "eq15_resid", "eq16_resid", "eq25_resid",
        "eq29_resid", "eq30_resid",
    ]
    s = {name: np.full(T + 1, np.nan, dtype=float) for name in names}

    a11, a12 = A[:, 0, 0], A[:, 0, 1]
    a21, a22 = A[:, 1, 0], A[:, 1, 1]
    l1, l2 = l[:, 0], l[:, 1]
    X1, X2 = X[:, 0], X[:, 1]

    def cost_from_previous(t):
        """C_i,t using t-period input coefficients/prices (for real margins)."""
        c1 = (
            d11[t] * a11[t] * s["P1"][t]
            + (1 - d11[t]) * a11[t] * e[t] * P1X[t]
            + d21[t] * a21[t] * s["P2"][t]
            + (1 - d21[t]) * a21[t] * e[t] * P2X[t]
        )
        c2 = (
            d12[t] * a12[t] * s["P1"][t]
            + (1 - d12[t]) * a12[t] * e[t] * P1X[t]
            + d22[t] * a22[t] * s["P2"][t]
            + (1 - d22[t]) * a22[t] * e[t] * P2X[t]
        )
        return c1, c2

    def cost_from_lagged(t):
        """C_i,t-1 entering the production-price equations at t."""
        c1 = (
            d11[t - 1] * a11[t - 1] * s["P1"][t - 1]
            + (1 - d11[t - 1]) * a11[t - 1] * e[t - 1] * P1X[t - 1]
            + d21[t - 1] * a21[t - 1] * s["P2"][t - 1]
            + (1 - d21[t - 1]) * a21[t - 1] * e[t - 1] * P2X[t - 1]
        )
        c2 = (
            d12[t - 1] * a12[t - 1] * s["P1"][t - 1]
            + (1 - d12[t - 1]) * a12[t - 1] * e[t - 1] * P1X[t - 1]
            + d22[t - 1] * a22[t - 1] * s["P2"][t - 1]
            + (1 - d22[t - 1]) * a22[t - 1] * e[t - 1] * P2X[t - 1]
        )
        return c1, c2

    for t in range(T + 1):
        s["rho_A"][t] = _spectral_radius(A[t])
        s["R"][t] = _profit_rate_from_rho(s["rho_A"][t])
        try:
            s["B"][t] = _safe_div(1.0, float(l[t] @ np.linalg.solve(np.eye(2) - A[t], X[t])), "B")
        except np.linalg.LinAlgError as exc:
            raise np.linalg.LinAlgError(f"I - A is singular at t={t}; cannot calculate B_t.") from exc

    # Initial sectoral prices are weighted averages of the initial
    # domestic, import and export-market prices. They are therefore not
    # independent inputs.
    s["P1d"][0] = float(P1d0)
    s["P1m"][0] = float(P1m0)
    s["P1X"][0] = float(P1X[0])
    s["P2d"][0] = float(P2d0)
    s["P2m"][0] = float(P2m0)
    s["P2X"][0] = float(P2X[0])
    s["P1_from_markets"][0] = (
        phi[0, 0, 0] * s["P1d"][0]
        + phi[0, 0, 1] * s["P1m"][0]
        + phi[0, 0, 2] * e[0] * s["P1X"][0]
    )
    s["P2_from_markets"][0] = (
        phi[0, 1, 0] * s["P2d"][0]
        + phi[0, 1, 1] * s["P2m"][0]
        + phi[0, 1, 2] * e[0] * s["P2X"][0]
    )
    s["P1"][0] = s["P1_from_markets"][0]
    s["P2"][0] = s["P2_from_markets"][0]
    s["P"][0] = x1[0] * s["P1"][0] + x2[0] * s["P2"][0]

    # Current-period input costs for the initial real-margin equations.
    c1_real0, c2_real0 = cost_from_previous(0)
    s["C1"][0], s["C2"][0] = c1_real0, c2_real0

    inactive_good2 = (
        np.isclose(x2[0], 0.0) and np.isclose(l2[0], 0.0) and np.isclose(X2[0], 0.0)
        and np.isclose(a12[0], 0.0) and np.isclose(a21[0], 0.0) and np.isclose(a22[0], 0.0)
    )

    def initial_margin(price, PX, li, cost, tax):
        return _safe_div(price / (1 + tax) - W[0] * li, cost, "m_0") - 1.0

    m10 = np.array([
        initial_margin(P1d0, P1X[0], l1[0], c1_real0, T1[0]),
        initial_margin(P1m0, P1X[0], l1[0], c1_real0, T1[0]),
        initial_margin(e[0] * P1X[0], P1X[0], l1[0], c1_real0, T1[0]),
    ])
    m20 = (
        np.full(3, np.nan) if inactive_good2 else np.array([
            initial_margin(P2d0, P2X[0], l2[0], c2_real0, T2[0]),
            initial_margin(P2m0, P2X[0], l2[0], c2_real0, T2[0]),
            initial_margin(e[0] * P2X[0], P2X[0], l2[0], c2_real0, T2[0]),
        ])
    )

    s["m1d"][0], s["m1m"][0], s["m1X"][0] = m10
    s["m2d"][0], s["m2m"][0], s["m2X"][0] = m20

    # Initial nominal margins for exogenous domestic/import markets inherit
    # the inflation rate from the period immediately before t=0:
    # (1+n_0) = (1+m_0)(1+p_0). These nominal margins are then fixed unless
    # an explicit shock changes their exogenous path. Export margins are
    # endogenous and are obtained directly from the export-price equation.
    p0 = initial_inflation
    s["n1d"][0] = (1.0 + m10[0]) * (1.0 + p0) - 1.0
    s["n1m"][0] = (1.0 + m10[1]) * (1.0 + p0) - 1.0
    s["n2d"][0] = (1.0 + m20[0]) * (1.0 + p0) - 1.0 if not inactive_good2 else 0.0
    s["n2m"][0] = (1.0 + m20[1]) * (1.0 + p0) - 1.0 if not inactive_good2 else 0.0
    # Export nominal margins are endogenous, but their t=0 value must be on
    # the same nominal basis as the t>=1 export-margin equation.
    s["n1X"][0] = (1.0 + m10[2]) * (1.0 + p0) - 1.0
    s["n2X"][0] = ((1.0 + m20[2]) * (1.0 + p0) - 1.0) if not inactive_good2 else 0.0

    s["n1"][0] = phi[0, 0, 0] * (1 + s["n1d"][0]) + phi[0, 0, 1] * (1 + s["n1m"][0]) + phi[0, 0, 2] * (1 + s["n1X"][0]) - 1
    s["n2"][0] = 0.0 if inactive_good2 else phi[0, 1, 0] * (1 + s["n2d"][0]) + phi[0, 1, 1] * (1 + s["n2m"][0]) + phi[0, 1, 2] * (1 + s["n2X"][0]) - 1
    s["n"][0] = x1[0] * (1 + s["n1"][0]) + x2[0] * (1 + s["n2"][0]) - 1

    s["P1_cost"][0] = (1 + T1[0]) * (W[0] * l1[0] + c1_real0 * (1 + s["n1"][0]))
    s["P2_cost"][0] = (1 + T2[0]) * (W[0] * l2[0] + c2_real0 * (1 + s["n2"][0]))

    s["b1"][0] = _safe_div(W[0], s["P1"][0], "b1_0")
    s["b2"][0] = np.nan if inactive_good2 and np.isclose(s["P2"][0], 0.0) else _safe_div(W[0], s["P2"][0], "b2_0")
    s["b"][0] = _safe_div(W[0], s["P"][0], "b_0")
    s["w"][0] = 0.0
    s["w_share"][0] = _safe_div(W[0] * float(l[0] @ X[0]), float(np.array([s["P1"][0], s["P2"][0]]) @ ((np.eye(2) - A[0]) @ X[0])), "w_share_0")
    # p(0) is the inflation rate inherited from the period immediately before
    # the simulation starts. It is an observed historical rate, not a rate
    # calculated from P(0), because P(-1) is outside the simulation horizon.
    s["p1"][0] = np.nan
    s["p2"][0] = np.nan
    s["p"][0] = initial_inflation

    # Aggregate real margins are calculated from the market-specific real
    # margins. With p(0) nonzero, these need not equal the nominal margins.
    s["m1"][0] = phi[0, 0, 0] * s["m1d"][0] + phi[0, 0, 1] * s["m1m"][0] + phi[0, 0, 2] * s["m1X"][0]
    s["m2"][0] = np.nan if inactive_good2 else phi[0, 1, 0] * s["m2d"][0] + phi[0, 1, 1] * s["m2m"][0] + phi[0, 1, 2] * s["m2X"][0]
    s["m"][0] = x1[0] * s["m1"][0] + x2[0] * (0.0 if inactive_good2 else s["m2"][0])

    s["n1d"][1:] = n1d[1:]; s["n1m"][1:] = n1m[1:]
    s["n2d"][1:] = n2d[1:]; s["n2m"][1:] = n2m[1:]
    s["P1X"][1:] = P1X[1:]; s["P2X"][1:] = P2X[1:]

    for t in range(1, T + 1):
        prev = t - 1
        inactive_good2 = (
            np.isclose(x2[t], 0.0) and np.isclose(l2[t], 0.0) and np.isclose(X2[t], 0.0)
            and np.isclose(a12[t], 0.0) and np.isclose(a21[t], 0.0) and np.isclose(a22[t], 0.0)
        )
        c1, c2 = cost_from_lagged(t)
        s["C1"][t], s["C2"][t] = c1, c2

        # Export nominal margins are implied by the export-price equations.
        s["n1X"][t] = _safe_div(e[t] * P1X[t] / (1 + T1[t]) - W[t] * l1[t], c1, "n1X") - 1.0
        s["n2X"][t] = np.nan if inactive_good2 else _safe_div(e[t] * P2X[t] / (1 + T2[t]) - W[t] * l2[t], c2, "n2X") - 1.0

        s["n1"][t] = phi[t, 0, 0] * (1 + s["n1d"][t]) + phi[t, 0, 1] * (1 + s["n1m"][t]) + phi[t, 0, 2] * (1 + s["n1X"][t]) - 1
        s["n2"][t] = 0.0 if inactive_good2 else phi[t, 1, 0] * (1 + s["n2d"][t]) + phi[t, 1, 1] * (1 + s["n2m"][t]) + phi[t, 1, 2] * (1 + s["n2X"][t]) - 1

        # Market-specific prices. The markup applies only to C_i; W*l_i is
        # added separately inside the tax-adjusted bracket.
        s["P1d"][t] = (1 + T1[t]) * (W[t] * l1[t] + c1 * (1 + s["n1d"][t]))
        s["P1m"][t] = (1 + T1[t]) * (W[t] * l1[t] + c1 * (1 + s["n1m"][t]))
        s["P2d"][t] = (1 + T2[t]) * (W[t] * l2[t] + c2 * (1 + s["n2d"][t]))
        s["P2m"][t] = (1 + T2[t]) * (W[t] * l2[t] + c2 * (1 + s["n2m"][t]))

        s["P1_from_markets"][t] = phi[t, 0, 0] * s["P1d"][t] + phi[t, 0, 1] * s["P1m"][t] + phi[t, 0, 2] * e[t] * P1X[t]
        s["P2_from_markets"][t] = phi[t, 1, 0] * s["P2d"][t] + phi[t, 1, 1] * s["P2m"][t] + phi[t, 1, 2] * e[t] * P2X[t]
        s["P1"][t] = s["P1_from_markets"][t]
        s["P2"][t] = s["P2_from_markets"][t]
        s["P1_cost"][t] = (1 + T1[t]) * (W[t] * l1[t] + c1 * (1 + s["n1"][t]))
        s["P2_cost"][t] = (1 + T2[t]) * (W[t] * l2[t] + c2 * (1 + s["n2"][t]))
        s["P"][t] = x1[t] * s["P1"][t] + x2[t] * s["P2"][t]
        s["n"][t] = x1[t] * (1 + s["n1"][t]) + x2[t] * (1 + s["n2"][t]) - 1

        s["b1"][t] = _safe_div(W[t], s["P1"][t], "b1")
        s["b2"][t] = np.nan if inactive_good2 and np.isclose(s["P2"][t], 0.0) else _safe_div(W[t], s["P2"][t], "b2")
        s["b"][t] = _safe_div(W[t], s["P"][t], "b")
        s["w_share"][t] = _safe_div(W[t] * float(l[t] @ X[t]), float(np.array([s["P1"][t], s["P2"][t]]) @ ((np.eye(2) - A[t]) @ X[t])), "w_share")
        s["w"][t] = _safe_div(W[t], W[prev], "w") - 1
        s["p1"][t] = _safe_div(s["P1"][t], s["P1"][prev], "p1") - 1
        s["p2"][t] = np.nan if inactive_good2 and np.isclose(s["P2"][prev], 0.0) else _safe_div(s["P2"][t], s["P2"][prev], "p2") - 1
        s["p"][t] = _safe_div(s["P"][t], s["P"][prev], "p") - 1

        # Real margins: current-period C_i and the tax-adjusted market price.
        c1r, c2r = cost_from_previous(t)
        s["m1d"][t] = _safe_div(s["P1d"][t] / (1 + T1[t]) - W[t] * l1[t], c1r, "m1d") - 1
        s["m1m"][t] = _safe_div(s["P1m"][t] / (1 + T1[t]) - W[t] * l1[t], c1r, "m1m") - 1
        s["m1X"][t] = _safe_div(e[t] * P1X[t] / (1 + T1[t]) - W[t] * l1[t], c1r, "m1X") - 1
        s["m2d"][t] = np.nan if inactive_good2 else _safe_div(s["P2d"][t] / (1 + T2[t]) - W[t] * l2[t], c2r, "m2d") - 1
        s["m2m"][t] = np.nan if inactive_good2 else _safe_div(s["P2m"][t] / (1 + T2[t]) - W[t] * l2[t], c2r, "m2m") - 1
        s["m2X"][t] = np.nan if inactive_good2 else _safe_div(e[t] * P2X[t] / (1 + T2[t]) - W[t] * l2[t], c2r, "m2X") - 1
        s["m1"][t] = phi[t, 0, 0] * s["m1d"][t] + phi[t, 0, 1] * s["m1m"][t] + phi[t, 0, 2] * s["m1X"][t]
        s["m2"][t] = np.nan if inactive_good2 else phi[t, 1, 0] * s["m2d"][t] + phi[t, 1, 1] * s["m2m"][t] + phi[t, 1, 2] * s["m2X"][t]
        s["m"][t] = x1[t] * s["m1"][t] + x2[t] * (0.0 if inactive_good2 else s["m2"][t])

        s["eq5_resid"][t] = s["P1"][t] - s["P1_cost"][t]
        s["eq6_resid"][t] = s["P2"][t] - s["P2_cost"][t]
        s["eq7_resid"][t] = s["P1"][t] - s["P1_from_markets"][t]
        s["eq8_resid"][t] = s["P2"][t] - s["P2_from_markets"][t]
        implied_n1 = phi[t, 0, 0] * (1 + s["n1d"][t]) + phi[t, 0, 1] * (1 + s["n1m"][t]) + phi[t, 0, 2] * (1 + s["n1X"][t]) - 1
        implied_n2 = 0.0 if inactive_good2 else phi[t, 1, 0] * (1 + s["n2d"][t]) + phi[t, 1, 1] * (1 + s["n2m"][t]) + phi[t, 1, 2] * (1 + s["n2X"][t]) - 1
        s["eq15_resid"][t] = s["n1"][t] - implied_n1
        s["eq16_resid"][t] = s["n2"][t] - implied_n2
        s["eq25_resid"][t] = s["n"][t] - (x1[t] * (1 + s["n1"][t]) + x2[t] * (1 + s["n2"][t]) - 1)
        s["eq29_resid"][t] = np.nan
        s["eq30_resid"][t] = s["w"][t] - (W[t] / W[prev] - 1)

    s["eq7_resid"][0] = s["P1"][0] - s["P1_from_markets"][0]
    s["eq8_resid"][0] = s["P2"][0] - s["P2_from_markets"][0]
    s["eq15_resid"][0] = 0.0; s["eq16_resid"][0] = 0.0; s["eq25_resid"][0] = 0.0; s["eq29_resid"][0] = np.nan

    df = pd.DataFrame({
        "period": np.arange(T + 1), "x1": x1, "x2": x2, "W": W, "e": e,
        "X1": X1, "X2": X2, "a11": a11, "a12": a12, "a21": a21, "a22": a22,
        "l1": l1, "l2": l2,
        "d11": d11, "d12": d12, "d21": d21, "d22": d22, "T1": T1, "T2": T2,
        **s,
    })

    def maxabs(name, start=0):
        vals = np.abs(s[name][start:])
        return float(np.nanmax(vals)) if np.any(np.isfinite(vals)) else np.nan

    checks = {
        "max_abs_eq5": maxabs("eq5_resid", 1), "max_abs_eq6": maxabs("eq6_resid", 1),
        "max_abs_eq7": maxabs("eq7_resid", 0), "max_abs_eq8": maxabs("eq8_resid", 0),
        "max_abs_eq15": maxabs("eq15_resid", 1), "max_abs_eq16": maxabs("eq16_resid", 1),
        "max_abs_eq25": maxabs("eq25_resid", 1), "max_abs_eq29": maxabs("eq29_resid", 1),
        "max_abs_eq30": maxabs("eq30_resid", 1), "max_rho_A": float(np.nanmax(s["rho_A"])),
        "R_from_max_rho": _profit_rate_from_rho(float(np.nanmax(s["rho_A"]))), "max_R": float(np.nanmax(s["R"])),
    }
    return df, checks

def simulate_one_good_special_case(
    T: int, a, l1, X1, W, e, phi1, n1d, n1m, P1X, *,
    P1d0=None, P1m0=None, d11=None, T1=None, initial_inflation=0.0,
):
    """One-good special case embedded in the two-good model."""
    T = int(T)
    phi1 = _as_path(phi1, T, (3,))
    a = _as_path(a, T); l1 = _as_path(l1, T); X1 = _as_path(X1, T)
    W = _as_path(W, T); e = _as_path(e, T)
    n1d = _as_path(n1d, T); n1m = _as_path(n1m, T); P1X = _as_path(P1X, T)
    if P1d0 is None or P1m0 is None:
        raise ValueError("P1d0 and P1m0 are required for initial real-margin initialization.")

    A = np.zeros((T + 1, 2, 2)); A[:, 0, 0] = a
    l = np.column_stack([l1, np.zeros(T + 1)])
    X = np.column_stack([X1, np.zeros(T + 1)])
    x1 = np.ones(T + 1); x2 = np.zeros(T + 1)
    phi = np.zeros((T + 1, 2, 3)); phi[:, 0, :] = phi1; phi[:, 1, 0] = 1.0
    zero = np.zeros(T + 1)

    df, checks = simulate_two_good_model(
        T=T, A=A, l=l, X=X, W=W, e=e, phi=phi, x1=x1, x2=x2,
        P1d0=P1d0, P1m0=P1m0, P2d0=1.0, P2m0=1.0,
        n1d=n1d, n1m=n1m, n2d=zero, n2m=zero, P1X=P1X, P2X=np.ones(T + 1),
        d11=np.ones(T + 1) if d11 is None else d11, d12=np.ones(T + 1), d21=np.ones(T + 1), d22=np.ones(T + 1),
        T1=np.zeros(T + 1) if T1 is None else T1, T2=np.zeros(T + 1),
        initial_inflation=initial_inflation,
    )
    df["P_old"] = df["P1"]; df["P_d_old"] = df["P1d"]; df["P_a_old"] = df["P1m"]; df["P_x_old"] = df["P1X"]
    df["N_d_old"] = df["n1d"]; df["N_a_old"] = df["n1m"]; df["N_x_old"] = df["n1X"]
    return df, checks


if __name__ == "__main__":
    T = 10
    A = np.array([[0.10, 0.05], [0.05, 0.10]])
    l = np.array([0.20, 0.25]); X = np.array([1.0, 1.0])
    W = np.ones(T + 1); e = np.ones(T + 1)
    phi = np.zeros((T + 1, 2, 3)); phi[:, 0, :] = [0.5, 0.2, 0.3]; phi[:, 1, :] = [0.4, 0.3, 0.3]
    x1 = np.full(T + 1, 0.5); x2 = 1 - x1
    n1d = np.full(T + 1, 0.10); n1m = np.full(T + 1, 0.10); n2d = np.full(T + 1, 0.10); n2m = np.full(T + 1, 0.10)
    P1X = np.full(T + 1, 1.0); P2X = np.full(T + 1, 1.0)
    df, checks = simulate_two_good_model(
        T, A, l, X, W, e, phi, x1, x2,
        P1d0=1.0, P1m0=1.0, P2d0=1.0, P2m0=1.0,
        n1d=n1d, n1m=n1m, n2d=n2d, n2m=n2m, P1X=P1X, P2X=P2X,
    )
    print(df[["period", "rho_A", "R", "P1", "P2", "P", "n1d", "m1d", "n1X", "m1X"]])
    print(checks)

"""Two-good SimTrigo model.

Initial-period convention used here:
- Real margins m are calculated from the current-period information.
- At t=0, the nominal margins are initialized from the corresponding real
  margins: n(0) = m(0). This supplies the initial values of the exogenous
  nominal margins and initializes the endogenous export margins as well.
- From t=1 onward, n1d, n1m, n2d and n2m are exogenous; n1X and n2X are
  endogenous from the export-price equations.
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
    P10=1.0,
    P20=1.0,
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
):
    """Simulate the two-good model for t=0,...,T.

    P1d0/P1m0/P2d0/P2m0 are the initial market-specific prices. They are
    required because the corrected real-margin equations calculate m(0)
    directly from current-period market prices. The model then imposes
    n_market(0) = m_market(0).

    For t>=1, n1d, n1m, n2d and n2m are exogenous paths. n1X and n2X are
    endogenous from the export-price equations.
    """
    T = int(T)
    if T < 1:
        raise ValueError("T must be at least 1.")

    A = _as_path(A, T, (2, 2))
    l = _as_path(l, T, (2,))
    X = _as_path(X, T, (2,))
    W = _as_path(W, T)
    e = _as_path(e, T)
    phi = _as_path(phi, T, (2, 3))
    x1 = _as_path(x1, T)
    x2 = _as_path(x2, T)

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
        "P1_cost", "P2_cost", "P1_from_markets", "P2_from_markets",
        "eq5_resid", "eq6_resid", "eq7_resid", "eq8_resid",
        "eq15_resid", "eq16_resid", "eq25_resid",
        "eq29_resid", "eq30_resid",
    ]
    s = {name: np.full(T + 1, np.nan, dtype=float) for name in names}

    a11, a12 = A[:, 0, 0], A[:, 0, 1]
    a21, a22 = A[:, 1, 0], A[:, 1, 1]
    l1, l2 = l[:, 0], l[:, 1]
    X1, X2 = X[:, 0], X[:, 1]

    # Equations (1)-(2): spectral radius and maximum profit rate.
    for t in range(T + 1):
        s["rho_A"][t] = _spectral_radius(A[t])
        s["R"][t] = _profit_rate_from_rho(s["rho_A"][t])

    # Equation (4): B_t = 1 / {l_t [I-A_t]^{-1} X_t}.
    for t in range(T + 1):
        try:
            s["B"][t] = _safe_div(1.0, float(l[t] @ np.linalg.solve(np.eye(2) - A[t], X[t])), "B")
        except np.linalg.LinAlgError as exc:
            raise np.linalg.LinAlgError(
                f"I - A is singular at t={t}; cannot calculate B_t."
            ) from exc

    # Initial sector prices and aggregate price index.
    s["P1"][0] = float(P10)
    s["P2"][0] = float(P20)
    s["P"][0] = x1[0] * s["P1"][0] + x2[0] * s["P2"][0]

    # Current-period input costs used by the corrected real-margin equations.
    c1_real0 = a11[0] * s["P1"][0] + a21[0] * s["P2"][0]
    c2_real0 = a12[0] * s["P1"][0] + a22[0] * s["P2"][0]

    m10 = _initial_real_margins(P1d0, P1m0, P1X[0], W[0], e[0], l1[0], c1_real0)
    inactive_good2 = (
        np.isclose(x2[0], 0.0) and np.isclose(l2[0], 0.0) and np.isclose(X2[0], 0.0)
        and np.isclose(a12[0], 0.0) and np.isclose(a21[0], 0.0) and np.isclose(a22[0], 0.0)
    )
    m20 = (
        np.full(3, np.nan) if inactive_good2
        else _initial_real_margins(P2d0, P2m0, P2X[0], W[0], e[0], l2[0], c2_real0)
    )

    s["m1d"][0], s["m1m"][0], s["m1X"][0] = m10
    s["m2d"][0], s["m2m"][0], s["m2X"][0] = m20

    # Initial condition requested by the user: nominal = real margins.
    s["n1d"][0], s["n1m"][0] = m10[0], m10[1]
    s["n2d"][0], s["n2m"][0] = m20[0], m20[1]
    s["n1X"][0] = m10[2]
    s["n2X"][0] = m20[2]

    # The initial market prices are observed/given values.
    s["P1d"][0], s["P1m"][0], s["P1X"][0] = float(P1d0), float(P1m0), float(P1X[0])
    s["P2d"][0], s["P2m"][0], s["P2X"][0] = float(P2d0), float(P2m0), float(P2X[0])

    s["n1"][0] = (
        phi[0, 0, 0] * (1 + s["n1d"][0])
        + phi[0, 0, 1] * (1 + s["n1m"][0])
        + phi[0, 0, 2] * (1 + s["n1X"][0]) - 1
    )
    s["n2"][0] = (
        0.0 if inactive_good2 else
        phi[0, 1, 0] * (1 + s["n2d"][0])
        + phi[0, 1, 1] * (1 + s["n2m"][0])
        + phi[0, 1, 2] * (1 + s["n2X"][0]) - 1
    )
    s["n"][0] = x1[0] * (1 + s["n1"][0]) + x2[0] * (1 + s["n2"][0]) - 1

    # Given market prices should reproduce the given sector prices.
    s["P1_from_markets"][0] = phi[0, 0, 0] * P1d0 + phi[0, 0, 1] * P1m0 + phi[0, 0, 2] * e[0] * P1X[0]
    s["P2_from_markets"][0] = phi[0, 1, 0] * P2d0 + phi[0, 1, 1] * P2m0 + phi[0, 1, 2] * e[0] * P2X[0]
    s["P1_cost"][0] = (1 + s["n1"][0]) * c1_real0 + W[0] * l1[0]
    s["P2_cost"][0] = (1 + s["n2"][0]) * c2_real0 + W[0] * l2[0]

    s["b1"][0] = _safe_div(W[0], s["P1"][0], "b1_0")
    s["b2"][0] = np.nan if inactive_good2 and np.isclose(s["P2"][0], 0.0) else _safe_div(W[0], s["P2"][0], "b2_0")
    s["b"][0] = _safe_div(W[0], s["P"][0], "b_0")
    # Initial-period growth rates are defined as zero: there is no prior period.
    s["w"][0] = 0.0
    s["w_share"][0] = _safe_div(
        W[0] * float(l[0] @ X[0]),
        float(np.array([s["P1"][0], s["P2"][0]]) @ ((np.eye(2) - A[0]) @ X[0])),
        "w_share_0",
    )
    s["p1"][0] = 0.0
    s["p2"][0] = 0.0
    s["p"][0] = 0.0
    # Aggregate real margins at t=0 are calculated directly from the
    # market-specific real margins and the market shares. Do not initialize
    # m1/m2/m from nominal margins: the real margins are primary endogenous
    # variables at the initial date. Because p_0=0 and n_0=m_0 by the
    # initial-condition convention, the corresponding nominal margins coincide
    # numerically with these calculated real margins.
    s["m1"][0] = (
        phi[0, 0, 0] * s["m1d"][0]
        + phi[0, 0, 1] * s["m1m"][0]
        + phi[0, 0, 2] * s["m1X"][0]
    )
    s["m2"][0] = (
        np.nan if inactive_good2 else
        phi[0, 1, 0] * s["m2d"][0]
        + phi[0, 1, 1] * s["m2m"][0]
        + phi[0, 1, 2] * s["m2X"][0]
    )
    s["m"][0] = x1[0] * s["m1"][0] + x2[0] * (0.0 if inactive_good2 else s["m2"][0])

    # t>=1: exogenous domestic/import nominal margins; export margins endogenous.
    s["n1d"][1:] = n1d[1:]
    s["n1m"][1:] = n1m[1:]
    s["n2d"][1:] = n2d[1:]
    s["n2m"][1:] = n2m[1:]
    s["P1X"][1:] = P1X[1:]
    s["P2X"][1:] = P2X[1:]

    for t in range(1, T + 1):
        prev = t - 1
        inactive_good2 = (
            np.isclose(x2[t], 0.0) and np.isclose(l2[t], 0.0) and np.isclose(X2[t], 0.0)
            and np.isclose(a12[t], 0.0) and np.isclose(a21[t], 0.0) and np.isclose(a22[t], 0.0)
        )

        c1 = a11[prev] * s["P1"][prev] + a21[prev] * s["P2"][prev]
        c2 = a12[prev] * s["P1"][prev] + a22[prev] * s["P2"][prev]

        # Equations (11) and (14): endogenous export nominal margins.
        s["n1X"][t] = _safe_div(e[t] * P1X[t] - W[t] * l1[t], c1, "n1X") - 1.0
        s["n2X"][t] = np.nan if inactive_good2 else _safe_div(e[t] * P2X[t] - W[t] * l2[t], c2, "n2X") - 1.0

        s["n1"][t] = phi[t, 0, 0] * (1 + s["n1d"][t]) + phi[t, 0, 1] * (1 + s["n1m"][t]) + phi[t, 0, 2] * (1 + s["n1X"][t]) - 1
        s["n2"][t] = 0.0 if inactive_good2 else phi[t, 1, 0] * (1 + s["n2d"][t]) + phi[t, 1, 1] * (1 + s["n2m"][t]) + phi[t, 1, 2] * (1 + s["n2X"][t]) - 1

        # Equations (17)-(22).
        s["P1d"][t] = (1 + s["n1d"][t]) * c1 + W[t] * l1[t]
        s["P1m"][t] = (1 + s["n1m"][t]) * c1 + W[t] * l1[t]
        s["P2d"][t] = (1 + s["n2d"][t]) * c2 + W[t] * l2[t]
        s["P2m"][t] = (1 + s["n2m"][t]) * c2 + W[t] * l2[t]

        s["P1_from_markets"][t] = phi[t, 0, 0] * s["P1d"][t] + phi[t, 0, 1] * s["P1m"][t] + phi[t, 0, 2] * e[t] * P1X[t]
        s["P2_from_markets"][t] = phi[t, 1, 0] * s["P2d"][t] + phi[t, 1, 1] * s["P2m"][t] + phi[t, 1, 2] * e[t] * P2X[t]
        s["P1"][t] = s["P1_from_markets"][t]
        s["P2"][t] = s["P2_from_markets"][t]
        s["P1_cost"][t] = (1 + s["n1"][t]) * c1 + W[t] * l1[t]
        s["P2_cost"][t] = (1 + s["n2"][t]) * c2 + W[t] * l2[t]
        s["P"][t] = x1[t] * s["P1"][t] + x2[t] * s["P2"][t]
        s["n"][t] = x1[t] * (1 + s["n1"][t]) + x2[t] * (1 + s["n2"][t]) - 1

        s["b1"][t] = _safe_div(W[t], s["P1"][t], "b1")
        s["b2"][t] = np.nan if inactive_good2 and np.isclose(s["P2"][t], 0.0) else _safe_div(W[t], s["P2"][t], "b2")
        s["b"][t] = _safe_div(W[t], s["P"][t], "b")
        s["w_share"][t] = _safe_div(
            W[t] * float(l[t] @ X[t]),
            float(np.array([s["P1"][t], s["P2"][t]]) @ ((np.eye(2) - A[t]) @ X[t])),
            "w_share",
        )
        s["w"][t] = _safe_div(W[t], W[prev], "w") - 1

        s["p1"][t] = _safe_div(s["P1"][t], s["P1"][prev], "p1") - 1
        s["p2"][t] = np.nan if inactive_good2 and np.isclose(s["P2"][prev], 0.0) else _safe_div(s["P2"][t], s["P2"][prev], "p2") - 1
        s["p"][t] = _safe_div(s["P"][t], s["P"][prev], "p") - 1

        # Corrected real margins: current prices and current input costs.
        c1r = a11[t] * s["P1"][t] + a21[t] * s["P2"][t]
        c2r = a12[t] * s["P1"][t] + a22[t] * s["P2"][t]
        s["m1d"][t] = _safe_div(s["P1d"][t] - W[t] * l1[t], c1r, "m1d") - 1
        s["m1m"][t] = _safe_div(s["P1m"][t] - W[t] * l1[t], c1r, "m1m") - 1
        s["m1X"][t] = _safe_div(e[t] * P1X[t] - W[t] * l1[t], c1r, "m1X") - 1
        s["m2d"][t] = np.nan if inactive_good2 else _safe_div(s["P2d"][t] - W[t] * l2[t], c2r, "m2d") - 1
        s["m2m"][t] = np.nan if inactive_good2 else _safe_div(s["P2m"][t] - W[t] * l2[t], c2r, "m2m") - 1
        s["m2X"][t] = np.nan if inactive_good2 else _safe_div(e[t] * P2X[t] - W[t] * l2[t], c2r, "m2X") - 1
        s["m1"][t] = (
            phi[t, 0, 0] * s["m1d"][t]
            + phi[t, 0, 1] * s["m1m"][t]
            + phi[t, 0, 2] * s["m1X"][t]
        )
        s["m2"][t] = np.nan if inactive_good2 else (
            phi[t, 1, 0] * s["m2d"][t]
            + phi[t, 1, 1] * s["m2m"][t]
            + phi[t, 1, 2] * s["m2X"][t]
        )
        # Aggregate real margin is explicitly the x-weighted aggregation of
        # the sectoral real margins.
        s["m"][t] = x1[t] * s["m1"][t] + x2[t] * (0.0 if inactive_good2 else s["m2"][t])

        # Residuals.
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

    # Initial residuals that are meaningful for observed initial conditions.
    s["eq7_resid"][0] = s["P1"][0] - s["P1_from_markets"][0]
    s["eq8_resid"][0] = s["P2"][0] - s["P2_from_markets"][0]
    s["eq15_resid"][0] = 0.0
    s["eq16_resid"][0] = 0.0
    s["eq25_resid"][0] = 0.0
    s["eq29_resid"][0] = np.nan

    df = pd.DataFrame({
        "period": np.arange(T + 1), "x1": x1, "x2": x2, "W": W, "e": e,
        "X1": X1, "X2": X2, "a11": a11, "a12": a12, "a21": a21, "a22": a22,
        "l1": l1, "l2": l2, **s,
    })

    def maxabs(name, start=0):
        vals = np.abs(s[name][start:])
        return float(np.nanmax(vals)) if np.any(np.isfinite(vals)) else np.nan

    checks = {
        "max_abs_eq5": maxabs("eq5_resid", 1),
        "max_abs_eq6": maxabs("eq6_resid", 1),
        "max_abs_eq7": maxabs("eq7_resid", 0),
        "max_abs_eq8": maxabs("eq8_resid", 0),
        "max_abs_eq15": maxabs("eq15_resid", 1),
        "max_abs_eq16": maxabs("eq16_resid", 1),
        "max_abs_eq25": maxabs("eq25_resid", 1),
        "max_abs_eq29": maxabs("eq29_resid", 1),
        "max_abs_eq30": maxabs("eq30_resid", 1),
        "max_rho_A": float(np.nanmax(s["rho_A"])),
        "R_from_max_rho": _profit_rate_from_rho(float(np.nanmax(s["rho_A"]))),
        "max_R": float(np.nanmax(s["R"])),
    }
    return df, checks


def simulate_one_good_special_case(
    T: int, a, l1, X1, W, e, phi1, n1d, n1m, P1X, *,
    P10=1.0, P1d0=None, P1m0=None,
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
        P10=P10, P20=1.0, P1d0=P1d0, P1m0=P1m0, P2d0=1.0, P2m0=1.0,
        n1d=n1d, n1m=n1m, n2d=zero, n2m=zero, P1X=P1X, P2X=np.ones(T + 1),
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
        P10=1.0, P20=1.0, P1d0=1.0, P1m0=1.0, P2d0=1.0, P2m0=1.0,
        n1d=n1d, n1m=n1m, n2d=n2d, n2m=n2m, P1X=P1X, P2X=P2X,
    )
    print(df[["period", "rho_A", "R", "P1", "P2", "P", "n1d", "m1d", "n1X", "m1X"]])
    print(checks)

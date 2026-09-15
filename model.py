"""
SimTrigo model engine.

The dashboard is the interface; this module contains only the simulation
logic and the initial-value definitions used by the dashboard.
"""

import numpy as np
import pandas as pd


OUTPUTS = [
    "a", "L", "N_a", "N_d", "P", "W", "P_d", "P_a", "P_x",
    "N_x", "E", "w", "p", "b", "r", "WE", "N", "M", "M_x",
    "M_a", "M_d", "P_ap", "P_dp", "P_xp"
]

# Single source of truth for primitive initial conditions.
INITIAL_VALUES = {
    "a_0": 0.5,
    "L_0": 0.125,
    "P_0": 1.0,
    "W_0": 1.0,
    "E_0": 1.0,
    "P_d0": 1.0,
    "P_a0": 1.0,
    "P_x0": 1.0,
}

SHOCKABLE = ["a", "L", "W", "E", "P_x", "N_a", "N_d"]


def _new_state(T):
    return {v: np.full(T + 1, np.nan, dtype=float) for v in OUTPUTS}


def _initial_state(par):
    T = int(par["T"])
    s = _new_state(T)

    a = float(par["a_0"])
    L = float(par["L_0"])
    P0 = float(par["P_0"])
    W0 = float(par["W_0"])
    E0 = float(par["E_0"])
    Pd0 = float(par["P_d0"])
    Pa0 = float(par["P_a0"])
    Px0 = float(par["P_x0"])

    N0 = (P0 - W0 * L) / a

    s["a"][0] = a
    s["L"][0] = L
    s["N_a"][0] = N0
    s["N_d"][0] = N0
    s["N_x"][0] = N0
    s["P"][0] = P0
    s["W"][0] = W0
    s["E"][0] = E0
    s["P_d"][0] = Pd0
    s["P_a"][0] = Pa0
    s["P_x"][0] = Px0

    s["w"][0] = 0.0
    s["p"][0] = 0.0
    s["b"][0] = W0 / P0
    s["WE"][0] = W0 / E0

    s["N"][0] = N0
    s["M"][0] = N0
    s["M_x"][0] = N0
    s["M_a"][0] = N0
    s["M_d"][0] = N0
    s["r"][0] = N0 - 1.0

    s["P_ap"][0] = Pa0 / P0
    s["P_dp"][0] = Pd0 / P0
    s["P_xp"][0] = E0 * Px0 / P0

    return s


def _endogenous_update(s, t, par, preserve_pa=False):
    prev = t - 1
    a_prev = s["a"][prev]
    P_prev = s["P"][prev]

    s["P_d"][t] = (
        a_prev * P_prev * s["N_d"][t]
        + s["L"][t] * s["W"][t]
    )

    if not preserve_pa:
        s["P_a"][t] = (
            a_prev * P_prev * s["N_a"][t]
            + s["L"][t] * s["W"][t]
        )

    s["P"][t] = (
        par["f_d"] * s["P_d"][t]
        + par["f_a"] * s["P_a"][t]
        + par["f_x"] * s["E"][t] * s["P_x"][t]
    )

    s["p"][t] = s["P"][t] / P_prev - 1.0

    s["N_x"][t] = (
        s["E"][t] * s["P_x"][t]
        - s["L"][t] * s["W"][t]
    ) / (a_prev * P_prev)

    s["b"][t] = s["W"][t] / s["P"][t]
    s["WE"][t] = s["W"][t] / s["E"][t]

    s["N"][t] = (
        par["f_d"] * s["N_d"][t]
        + par["f_a"] * s["N_a"][t]
        + par["f_x"] * s["N_x"][t]
    )

    s["M"][t] = (
        s["P"][t] - s["L"][t] * s["W"][t]
    ) / (a_prev * s["P"][t])
    s["r"][t] = s["M"][t] - 1.0

    s["M_x"][t] = (
        s["E"][t] * s["P_x"][t] - s["L"][t] * s["W"][t]
    ) / (a_prev * s["P"][t])

    s["M_a"][t] = (
        s["P_a"][t] - s["L"][t] * s["W"][t]
    ) / (a_prev * s["P"][t])

    s["M_d"][t] = (
        s["P_d"][t] - s["L"][t] * s["W"][t]
    ) / (a_prev * s["P"][t])

    s["P_ap"][t] = s["P_a"][t] / s["P"][t]
    s["P_dp"][t] = s["P_d"][t] / s["P"][t]
    s["P_xp"][t] = s["E"][t] * s["P_x"][t] / s["P"][t]


def _to_frame(s, T):
    return pd.DataFrame({
        "period": np.arange(T + 1),
        **{v: s[v][:T + 1] for v in OUTPUTS}
    })


def _apply_WE_indexation(s, t, wage_k, wage_alpha, exchange_k, exchange_alpha):
    prev = t - 1
    inflation = s["p"][prev]

    s["w"][t] = float(wage_k) + float(wage_alpha) * inflation
    s["W"][t] = (1.0 + s["w"][t]) * s["W"][prev]

    e_growth = float(exchange_k) + float(exchange_alpha) * inflation
    s["E"][t] = (1.0 + e_growth) * s["E"][prev]


def _run_indexed(par, wage_k, wage_alpha, exchange_k, exchange_alpha,
                 mode="WE", pa_k=0.0, pa_alpha=1.0, L_growth=0.0):
    T = int(par["T"])
    s = _initial_state(par)

    for t in range(1, T + 1):
        prev = t - 1

        s["a"][t] = float(par["a_0"])
        s["L"][t] = (
            (1.0 - float(L_growth)) * s["L"][prev]
            if mode == "L_growth"
            else float(par["L_0"])
        )
        s["N_d"][t] = s["N_d"][0]
        s["P_x"][t] = s["P_x"][0]

        _apply_WE_indexation(
            s, t, wage_k, wage_alpha, exchange_k, exchange_alpha
        )

        if mode == "Pa_indexation":
            inflation = s["p"][prev]
            pa_growth = float(pa_k) + float(pa_alpha) * inflation
            s["P_a"][t] = (1.0 + pa_growth) * s["P_a"][prev]

            s["N_a"][t] = (
                s["P_a"][t] - s["L"][t] * s["W"][t]
            ) / (s["a"][prev] * s["P"][prev])
        else:
            s["N_a"][t] = s["N_a"][0]

        _endogenous_update(s, t, par)

        # Re-impose the indexed P_a after the endogenous identity is used.
        # Its associated N_a is the endogenous residual.
        if mode == "Pa_indexation":
            s["N_a"][t] = (
                s["P_a"][t] - s["L"][t] * s["W"][t]
            ) / (s["a"][prev] * s["P"][prev])

    return _to_frame(s, T)


def baseline_model(par, wage_k=0.0, wage_alpha=0.0,
                   exchange_k=0.0, exchange_alpha=0.0):
    """Baseline with the supplied W/E rules (defaults imply no indexation)."""
    return _run_indexed(
        par, wage_k, wage_alpha, exchange_k, exchange_alpha, mode="WE"
    )


def simulate_custom_indexation(par, wage_k, F_w, F_p,
                               exchange_k, exchange_alpha):
    """
    W/E indexation.

    Wage pass-through:
        alpha_W = F_W / (F_W + F_P)
    Exchange-rate pass-through:
        alpha_E is entered directly in the dashboard.
    """
    F_w = float(F_w)
    F_p = float(F_p)
    if F_w < 0 or F_p < 0 or F_w + F_p <= 0:
        raise ValueError("F_W and F_P must be non-negative and not both zero.")

    alpha_w = F_w / (F_w + F_p)

    return _run_indexed(
        par,
        wage_k=wage_k,
        wage_alpha=alpha_w,
        exchange_k=exchange_k,
        exchange_alpha=exchange_alpha,
        mode="WE",
    )


def simulate_pa_indexation(par, wage_k, F_w, F_p,
                           exchange_k, exchange_alpha,
                           pa_k, pa_alpha):
    F_w = float(F_w)
    F_p = float(F_p)
    if F_w < 0 or F_p < 0 or F_w + F_p <= 0:
        raise ValueError("F_W and F_P must be non-negative and not both zero.")

    alpha_w = F_w / (F_w + F_p)

    return _run_indexed(
        par,
        wage_k=wage_k,
        wage_alpha=alpha_w,
        exchange_k=exchange_k,
        exchange_alpha=exchange_alpha,
        mode="Pa_indexation",
        pa_k=pa_k,
        pa_alpha=pa_alpha,
    )


def simulate_L_growth(par, wage_k, F_w, F_p,
                      exchange_k, exchange_alpha, L_growth):
    F_w = float(F_w)
    F_p = float(F_p)
    if F_w < 0 or F_p < 0 or F_w + F_p <= 0:
        raise ValueError("F_W and F_P must be non-negative and not both zero.")

    alpha_w = F_w / (F_w + F_p)

    return _run_indexed(
        par,
        wage_k=wage_k,
        wage_alpha=alpha_w,
        exchange_k=exchange_k,
        exchange_alpha=exchange_alpha,
        mode="L_growth",
        L_growth=L_growth,
    )


def simulate_combined(par, wage_k, F_w, F_p, exchange_k, exchange_alpha,
                     activate_shock=False, variable=None, shock_pct=0.0,
                     shock_period=1, persistence="Permanente",
                     activate_pa=False, pa_k=0.0, pa_alpha=0.0,
                     activate_L=False, L_growth=0.0):
    """Run W/E indexation together with any combination of optional effects.

    The three optional mechanisms are independent and can operate together:
    structural shock, P^m indexation, and technical progress (a reduction in l).
    A permanent structural shock is a persistent level shift, not a repeated
    multiplication of the shocked variable.
    """
    T = int(par["T"])
    F_w, F_p = float(F_w), float(F_p)
    if F_w < 0 or F_p < 0 or F_w + F_p <= 0:
        raise ValueError("F_W and F_P must be non-negative and not both zero.")
    alpha_w = F_w / (F_w + F_p)

    if activate_shock:
        if variable not in SHOCKABLE:
            raise ValueError(f"Unknown shock variable: {variable}")
        t0 = int(shock_period)
        if not 1 <= t0 <= T:
            raise ValueError("Shock period must be between 1 and T.")
        factor = 1.0 + float(shock_pct) / 100.0
        permanent = str(persistence).strip().lower().startswith("permanent")
    else:
        t0, factor, permanent = -1, 1.0, False

    if activate_L and float(L_growth) >= 1.0:
        raise ValueError("Technical-progress rate must be below 1.")

    s = _initial_state(par)

    # Keep separate unshocked paths for permanent shocks. This prevents a
    # permanent shock from being compounded at every subsequent period.
    base_L = float(par["L_0"])

    for t in range(1, T + 1):
        prev = t - 1

        s["a"][t] = float(par["a_0"])
        if activate_L:
            base_L = (1.0 - float(L_growth)) * base_L
            s["L"][t] = base_L
        else:
            s["L"][t] = float(par["L_0"])

        s["N_d"][t] = s["N_d"][0]
        s["N_a"][t] = s["N_a"][0]
        s["P_x"][t] = s["P_x"][0]

        # W and E are always configurable and use the model's inflation rate p.
        _apply_WE_indexation(
            s, t, wage_k, alpha_w, exchange_k, exchange_alpha
        )

        # P^m indexation is applied independently of W/E indexation.
        if activate_pa:
            inflation = s["p"][prev]
            pa_growth = float(pa_k) + float(pa_alpha) * inflation
            s["P_a"][t] = (1.0 + pa_growth) * s["P_a"][prev]
            s["N_a"][t] = (
                s["P_a"][t] - s["L"][t] * s["W"][t]
            ) / (s["a"][prev] * s["P"][prev])

        # Structural shock is a level disturbance to the configured path.
        # For a permanent shock the same level factor is maintained from t0,
        # rather than multiplying an already-shocked state repeatedly.
        shocked = activate_shock and (t == t0 or (permanent and t > t0))
        if shocked:
            if variable == "a":
                s["a"][t] *= factor
            elif variable == "L":
                s["L"][t] *= factor
            elif variable == "W":
                s["W"][t] *= factor
                s["w"][t] = s["W"][t] / s["W"][prev] - 1.0
            elif variable == "E":
                s["E"][t] *= factor
            elif variable == "P_x":
                s["P_x"][t] *= factor
            elif variable == "N_a":
                s["N_a"][t] *= factor
            elif variable == "N_d":
                s["N_d"][t] *= factor

            # If N_a is shocked while P^m is indexed, the structural shock is
            # applied to the markup and therefore determines P^m in that period.
            if variable == "N_a":
                s["P_a"][t] = (
                    s["a"][prev] * s["P"][prev] * s["N_a"][t]
                    + s["L"][t] * s["W"][t]
                )

        _endogenous_update(s, t, par, preserve_pa=activate_pa and variable != "N_a")

    return _to_frame(s, T)

def simulate_parameter_shock(par, variable, shock_pct, shock_period,
                             persistence="Permanente",
                             wage_k=0.0, F_w=1.0, F_p=1.0,
                             exchange_k=0.0, exchange_alpha=0.0):
    """
    Shock mode with W/E indexation active.

    Permanent shocks are relative to the original baseline level and therefore
    do not compound period after period.
    """
    if variable not in SHOCKABLE:
        raise ValueError(f"Unknown shock variable: {variable}")

    T = int(par["T"])
    t0 = int(shock_period)
    if t0 < 1 or t0 > T:
        raise ValueError("Shock period must be between 1 and T.")

    F_w = float(F_w)
    F_p = float(F_p)
    if F_w < 0 or F_p < 0 or F_w + F_p <= 0:
        raise ValueError("F_W and F_P must be non-negative and not both zero.")

    alpha_w = F_w / (F_w + F_p)

    s = _initial_state(par)
    factor = 1.0 + float(shock_pct) / 100.0
    permanent = str(persistence).strip().lower().startswith("permanent")

    baseline = {
        "a": float(par["a_0"]),
        "L": float(par["L_0"]),
        "W": float(s["W"][0]),
        "E": float(s["E"][0]),
        "P_x": float(s["P_x"][0]),
        "N_a": float(s["N_a"][0]),
        "N_d": float(s["N_d"][0]),
    }

    for t in range(1, T + 1):
        prev = t - 1

        s["a"][t] = float(par["a_0"])
        s["L"][t] = float(par["L_0"])
        s["N_a"][t] = s["N_a"][0]
        s["N_d"][t] = s["N_d"][0]
        s["P_x"][t] = s["P_x"][0]

        _apply_WE_indexation(
            s, t, wage_k, alpha_w, exchange_k, exchange_alpha
        )

        shocked = (t == t0) or (permanent and t > t0)
        if shocked:
            s[variable][t] = baseline[variable] * factor

            if variable == "W":
                s["w"][t] = s["W"][t] / s["W"][prev] - 1.0

        _endogenous_update(s, t, par)

    return _to_frame(s, T)

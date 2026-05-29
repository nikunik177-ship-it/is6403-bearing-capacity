"""
app/core/is_6403_engine.py
==========================
Mathematical core for IS 6403:1981 Soil Bearing Capacity estimation.

Net Ultimate Bearing Capacity (IS 6403:1981 Clause 5.1):
    q_nu = c·Nc·sc·dc·ic
         + q·(Nq − 1)·sq·dq·iq·W'_q
         + 0.5·γ·B·Nγ·sγ·dγ·iγ·W'_γ

Engineering Units (SI):
    Cohesion / capacity : kPa
    Unit weight         : kN/m³
    Dimensions          : m
    Angles              : degrees
"""

from __future__ import annotations

import math
from typing import Any

# pyrefly: ignore [missing-import]
import numpy as np

# ---------------------------------------------------------------------------
# IS 6403:1981 Table 1 — Bearing Capacity Factors (φ = 0° to 45°)
# ---------------------------------------------------------------------------
# Source: IS 6403:1981 Table 1 as reproduced in Arora (9th ed.) and
# Punmia (16th ed.).
# Structure: { phi_degrees: (Nc, Nq, N_gamma) }
# ---------------------------------------------------------------------------

_TABLE_1: dict[int, tuple[float, float, float]] = {
    0:  (5.14,   1.00,   0.00),
    1:  (5.38,   1.09,   0.07),
    2:  (5.63,   1.20,   0.15),
    3:  (5.90,   1.31,   0.24),
    4:  (6.19,   1.43,   0.34),
    5:  (6.49,   1.57,   0.45),
    6:  (6.81,   1.72,   0.57),
    7:  (7.16,   1.88,   0.71),
    8:  (7.53,   2.06,   0.86),
    9:  (7.92,   2.25,   1.03),
    10: (8.35,   2.47,   1.22),
    11: (8.80,   2.71,   1.44),
    12: (9.28,   2.97,   1.69),
    13: (9.81,   3.26,   1.97),
    14: (10.37,  3.59,   2.29),
    15: (10.98,  3.94,   2.65),
    16: (11.63,  4.34,   3.06),
    17: (12.34,  4.77,   3.53),
    18: (13.10,  5.26,   4.07),
    19: (13.93,  5.80,   4.68),
    20: (14.83,  6.40,   5.39),
    21: (15.82,  7.07,   6.20),
    22: (16.88,  7.82,   7.13),
    23: (18.05,  8.66,   8.20),
    24: (19.32,  9.60,   9.44),
    25: (20.72, 10.66,  10.88),
    26: (22.25, 11.85,  12.54),
    27: (23.94, 13.20,  14.47),
    28: (25.80, 14.72,  16.72),
    29: (27.86, 16.44,  19.34),
    30: (30.14, 18.40,  22.40),
    31: (32.67, 20.63,  25.99),
    32: (35.49, 23.18,  30.22),
    33: (38.64, 26.09,  35.19),
    34: (42.16, 29.44,  41.06),
    35: (46.12, 33.30,  48.03),
    36: (50.59, 37.75,  56.31),
    37: (55.63, 42.92,  66.19),
    38: (61.35, 48.93,  78.03),
    39: (67.87, 55.96,  92.25),
    40: (75.31, 64.20, 109.41),
    41: (83.86, 73.90, 130.22),
    42: (93.71, 85.38, 155.55),
    43: (105.11, 99.02, 186.54),
    44: (118.37, 115.31, 224.64),
    45: (133.88, 134.88, 271.76),
}

# Pre-build NumPy arrays for fast vectorised interpolation
_PHI_ARR: np.ndarray = np.array(list(_TABLE_1.keys()), dtype=float)
_NC_ARR: np.ndarray  = np.array([v[0] for v in _TABLE_1.values()], dtype=float)
_NQ_ARR: np.ndarray  = np.array([v[1] for v in _TABLE_1.values()], dtype=float)
_NG_ARR: np.ndarray  = np.array([v[2] for v in _TABLE_1.values()], dtype=float)


# ===========================================================================
# Public helper functions
# ===========================================================================

def interpolate_bc_factors(phi_deg: float) -> tuple[float, float, float]:
    """
    Linearly interpolate IS 6403 Table 1 Bearing Capacity Factors.

    Parameters
    ----------
    phi_deg : float
        Angle of internal friction φ  [degrees, 0–45].

    Returns
    -------
    (Nc, Nq, N_gamma) : tuple[float, float, float]
        Dimensionless bearing capacity factors.

    Raises
    ------
    ValueError
        If phi_deg is outside [0, 45]°.
    """
    if not (0.0 <= phi_deg <= 45.0):
        raise ValueError(
            f"Friction angle must be in [0, 45]°; received {phi_deg}°."
        )
    nc = float(np.interp(phi_deg, _PHI_ARR, _NC_ARR))
    nq = float(np.interp(phi_deg, _PHI_ARR, _NQ_ARR))
    ng = float(np.interp(phi_deg, _PHI_ARR, _NG_ARR))
    return nc, nq, ng


def local_shear_params(
    cohesion: float,
    phi_deg: float,
) -> tuple[float, float]:
    """
    Compute modified shear parameters for Local Shear Failure per
    IS 6403:1981 Clause 5.1.1.

    Parameters
    ----------
    cohesion : float
        Undrained cohesion c  [kPa].
    phi_deg : float
        Friction angle φ  [degrees].

    Returns
    -------
    c_prime : float
        Modified cohesion  c' = (2/3)·c  [kPa].
    phi_prime : float
        Modified friction angle  φ' = arctan(0.67·tan φ)  [degrees].
    """
    c_prime = (2.0 / 3.0) * cohesion
    phi_prime = math.degrees(
        math.atan(0.67 * math.tan(math.radians(phi_deg)))
    )
    return c_prime, phi_prime


def shape_factors(
    shape: str,
    b: float,
    length: float,
) -> tuple[float, float, float]:
    """
    IS 6403:1981 Table 2 shape correction factors.

    Parameters
    ----------
    shape : str
        ``"strip"``, ``"rectangular"``, ``"square"``, or ``"circular"``.
    b : float
        Footing width B  [m].
    length : float
        Footing length L  [m].  Ignored for non-rectangular shapes.

    Returns
    -------
    (sc, sq, s_gamma) : tuple[float, float, float]
        Dimensionless shape factors.
    """
    shape = shape.lower()
    if shape == "strip":
        return 1.0, 1.0, 1.0
    if shape == "square":
        return 1.3, 1.2, 0.8
    if shape == "circular":
        return 1.3, 1.2, 0.6
    # rectangular
    bl = b / length if length > 0 else 0.0
    return 1.0 + 0.2 * bl, 1.0 + 0.2 * bl, 1.0 - 0.4 * bl


def depth_factors(
    phi_deg: float,
    df: float,
    b: float,
) -> tuple[float, float, float]:
    """
    IS 6403:1981 Clause 5.1.2 depth correction factors.

    Parameters
    ----------
    phi_deg : float
        Effective friction angle φ  [degrees].
    df : float
        Founding depth Df  [m].
    b : float
        Footing width B  [m].

    Returns
    -------
    (dc, dq, d_gamma) : tuple[float, float, float]
        Dimensionless depth factors.

    Notes
    -----
    N_phi = tan²(45 + φ/2)
    dc    = 1 + 0.2·(Df/B)·√N_phi
    dq = dγ = 1 + 0.1·(Df/B)·√N_phi   for φ > 10°
    dq = dγ = 1.0                       for φ ≤ 10°
    """
    n_phi = math.tan(math.radians(45.0 + phi_deg / 2.0)) ** 2
    sqrt_np = math.sqrt(n_phi)
    ratio = df / b if b > 0 else 0.0

    dc = 1.0 + 0.2 * ratio * sqrt_np
    if phi_deg > 10.0:
        dq = dg = 1.0 + 0.1 * ratio * sqrt_np
    else:
        dq = dg = 1.0
    return dc, dq, dg


def inclination_factors(
    alpha_deg: float,
    phi_deg: float,
) -> tuple[float, float, float]:
    """
    IS 6403:1981 Clause 5.1.2 inclination correction factors.

    Parameters
    ----------
    alpha_deg : float
        Angle of resultant load from vertical α  [degrees].
    phi_deg : float
        Effective friction angle φ  [degrees].

    Returns
    -------
    (ic, iq, i_gamma) : tuple[float, float, float]
        Dimensionless inclination factors.

    Notes
    -----
    ic = iq = (1 − α/90)²
    iγ = (max(0, 1 − α/φ))²   for φ > 0°
    iγ = 1.0                    for φ = 0°
    """
    if alpha_deg == 0.0:
        return 1.0, 1.0, 1.0
    ic = iq = max(0.0, 1.0 - alpha_deg / 90.0) ** 2
    if phi_deg > 0.0:
        i_gamma = max(0.0, 1.0 - alpha_deg / phi_deg) ** 2
    else:
        i_gamma = 1.0
    return ic, iq, i_gamma


def water_table_factors(
    dw: float,
    df: float,
    b: float,
) -> tuple[float, float]:
    """
    IS 6403:1981 Clause 5.1.3 water table correction factors W'.

    Parameters
    ----------
    dw : float
        Depth of water table below GL  [m].
    df : float
        Founding depth Df  [m].
    b : float
        Footing width B  [m].

    Returns
    -------
    (W_prime_q, W_prime_gamma) : tuple[float, float]
        Reduction factors for surcharge and unit weight terms (0.5–1.0).
    """
    if dw <= 0.0:
        return 0.5, 0.5
    if dw <= df:
        wq = 0.5 + 0.5 * (dw / df) if df > 0 else 0.5
        return wq, 0.5
    if dw <= df + b:
        wg = 0.5 + 0.5 * ((dw - df) / b)
        return 1.0, wg
    return 1.0, 1.0


def _effective_overburden(
    gamma: float,
    gamma_sat: float,
    dw: float,
    df: float,
) -> float:
    """
    Compute effective overburden pressure q = γ_eff × Df  [kPa].

    Uses a depth-weighted average unit weight when the water table
    lies within the overburden column.

    Parameters
    ----------
    gamma : float
        Bulk unit weight  [kN/m³].
    gamma_sat : float
        Saturated unit weight  [kN/m³].
    dw : float
        Depth of water table below GL  [m].
    df : float
        Founding depth Df  [m].
    """
    gamma_sub = gamma_sat - 9.81       # submerged unit weight [kN/m³]
    if dw <= 0.0:
        return gamma_sub * df
    if dw < df:
        gamma_eff = (gamma * dw + gamma_sub * (df - dw)) / df
        return gamma_eff * df
    return gamma * df


# ===========================================================================
# Main calculation engine
# ===========================================================================

def calculate_bearing_capacity(inputs: dict[str, Any]) -> dict[str, Any]:
    """
    Compute IS 6403:1981 net ultimate, net safe, and gross safe bearing
    capacities from a dictionary of site parameters.

    Parameters
    ----------
    inputs : dict
        Keys (all required):
            cohesion        : float  [kPa]
            friction_angle  : float  [degrees]
            unit_weight     : float  [kN/m³]
            sat_unit_weight : float  [kN/m³]
            water_table_depth : float  [m]
            footing_shape   : str   (strip | rectangular | square | circular)
            width           : float  [m]
            length          : float  [m]   (ignored for non-rectangular)
            depth           : float  [m]   (Df)
            load_inclination: float  [degrees]
            failure_mode    : str   (auto | general | local)
            fos             : float  Factor of Safety

    Returns
    -------
    dict
        All intermediate factors and final capacities  [kPa].
    """
    # ------------------------------------------------------------------ #
    # Unpack inputs
    # ------------------------------------------------------------------ #
    c          = float(inputs["cohesion"])
    phi        = float(inputs["friction_angle"])
    gamma      = float(inputs["unit_weight"])
    gamma_sat  = float(inputs.get("sat_unit_weight", gamma + 2.0))
    dw         = float(inputs.get("water_table_depth", 1e6))
    shape      = str(inputs.get("footing_shape", "strip")).lower()
    b          = float(inputs["width"])
    length     = float(inputs.get("length", b))
    df         = float(inputs["depth"])
    alpha      = float(inputs.get("load_inclination", 0.0))
    mode       = str(inputs.get("failure_mode", "auto")).lower()
    fos        = float(inputs.get("fos", 2.5))

    # ------------------------------------------------------------------ #
    # Step 1 — Resolve failure mode and effective parameters
    # ------------------------------------------------------------------ #
    if mode == "auto":
        mode = "local" if phi < 28.0 else "general"

    if mode == "local":
        c_eff, phi_eff = local_shear_params(c, phi)
    else:
        c_eff, phi_eff = c, phi

    # ------------------------------------------------------------------ #
    # Step 2 — Bearing Capacity Factors (IS 6403 Table 1)
    # ------------------------------------------------------------------ #
    nc, nq, ng = interpolate_bc_factors(phi_eff)

    # ------------------------------------------------------------------ #
    # Step 3 — Shape Factors (IS 6403 Table 2)
    # ------------------------------------------------------------------ #
    sc, sq, sg = shape_factors(shape, b, length)

    # ------------------------------------------------------------------ #
    # Step 4 — Depth Factors (IS 6403 Cl. 5.1.2)
    # ------------------------------------------------------------------ #
    dc, dq, dg = depth_factors(phi_eff, df, b)

    # ------------------------------------------------------------------ #
    # Step 5 — Inclination Factors (IS 6403 Cl. 5.1.2)
    # ------------------------------------------------------------------ #
    ic, iq, ig = inclination_factors(alpha, phi_eff)

    # ------------------------------------------------------------------ #
    # Step 6 — Water Table Correction (IS 6403 Cl. 5.1.3)
    # ------------------------------------------------------------------ #
    wq, wg = water_table_factors(dw, df, b)

    # ------------------------------------------------------------------ #
    # Step 7 — Effective overburden at founding level
    # ------------------------------------------------------------------ #
    q_ovb = _effective_overburden(gamma, gamma_sat, dw, df)

    # ------------------------------------------------------------------ #
    # Step 8 — Net Ultimate Bearing Capacity
    # ------------------------------------------------------------------ #
    term_c     = c_eff * nc * sc * dc * ic
    term_q     = q_ovb * (nq - 1.0) * sq * dq * iq * wq
    term_gamma = 0.5 * gamma * b * ng * sg * dg * ig * wg
    q_nu       = term_c + term_q + term_gamma

    # ------------------------------------------------------------------ #
    # Step 9 — Safe Capacities (IS 1904:1986)
    # ------------------------------------------------------------------ #
    q_f  = q_nu + q_ovb           # gross ultimate
    q_ns = q_nu / fos             # net safe
    q_s  = q_ns + q_ovb           # gross safe

    return {
        # --- inputs echoed ---
        "failure_mode_used": mode,
        "c_eff":    round(c_eff,    4),
        "phi_eff":  round(phi_eff,  4),
        # --- BC factors ---
        "N_c":      round(nc, 4),
        "N_q":      round(nq, 4),
        "N_gamma":  round(ng, 4),
        # --- shape factors ---
        "s_c":      round(sc, 4),
        "s_q":      round(sq, 4),
        "s_gamma":  round(sg, 4),
        # --- depth factors ---
        "d_c":      round(dc, 4),
        "d_q":      round(dq, 4),
        "d_gamma":  round(dg, 4),
        # --- inclination factors ---
        "i_c":      round(ic, 4),
        "i_q":      round(iq, 4),
        "i_gamma":  round(ig, 4),
        # --- water table ---
        "W_prime_q":     round(wq, 4),
        "W_prime_gamma": round(wg, 4),
        # --- capacities ---
        "q_overburden": round(q_ovb, 3),
        "term_c":       round(term_c,     3),
        "term_q":       round(term_q,     3),
        "term_gamma":   round(term_gamma, 3),
        "q_f":   round(q_f,  3),
        "q_nu":  round(q_nu, 3),
        "q_ns":  round(q_ns, 3),
        "q_s":   round(q_s,  3),
        "fos":   fos,
    }

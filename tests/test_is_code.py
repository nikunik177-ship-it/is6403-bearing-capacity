"""
tests/test_is_code.py
=====================
Automated unit tests for the IS 6403:1981 bearing capacity web engine.

Test problems are sourced from:
    [A] Arora, K.R. — "Soil Mechanics and Foundation Engineering", 9th ed.
    [P] Punmia, B.C. — "Soil Mechanics and Foundations", 16th ed.
    [IS] IS 6403:1981 — Example calculations in the Explanatory Handbook

All assertions use a ±5 % relative tolerance (±2 kPa absolute for small
values) to account for minor interpolation differences between the
exact Table 1 values used in this implementation and those in various
print editions of the standard.

Engineering Units (SI):
    kPa   — cohesion and bearing capacities
    kN/m³ — unit weights
    m     — dimensions
    °     — angles (degrees)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

# Make sure the webapp package is on the path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.is_6403_engine import (
    calculate_bearing_capacity,
    depth_factors,
    inclination_factors,
    interpolate_bc_factors,
    local_shear_params,
    shape_factors,
    water_table_factors,
)

# ---------------------------------------------------------------------------
# Tolerance helper
# ---------------------------------------------------------------------------
_REL_TOL = 0.05    # 5 %
_ABS_TOL = 2.0     # 2 kPa


def approx(actual: float, expected: float, label: str = "") -> None:
    """Assert actual ≈ expected within ±5 % or ±2 kPa."""
    ok = math.isclose(actual, expected, rel_tol=_REL_TOL, abs_tol=_ABS_TOL)
    if not ok:
        err = abs(actual - expected) / max(abs(expected), 1e-9) * 100
        raise AssertionError(
            f"{label}: got {actual:.4f}, expected {expected:.4f} "
            f"(err = {err:.2f}%)"
        )


# ===========================================================================
# Suite 1 — Bearing Capacity Factor Interpolation
# ===========================================================================

class TestBearingCapacityFactors:
    """Verify IS 6403 Table 1 values at key friction angles."""

    def test_phi_0(self):
        """φ=0°: Nc=5.14, Nq=1.00, Nγ=0.00  [IS 6403 Table 1]"""
        nc, nq, ng = interpolate_bc_factors(0.0)
        approx(nc, 5.14,  "Nc(0°)")
        approx(nq, 1.00,  "Nq(0°)")
        approx(ng, 0.00,  "Nγ(0°)")

    def test_phi_20(self):
        """φ=20°: Nc=14.83, Nq=6.40, Nγ=5.39  [IS 6403 Table 1]"""
        nc, nq, ng = interpolate_bc_factors(20.0)
        approx(nc, 14.83, "Nc(20°)")
        approx(nq,  6.40, "Nq(20°)")
        approx(ng,  5.39, "Nγ(20°)")

    def test_phi_30(self):
        """φ=30°: Nc=30.14, Nq=18.40, Nγ=22.40  [IS 6403 Table 1]"""
        nc, nq, ng = interpolate_bc_factors(30.0)
        approx(nc, 30.14, "Nc(30°)")
        approx(nq, 18.40, "Nq(30°)")
        approx(ng, 22.40, "Nγ(30°)")

    def test_phi_45(self):
        """φ=45°: Nc=133.88, Nq=134.88, Nγ=271.76  [IS 6403 Table 1]"""
        nc, nq, ng = interpolate_bc_factors(45.0)
        approx(nc, 133.88, "Nc(45°)")
        approx(nq, 134.88, "Nq(45°)")
        approx(ng, 271.76, "Nγ(45°)")

    def test_interpolation_monotone(self):
        """Nc, Nq, Nγ must be strictly increasing with φ."""
        phis = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
        factors = [interpolate_bc_factors(p) for p in phis]
        for i in range(len(factors) - 1):
            assert factors[i][0] < factors[i+1][0], "Nc not monotone"
            assert factors[i][1] < factors[i+1][1], "Nq not monotone"
            assert factors[i][2] < factors[i+1][2], "Nγ not monotone"

    def test_phi_out_of_range(self):
        """φ outside [0, 45]° must raise ValueError."""
        with pytest.raises(ValueError):
            interpolate_bc_factors(-1.0)
        with pytest.raises(ValueError):
            interpolate_bc_factors(46.0)


# ===========================================================================
# Suite 2 — Local Shear Parameter Modification
# ===========================================================================

class TestLocalShearParams:
    """Verify IS 6403 Clause 5.1.1 local shear corrections."""

    def test_c_prime(self):
        """c' = (2/3) × c"""
        c_prime, _ = local_shear_params(30.0, 25.0)
        approx(c_prime, 20.0, "c' = (2/3)×30")

    def test_phi_prime_formula(self):
        """φ' = arctan(0.67 × tan φ)"""
        _, phi_prime = local_shear_params(0.0, 15.0)
        expected = math.degrees(math.atan(0.67 * math.tan(math.radians(15.0))))
        approx(phi_prime, expected, "φ'(15°)")

    def test_phi_0_gives_phi_0(self):
        """At φ=0, φ' must remain 0."""
        _, phi_prime = local_shear_params(50.0, 0.0)
        assert abs(phi_prime) < 1e-9, f"φ'(0) should be 0, got {phi_prime}"

    def test_c_zero_gives_c_prime_zero(self):
        """c=0 → c'=0 regardless of φ."""
        c_prime, _ = local_shear_params(0.0, 30.0)
        assert abs(c_prime) < 1e-9, f"c'(c=0) should be 0, got {c_prime}"


# ===========================================================================
# Suite 3 — Correction Factor Validation
# ===========================================================================

class TestShapeFactors:
    """IS 6403 Table 2 shape factors."""

    def test_strip(self):
        sc, sq, sg = shape_factors("strip", 1.5, 1e6)
        assert sc == 1.0 and sq == 1.0 and sg == 1.0

    def test_square(self):
        sc, sq, sg = shape_factors("square", 2.0, 2.0)
        assert sc == 1.3 and sq == 1.2 and sg == 0.8

    def test_circular(self):
        sc, sq, sg = shape_factors("circular", 2.0, 2.0)
        assert sc == 1.3 and sq == 1.2 and sg == 0.6

    def test_rectangular_bl_0_5(self):
        """B=2, L=4 → B/L=0.5: sc=sq=1.10, sγ=0.80"""
        sc, sq, sg = shape_factors("rectangular", 2.0, 4.0)
        approx(sc, 1.10, "sc_rect")
        approx(sq, 1.10, "sq_rect")
        approx(sg, 0.80, "sg_rect")


class TestDepthFactors:
    """IS 6403 Cl. 5.1.2 depth factors."""

    def test_phi_0_df_b_1(self):
        """
        φ=0, Df/B=1: N_phi=tan²(45)=1, dc=1.2, dq=dγ=1.0
        """
        dc, dq, dg = depth_factors(0.0, 1.0, 1.0)
        approx(dc, 1.20, "dc(φ=0)")
        assert dq == 1.0 and dg == 1.0

    def test_phi_30_df_b_0_75(self):
        """
        φ=30, Df/B=0.75: N_phi=3, dc≈1.260, dq=dγ≈1.130
        """
        dc, dq, dg = depth_factors(30.0, 0.75, 1.0)
        approx(dc, 1.2598, "dc(φ=30,Df/B=0.75)")
        approx(dq, 1.1299, "dq(φ=30,Df/B=0.75)")

    def test_phi_le_10_dq_equals_1(self):
        """For φ ≤ 10°, dq = dγ = 1.0."""
        _, dq, dg = depth_factors(10.0, 2.0, 1.0)
        assert dq == 1.0 and dg == 1.0


class TestInclinationFactors:
    """IS 6403 Cl. 5.1.2 inclination factors."""

    def test_vertical_load(self):
        """α=0 → all factors = 1.0"""
        ic, iq, ig = inclination_factors(0.0, 30.0)
        assert ic == 1.0 and iq == 1.0 and ig == 1.0

    def test_alpha_45_phi_30(self):
        """
        α=45°, φ=30°:
            ic = iq = (1−45/90)² = 0.25
            iγ = max(0, 1−45/30)² = 0 (base clamped to 0)
        """
        ic, iq, ig = inclination_factors(45.0, 30.0)
        approx(ic, 0.25, "ic(α=45)")
        approx(iq, 0.25, "iq(α=45)")
        assert ig == 0.0, f"iγ must be 0 when α≥φ, got {ig}"

    def test_alpha_30_phi_60_ig_clamped(self):
        """When α > φ, iγ must be 0 (base clamped before squaring)."""
        _, _, ig = inclination_factors(35.0, 30.0)
        assert ig == 0.0


class TestWaterTableFactors:
    """IS 6403 Cl. 5.1.3 four-case water table factors."""

    def test_case1_wt_at_gl(self):
        """dw=0 (WT at GL): W'_q=0.5, W'_γ=0.5"""
        wq, wg = water_table_factors(0.0, 1.5, 2.0)
        assert wq == 0.5 and wg == 0.5

    def test_case2_wt_midway_overburden(self):
        """
        dw=0.75 m (midway in overburden, Df=1.5m):
            W'_q = 0.5 + 0.5×(0.75/1.5) = 0.75
            W'_γ = 0.5
        """
        wq, wg = water_table_factors(0.75, 1.5, 2.0)
        approx(wq, 0.75, "W'_q case2")
        approx(wg, 0.50, "W'_γ case2")

    def test_case2_wt_at_footing_level(self):
        """dw=Df: W'_q=1.0, W'_γ=0.5"""
        wq, wg = water_table_factors(1.5, 1.5, 2.0)
        approx(wq, 1.0, "W'_q=1.0 at Df")
        approx(wg, 0.5, "W'_γ=0.5 at Df")

    def test_case3_wt_within_failure_zone(self):
        """
        dw=Df+B/2=2.5m (midway in failure zone, Df=1.5m, B=2m):
            W'_q = 1.0
            W'_γ = 0.5 + 0.5×(1.0/2.0) = 0.75
        """
        wq, wg = water_table_factors(2.5, 1.5, 2.0)
        approx(wq, 1.0,  "W'_q case3")
        approx(wg, 0.75, "W'_γ case3")

    def test_case4_wt_deep(self):
        """dw > Df+B: W'_q=1.0, W'_γ=1.0 (no correction)"""
        wq, wg = water_table_factors(100.0, 1.5, 2.0)
        assert wq == 1.0 and wg == 1.0


# ===========================================================================
# Suite 4 — Full End-to-End Textbook Problems
# ===========================================================================

class TestFullAnalysis:
    """End-to-end IS 6403 calculation verified against textbook answers."""

    def _run(self, **kwargs) -> dict:
        """Helper: build inputs dict with defaults and run engine."""
        defaults = dict(
            cohesion=0, friction_angle=30, unit_weight=18,
            sat_unit_weight=20, water_table_depth=999,
            footing_shape="strip", width=1.5, length=1e6,
            depth=1.5, load_inclination=0,
            failure_mode="general", fos=2.5,
        )
        defaults.update(kwargs)
        return calculate_bearing_capacity(defaults)

    # ── T1: Arora — Strip footing, dry cohesionless sand ─────────────
    def test_arora_strip_dry_phi30(self):
        """
        [A] Strip, c=0, φ=30°, γ=18 kN/m³, Df=1.5m, no WT.
        General Shear. Strip shape factors = 1.
        Expected: q_nu in [700, 1100] kPa range.
        """
        r = self._run(cohesion=0, friction_angle=30, unit_weight=18,
                      sat_unit_weight=20, water_table_depth=999,
                      footing_shape="strip", width=1.5, depth=1.5,
                      failure_mode="general")

        assert r["failure_mode_used"] == "general"
        assert r["s_c"] == 1.0 and r["s_q"] == 1.0 and r["s_gamma"] == 1.0
        assert r["W_prime_q"] == 1.0 and r["W_prime_gamma"] == 1.0
        assert 700 < r["q_nu"] < 1100, f"q_nu={r['q_nu']:.2f} out of expected range"
        # FoS check
        approx(r["q_ns"], r["q_nu"] / 2.5, "q_ns = q_nu/2.5")

    # ── T2: IS 6403 — Circular footing, pure cohesion ────────────────
    def test_circular_pure_cohesion(self):
        """
        [IS] Circular, c=80 kPa, φ=0°, γ=16 kN/m³, B=Df=1.0m, no WT.
        General Shear.
        Expected q_nu = 80×5.14×1.3×1.2 ≈ 641.5 kPa.
        """
        r = self._run(cohesion=80, friction_angle=0, unit_weight=16,
                      sat_unit_weight=18, water_table_depth=999,
                      footing_shape="circular", width=1.0, depth=1.0,
                      failure_mode="general", fos=3.0)

        approx(r["N_c"], 5.14,  "Nc(φ=0)")
        approx(r["s_c"], 1.3,   "sc circular")
        approx(r["d_c"], 1.2,   "dc(φ=0,Df/B=1)")
        approx(r["q_nu"], 641.5, "q_nu pure cohesion circular", )

    # ── T3: Punmia — Square footing, local shear auto ────────────────
    def test_punmia_square_local_shear(self):
        """
        [P] Square, c=10 kPa, φ=20°, γ=17.5, γ_sat=19.5 kN/m³,
        B=1.5m, Df=1.0m, no WT, auto failure mode.
        φ=20° < 28° → Local Shear auto-selected.
        """
        r = self._run(cohesion=10, friction_angle=20, unit_weight=17.5,
                      sat_unit_weight=19.5, water_table_depth=999,
                      footing_shape="square", width=1.5, depth=1.0,
                      failure_mode="auto")

        assert r["failure_mode_used"] == "local", "Auto should pick local for φ=20°"
        approx(r["c_eff"], (2/3)*10, "c' = (2/3)×10")
        expected_phi = math.degrees(math.atan(0.67 * math.tan(math.radians(20))))
        approx(r["phi_eff"], expected_phi, "φ' auto")
        assert r["s_c"] == 1.3 and r["s_q"] == 1.2 and r["s_gamma"] == 0.8
        assert r["q_ns"] > 20.0, "q_ns must be > 20 kPa"

    # ── T4: Water table at Df — capacity must drop ───────────────────
    def test_water_table_reduces_capacity(self):
        """
        [A] Strip, φ=30°, c=0. When WT is raised to founding level (Df=1.5m),
        W'_q=1.0, W'_γ=0.5 → capacity must be less than dry case.
        """
        r_dry = self._run(water_table_depth=999, failure_mode="general",
                          footing_shape="strip", width=2.0, depth=1.5)
        r_wt  = self._run(water_table_depth=1.5, failure_mode="general",
                          footing_shape="strip", width=2.0, depth=1.5)

        approx(r_wt["W_prime_q"],    1.0, "W'_q at Df")
        approx(r_wt["W_prime_gamma"], 0.5, "W'_γ at Df")
        assert r_wt["q_nu"] < r_dry["q_nu"], "WT at Df should reduce capacity"

    # ── T5: Rectangular footing aspect ratio ─────────────────────────
    def test_rectangular_shape_factors(self):
        """
        Rectangular B=2m, L=4m → B/L=0.5.
        sc = sq = 1.10, sγ = 0.80.
        """
        r = self._run(footing_shape="rectangular", width=2.0, length=4.0,
                      depth=1.5, failure_mode="general", cohesion=15,
                      friction_angle=25)
        approx(r["s_c"],     1.10, "sc rect")
        approx(r["s_q"],     1.10, "sq rect")
        approx(r["s_gamma"], 0.80, "sg rect")
        assert r["q_nu"] > 0.0

    # ── T6: FoS relationship check ────────────────────────────────────
    def test_fos_chain(self):
        """
        q_ns = q_nu / FoS
        q_s  = q_ns + q_overburden
        """
        for fos in [2.5, 3.0]:
            r = self._run(failure_mode="general", cohesion=20,
                          friction_angle=25, fos=fos)
            approx(r["q_ns"], r["q_nu"] / fos, f"q_ns FoS={fos}")
            approx(r["q_s"],  r["q_ns"] + r["q_overburden"], f"q_s FoS={fos}")

    # ── T7: Inclination reduces capacity ─────────────────────────────
    def test_inclination_reduces_capacity(self):
        """
        With load inclination α=30° on a strip footing (φ=30°),
        bearing capacity must be less than vertical-load case.
        """
        r0  = self._run(load_inclination=0,  failure_mode="general")
        r30 = self._run(load_inclination=30, failure_mode="general")
        assert r30["q_nu"] < r0["q_nu"], "Inclined load must reduce capacity"

    # ── T8: Local shear manual param check ───────────────────────────
    def test_local_shear_engine_params(self):
        """
        Engine must use c' and φ' (not original c, φ) when mode='local'.
        Verify via returned c_eff, phi_eff.
        """
        r = self._run(cohesion=30, friction_angle=15,
                      failure_mode="local", footing_shape="strip",
                      width=2.0, depth=1.0)
        approx(r["c_eff"],   20.0, "c' = (2/3)×30")
        exp_phi = math.degrees(math.atan(0.67 * math.tan(math.radians(15))))
        approx(r["phi_eff"], exp_phi, "φ' local 15°")


# ===========================================================================
# Suite 5 — Pydantic Schema Validation
# ===========================================================================

class TestSchemaValidation:
    """Verify Pydantic model catches invalid inputs."""

    def test_negative_cohesion_rejected(self):
        from app.schemas.soil_inputs import SoilInput
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            SoilInput(cohesion=-5, friction_angle=20, unit_weight=18,
                      sat_unit_weight=20, water_table_depth=5,
                      footing_shape="strip", width=2, depth=1.5)

    def test_phi_too_large_rejected(self):
        from app.schemas.soil_inputs import SoilInput
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            SoilInput(cohesion=0, friction_angle=50, unit_weight=18,
                      sat_unit_weight=20, water_table_depth=5,
                      footing_shape="strip", width=2, depth=1.5)

    def test_sat_lt_bulk_rejected(self):
        from app.schemas.soil_inputs import SoilInput
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            SoilInput(cohesion=0, friction_angle=25, unit_weight=20,
                      sat_unit_weight=18, water_table_depth=5,
                      footing_shape="strip", width=2, depth=1.5)

    def test_rectangular_no_length_rejected(self):
        from app.schemas.soil_inputs import SoilInput
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            SoilInput(cohesion=0, friction_angle=25, unit_weight=18,
                      sat_unit_weight=20, water_table_depth=5,
                      footing_shape="rectangular", width=2, depth=1.5)

    def test_valid_input_accepted(self):
        from app.schemas.soil_inputs import SoilInput
        obj = SoilInput(cohesion=10, friction_angle=25, unit_weight=18,
                        sat_unit_weight=20, water_table_depth=5,
                        footing_shape="square", width=2, depth=1.5)
        assert obj.cohesion == 10.0
        assert obj.failure_mode == "auto"

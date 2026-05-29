"""
app/schemas/soil_inputs.py
==========================
Pydantic v2 models for strict validation of all IS 6403:1981 calculation
inputs received from the frontend or API.

Engineering Units (SI):
    Cohesion           : kPa
    Friction angle     : degrees
    Unit weights       : kN/m³
    Dimensions (B, Df) : m
    Angles             : degrees
"""

from __future__ import annotations

from typing import Literal

# pyrefly: ignore [missing-import]
from pydantic import BaseModel, Field, field_validator, model_validator


class SoilInput(BaseModel):
    """
    Validates and normalises all site and footing parameters required
    by the IS 6403:1981 bearing capacity engine.

    All length / pressure / angle constraints reflect the physical
    limits imposed by IS 6403:1981 and IS 1904:1986.
    """

    # ------------------------------------------------------------------ #
    # Soil properties
    # ------------------------------------------------------------------ #
    cohesion: float = Field(
        ge=0.0,
        le=500.0,
        description="Undrained / effective cohesion c  [kPa, 0–500]",
    )
    friction_angle: float = Field(
        ge=0.0,
        le=45.0,
        description="Angle of internal friction φ  [degrees, 0–45]",
    )
    unit_weight: float = Field(
        gt=0.0,
        le=30.0,
        description="Bulk unit weight γ  [kN/m³, 0–30]",
    )
    sat_unit_weight: float = Field(
        gt=0.0,
        le=30.0,
        description="Saturated unit weight γ_sat  [kN/m³, 0–30]",
    )

    # ------------------------------------------------------------------ #
    # Water table
    # ------------------------------------------------------------------ #
    water_table_depth: float = Field(
        ge=0.0,
        le=999.0,
        description=(
            "Depth of water table below GL  [m]. "
            "Use 999 to indicate no influence."
        ),
    )

    # ------------------------------------------------------------------ #
    # Footing geometry
    # ------------------------------------------------------------------ #
    footing_shape: Literal["strip", "rectangular", "square", "circular"] = Field(
        description="Plan shape of the footing.",
    )
    width: float = Field(
        gt=0.0,
        le=50.0,
        description="Least lateral dimension B  [m, > 0]",
    )
    length: float = Field(
        default=1.0e6,
        gt=0.0,
        le=1.0e6,
        description="Longer plan dimension L  [m].  Required for rectangular.",
    )
    depth: float = Field(
        ge=0.0,
        le=30.0,
        description="Founding depth Df  [m, ≥ 0]",
    )

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #
    load_inclination: float = Field(
        default=0.0,
        ge=0.0,
        lt=90.0,
        description="Angle of resultant load from vertical α  [degrees, 0–89]",
    )

    # ------------------------------------------------------------------ #
    # Analysis options
    # ------------------------------------------------------------------ #
    failure_mode: Literal["auto", "general", "local"] = Field(
        default="auto",
        description=(
            "Shear failure mode: "
            "'auto' selects based on φ (<28° → local), "
            "'general', or 'local'."
        ),
    )
    fos: float = Field(
        default=2.5,
        ge=1.0,
        le=10.0,
        description="Factor of Safety (IS 1904 recommends 2.5–3.0).",
    )

    # ------------------------------------------------------------------ #
    # Cross-field validators
    # ------------------------------------------------------------------ #
    @field_validator("sat_unit_weight")
    @classmethod
    def sat_must_be_gte_bulk(cls, v: float) -> float:
        """
        Saturated unit weight must be ≥ bulk unit weight.
        (Full cross-check done in model_validator below.)
        """
        return v

    @model_validator(mode="after")
    def cross_validate(self) -> "SoilInput":
        """
        Enforce cross-field physical constraints:
        - γ_sat ≥ γ_bulk
        - Rectangular footing: L must be supplied and ≥ B
        """
        if self.sat_unit_weight < self.unit_weight:
            raise ValueError(
                "Saturated unit weight γ_sat must be ≥ bulk unit weight γ. "
                f"Got γ={self.unit_weight} kN/m³, "
                f"γ_sat={self.sat_unit_weight} kN/m³."
            )
        if self.footing_shape == "rectangular":
            if self.length <= 0 or self.length >= 1.0e6:
                raise ValueError(
                    "Rectangular footing requires a valid length L > 0 m."
                )
            if self.length < self.width:
                # Swap silently so width is always the shorter dimension
                self.length, self.width = self.width, self.length
        return self

    def to_engine_dict(self) -> dict:
        """
        Convert validated model to the flat dict expected by
        ``is_6403_engine.calculate_bearing_capacity()``.
        """
        return {
            "cohesion":          self.cohesion,
            "friction_angle":    self.friction_angle,
            "unit_weight":       self.unit_weight,
            "sat_unit_weight":   self.sat_unit_weight,
            "water_table_depth": self.water_table_depth,
            "footing_shape":     self.footing_shape,
            "width":             self.width,
            "length":            self.length,
            "depth":             self.depth,
            "load_inclination":  self.load_inclination,
            "failure_mode":      self.failure_mode,
            "fos":               self.fos,
        }


class CalculationResponse(BaseModel):
    """
    Structured API response returned by ``POST /api/calculate``.
    """

    # Failure mode
    failure_mode_used: str
    c_eff:   float
    phi_eff: float

    # Bearing capacity factors
    N_c:     float
    N_q:     float
    N_gamma: float

    # Shape factors
    s_c:     float
    s_q:     float
    s_gamma: float

    # Depth factors
    d_c:     float
    d_q:     float
    d_gamma: float

    # Inclination factors
    i_c:     float
    i_q:     float
    i_gamma: float

    # Water table factors
    W_prime_q:     float
    W_prime_gamma: float

    # Capacities  [kPa]
    q_overburden: float
    term_c:       float
    term_q:       float
    term_gamma:   float
    q_f:          float
    q_nu:         float
    q_ns:         float
    q_s:          float
    fos:          float

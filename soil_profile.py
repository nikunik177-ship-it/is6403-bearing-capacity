"""
Module: Soil Profile Data Structures
Handles core geomechanical parameters, input validation, and automatic failure mode resolution.
"""
import math

# Store structural foundation type conversion rules using a Tuple (Unit II: Tuples are immutable)
SUPPORTED_SHAPES = ("strip", "square", "rectangular", "circular")

def resolve_shear_failure_mode(phi_degrees, cohesive_c, mode="auto"):
    """
    Applies IS 6403:1981 Clause 5.1.1 parameters.
    Automatically modifies shear parameters if Local Shear Failure conditions are met.
    """
    # Defensive Input Guardrails (Unit I: If Statements)
    if phi_degrees < 0 or phi_degrees > 45:
        raise ValueError("Friction angle phi must fall realistically between 0° and 45°.")
    if cohesive_c < 0:
        raise ValueError("Cohesion values cannot be mathematically negative.")

    # Rule: If auto-detection is on, phi < 28° triggers Local Shear Failure adjustments
    if mode == "auto" and phi_degrees < 28.0:
        calculated_mode = "Local Shear Failure"
        # Mathematical adjustments as per code: c' = (2/3)*c and tan(phi') = (2/3)*tan(phi)
        effective_c = (2.0 / 3.0) * cohesive_c
        phi_rad = math.radians(phi_degrees)
        effective_phi = math.degrees(math.atan((2.0 / 3.0) * math.tan(phi_rad)))
    else:
        calculated_mode = "General Shear Failure"
        effective_c = cohesive_c
        effective_phi = phi_degrees

    # Returns data using a custom Dictionary payload (Unit III: Dictionaries)
    return {
        "failure_type": calculated_mode,
        "c_eff": round(effective_c, 2),
        "phi_eff": round(effective_phi, 2)
    }

"""
Module: Codal Lookup Tables & Analytical Corrections
Handles IS 6403 Table 1 Linear Interpolation and Shape/Depth Correction Multipliers.
"""
import math

# Digital lookup translation of IS 6403:1981 Table 1 (Bearing Capacity Factors)
# Format: Friction_Angle: (Nc, Nq, Ngamma)
IS_6403_TABLE_1 = {
    0:  (5.14,  1.00,  0.00),
    5:  (6.49,  1.57,  0.45),
    10: (8.35,  2.47,  1.22),
    15: (10.98, 3.94,  2.65),
    20: (14.83, 6.40,  5.39),
    25: (20.72, 10.66, 10.88),
    30: (30.14, 18.40, 22.40),
    35: (46.12, 33.30, 48.03),
    40: (75.31, 64.20, 109.41),
    45: (133.88,134.88,271.76)
}

def interpolate_bearing_factors(phi):
    """Linearly interpolates values for fractional friction angles from Table 1."""
    angles = sorted(IS_6403_TABLE_1.keys())
    
    # Exact match check
    if phi in IS_6403_TABLE_1:
        return IS_6403_TABLE_1[phi]
        
    # Standard Step-wise Linear Interpolation Loop (Unit II: Loops)
    for i in range(len(angles) - 1):
        phi1, phi2 = angles[i], angles[i+1]
        if phi1 < phi < phi2:
            nc1, nq1, ng1 = IS_6403_TABLE_1[phi1]
            nc2, nq2, ng2 = IS_6403_TABLE_1[phi2]
            
            # Interpolation calculation weight fraction
            weight = (phi - phi1) / (phi2 - phi1)
            nc = nc1 + weight * (nc2 - nc1)
            nq = nq1 + weight * (nq2 - nq1)
            ng = ng1 + weight * (ng2 - ng1)
            return round(nc, 3), round(nq, 3), round(ng, 3)
            
    return IS_6403_TABLE_1[max(angles)]

def compute_shape_factors(shape, B, L=None):
    """Returns correction multipliers sc, sq, sg according to IS 6403 Table 2."""
    if shape == "strip":
        return 1.0, 1.0, 1.0
    elif shape == "square":
        return 1.3, 1.2, 0.8
    elif shape == "circular":
        return 1.3, 1.2, 0.6
    elif shape == "rectangular":
        if not L: 
            L = B
        return round(1.0 + 0.2 * (B / L), 2), round(1.0 + 0.2 * (B / L), 2), round(1.0 - 0.4 * (B / L), 2)
    return 1.0, 1.0, 1.0

"""
Module: Ultimate Bearing Capacity Analytical Core Engine
Leverages Object-Oriented Programming (OOP) paradigms and saves data results to files.
"""
from is_6403_factors import interpolate_bearing_factors, compute_shape_factors

class FoundationProfile:
    """Base Class encapsulating foundational properties (Unit IV: Encapsulation)."""
    def __init__(self, depth_df, width_b, factor_of_safety=2.5):
        self.df = depth_df               # Founding Depth (Df)
        self.b = width_b                 # Breadth/Width (B)
        self._fos = factor_of_safety     # Protected system safety factor
        
    def get_safety_factor(self):
        return self._fos

class AnalyticalBearingEngine(FoundationProfile):
    """Derived Class inheriting foundational data and executing analytical formulas (Unit IV: Inheritance)."""
    
    def __init__(self, depth_df, width_b, shape="strip", length_l=None, fos=2.5):
        super().__init__(depth_df, width_b, fos)
        self.shape = shape
        self.l = length_l

    def calculate_capacity(self, gamma_bulk, soil_metrics, water_table_depth):
        """Executes the generalized IS 6403 Net Ultimate Bearing Capacity Equation."""
        c = soil_metrics["c_eff"]
        phi = soil_metrics["phi_eff"]
        
        # 1. Fetch Codal Factors
        nc, nq, ng = interpolate_bearing_factors(phi)
        sc, sq, sg = compute_shape_factors(self.shape, self.b, self.l)
        
        # 2. Water Table Surcharge Factor Calculation (IS 6403 Clause 5.1.3)
        # Wq calculation based on position relative to foundation depth
        if water_table_depth <= self.df:
            wq = 0.5  # Submerged surcharge reduction
        elif water_table_depth >= (self.df + self.b):
            wq = 1.0  # Safe depth, no reduction
        else:
            # Linear interpolation inside failure zone
            wq = 0.5 + 0.5 * ((water_table_depth - self.df) / self.b)
            
        # 3. Process Overburden Surcharge Stress
        effective_surcharge_q = gamma_bulk * self.df
        
        # 4. Apply The Ultimate IS 6403 Mathematical Formula
        term1 = c * nc * sc               # Cohesion Component
        term2 = effective_surcharge_q * (nq - 1.0) * sq * wq  # Surcharge Component
        term3 = 0.5 * gamma_bulk * self.b * ng * sg * wq      # Self-Weight Wedge Component
        
        q_nu = term1 + term2 + term3
        q_ns = q_nu / self._fos            # Net Safe Bearing Capacity
        q_safe_gross = q_ns + (gamma_bulk * self.df)
        
        # Unit IV: Persistent File Logging (Appending execution data to local text log)
        with open("geotech_design_runs.txt", "a") as log_file:
            log_file.write(f"Shape: {self.shape} | Q_net_safe: {q_ns:.2f} kN/m² | Q_gross_safe: {q_safe_gross:.2f} kN/m²\n")
            
        return {
            "q_net_ultimate": round(q_nu, 2),
            "q_net_safe": round(q_ns, 2),
            "q_gross_safe": round(q_safe_gross, 2),
            "factors": (nc, nq, ng),
            "corrections": (sc, sq, sg)
        }

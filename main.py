"""
Main Application Interface 
Integrates terminal input processing, NumPy calculations, Pandas records, and Matplotlib visualization.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from soil_profile import resolve_shear_failure_mode
from bearing_calculator import AnalyticalBearingEngine

def launch_geotechnical_app():
    print("=================================================================")
    # Unit I: Printing and Input collection
    print("        IS 6403:1981 GEOTECHNICAL BEARING CAPACITY ESTIMATOR      ")
    print("=================================================================")
    
    try:
        # Prompt user for geotechnical input parameters
        c_val = float(input("Enter Soil Cohesion property (c in kN/m²): "))
        phi_val = float(input("Enter Internal Friction Angle (phi in degrees): "))
        gamma_val = float(input("Enter Bulk Unit Weight of Soil (gamma in kN/m³): "))
        b_val = float(input("Enter Foundation Base Width (B in meters): "))
        df_val = float(input("Enter Foundation Depth (Df in meters): "))
        dw_val = float(input("Enter Depth of Water Table below Ground Level (meters): "))
        
        # 1. Modular Execution Pipeline: Resolve failure modes
        soil_profile = resolve_shear_failure_mode(phi_val, c_val, mode="auto")
        print(f"\n[SYSTEM ALERT] Failure Assessment: Tracing under '{soil_profile['failure_type']}' parameters.")
        
        # 2. OOP Integration Core
        engine = AnalyticalBearingEngine(depth_df=df_val, width_b=b_val, shape="square", fos=3.0)
        results = engine.calculate_capacity(gamma_val, soil_profile, dw_val)
        
        print("\n---------------- DESIGN COMPUTATION SUMMARY ----------------")
        print(f"Interpolated Factors Table 1 -> Nc: {results['factors'][0]}, Nq: {results['factors'][1]}, Ny: {results['factors'][2]}")
        print(f"NET ULTIMATE CAPACITY (q_nu)  : {results['q_net_ultimate']} kN/m²")
        print(f"NET SAFE CAPACITY (q_ns)      : {results['q_net_safe']} kN/m² (FOS = 3.0)")
        print(f"GROSS SAFE BEARING CAPACITY   : {results['q_gross_safe']} kN/m²")
        print("------------------------------------------------------------\n")
        
        # 3. Unit V: Advanced Structural Parametric Matrix Sweeps using NumPy
        print("[INFO] Simulating Bearing Performance variance across multiple widths using NumPy arrays...")
        # Create a vectorized array of various foundation widths from 1.0m to 5.0m
        width_matrix = np.linspace(1.0, 5.0, 5) 
        safe_capacity_results = []
        
        for w in width_matrix:
            loop_engine = AnalyticalBearingEngine(depth_df=df_val, width_b=w, shape="square", fos=3.0)
            loop_res = loop_engine.calculate_capacity(gamma_val, soil_profile, dw_val)
            safe_capacity_results.append(loop_res["q_net_safe"])
            
        # 4. Unit V: Storage and Manipulation via Pandas DataFrames
        design_data = {
            "Foundation_Width_Meters": width_matrix,
            "Net_Safe_Capacity_KN_M2": safe_capacity_results
        }
        df_summary = pd.DataFrame(design_data)
        print("\n--- Pandas Compiled Summary Matrix (df.head()) ---")
        print(df_summary)
        
        # 5. Unit V: Advanced Plot Visualization Export (Matplotlib Layouts)
        plt.figure(figsize=(7, 4))
        plt.plot(df_summary["Foundation_Width_Meters"], df_summary["Net_Safe_Capacity_KN_M2"], 
                 marker='s', color='darkgreen', linestyle='--', linewidth=2, label='q_ns vs Foundation Width')
        plt.title("Bearing Capacity vs Foundation Dimension Scaling")
        plt.xlabel("Foundation Footing Base Width (B in meters)")
        plt.ylabel("Net Safe Capacity (kN/m²)")
        plt.grid(True, linestyle=':')
        plt.legend()
        plt.tight_layout()
        
        plt.savefig("soil_bearing_scaling_chart.png")
        print("\n[SUCCESS] Design plot chart saved locally as 'soil_bearing_scaling_chart.png'!")
        plt.close()
        
    except ValueError as e:
        print(f"\n[FATAL VALIDATION ERROR]: {str(e)}")

if __name__ == "__main__":
    launch_geotechnical_app()

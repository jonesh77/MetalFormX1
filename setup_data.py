# setup_data.py
import os
import numpy as np
import pandas as pd

def setup_data_directories():
    """Setup all required data directories and placeholder files"""
    print("Setting up data directories...")
    
    # Create base data directory
    base_dir = os.getcwd()
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Create subdirectories
    deform_dir = os.path.join(data_dir, "deform")
    simufact_dir = os.path.join(data_dir, "simufact")
    os.makedirs(deform_dir, exist_ok=True)
    os.makedirs(simufact_dir, exist_ok=True)
    
    # Create AISI data placeholder
    if not os.path.exists(os.path.join(data_dir, "_RAW_Processing_map_AISI4340.xlsx")):
        create_aisi_data_placeholder(data_dir)
        
    # Create Deform data placeholders
    create_deform_data_placeholders(deform_dir)
    
    # Create Simufact data placeholders
    create_simufact_data_placeholders(simufact_dir)
    
    # Create other required directories
    train_npy_dir = os.path.join(base_dir, "train_npy")
    train_npy_y_dir = os.path.join(train_npy_dir, "train_y")
    result_dir = os.path.join(base_dir, "result")
    train_data_dir = os.path.join(base_dir, "train_data")
    os.makedirs(train_npy_dir, exist_ok=True)
    os.makedirs(train_npy_y_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(train_data_dir, exist_ok=True)
    
    print("Data directories and placeholder files created successfully!")

def create_aisi_data_placeholder(data_dir):
    """Create placeholder AISI Excel file"""
    print("Creating AISI data placeholder...")
    
    # Create a DataFrame with a simple structure similar to what the app expects
    df = pd.DataFrame()
    
    # Create strain and stress columns for different temperatures
    temperatures = ['1200', '1100', '1000', '900']
    
    for i, temp in enumerate(temperatures, 1):
        for j in range(1, 5):
            idx = (i-1)*4 + j
            strain_col = f'strain{idx}'
            stress_col = f'stress{idx}'
            
            # Create sample strain data (0.1 to 1.0)
            df[strain_col] = np.linspace(0.1, 1.0, 100)
            
            # Create sample stress data - higher for lower temps, lower for higher strains
            base_stress = 100 + (5-i)*30 + j*10  # Base stress depends on temperature
            df[stress_col] = base_stress * (1 - 0.3*df[strain_col]) + np.random.normal(0, 5, 100)
    
    # Save to Excel
    file_path = os.path.join(data_dir, "_RAW_Processing_map_AISI4340.xlsx")
    df.to_excel(file_path, sheet_name="Sheet1", index=False)
    print(f"Created AISI data placeholder at {file_path}")

def create_deform_data_placeholders(deform_dir):
    """Create placeholder Deform data files"""
    print("Creating Deform data placeholders...")
    
    # Create S.dat (Strain)
    create_dat_file(
        os.path.join(deform_dir, "S.dat"),
        header=["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10", "P11", "P12", "P13", "P14", "P15", "P16"],
        rows=40,
        value_generator=lambda i, j: 0.1 + i*0.025  # Strain increases with row index
    )
    
    # Create SR.dat (Strain Rate)
    create_dat_file(
        os.path.join(deform_dir, "SR.dat"),
        header=["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10", "P11", "P12", "P13", "P14", "P15", "P16"],
        rows=40,
        value_generator=lambda i, j: 0.01 * (10 ** (j % 4))  # Different strain rates for different columns
    )
    
    # Create T.dat (Temperature)
    create_dat_file(
        os.path.join(deform_dir, "T.dat"),
        header=["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10", "P11", "P12", "P13", "P14", "P15", "P16"],
        rows=40,
        value_generator=lambda i, j: 900 + (j % 4) * 100  # Temperatures: 900, 1000, 1100, 1200
    )

def create_dat_file(file_path, header, rows, value_generator):
    """Create a .dat file with specified header and generated values"""
    with open(file_path, 'w') as f:
        # Write dummy header lines
        f.write("# Deform simulation data\n")
        f.write("# Generated placeholder\n")
        
        # Write header
        f.write("\t".join(header) + "\n")
        
        # Write data rows
        for i in range(rows):
            row_values = [f"{value_generator(i, j):.6f}" for j in range(len(header))]
            f.write("\t".join(row_values) + "\n")
    
    print(f"Created {file_path}")

def create_simufact_data_placeholders(simufact_dir):
    """Create placeholder Simufact data files"""
    print("Creating Simufact data placeholders...")
    
    # Create all.csv - combined data
    df_all = pd.DataFrame({
        'Step': np.arange(1, 101),
        'Strain': np.linspace(0.1, 1.1, 100),
        'StrainRate': np.random.uniform(0.01, 10, 100),
        'Temperature': np.random.uniform(900, 1200, 100)
    })
    df_all.to_csv(os.path.join(simufact_dir, "all.csv"), index=False)
    
    # Create s.csv - strain data
    df_s = pd.DataFrame({
        'X': np.linspace(-50, 50, 20),
        'Y': np.linspace(-50, 50, 20)
    })
    for strain in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]:
        col_name = f"Strain_{strain:.1f}"
        df_s[col_name] = np.random.uniform(strain*0.8, strain*1.2, 20)
    df_s.to_csv(os.path.join(simufact_dir, "s.csv"), index=False)
    
    # Create sr.csv - strain rate data
    df_sr = pd.DataFrame({
        'X': np.linspace(-50, 50, 20),
        'Y': np.linspace(-50, 50, 20)
    })
    for i in range(1, 11):
        col_name = f"StrainRate_{i}"
        base_rate = 0.1 * (10 ** (i % 4))
        df_sr[col_name] = np.random.uniform(base_rate*0.8, base_rate*1.2, 20)
    df_sr.to_csv(os.path.join(simufact_dir, "sr.csv"), index=False)
    
    # Create t.csv - temperature data
    df_t = pd.DataFrame({
        'X': np.linspace(-50, 50, 20),
        'Y': np.linspace(-50, 50, 20)
    })
    for i in range(1, 5):
        col_name = f"Temperature_{i}"
        base_temp = 900 + i * 75
        df_t[col_name] = np.random.uniform(base_temp*0.95, base_temp*1.05, 20)
    df_t.to_csv(os.path.join(simufact_dir, "t.csv"), index=False)
    
    print(f"Created Simufact data placeholders in {simufact_dir}")

if __name__ == "__main__":
    setup_data_directories()
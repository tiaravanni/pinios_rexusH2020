import pandas as pd
import numpy as np

# Function to read ESRI ASCII raster file
def read_ascii_raster(filename):
    with open(filename, 'r') as file:
        header = {line.split()[0]: float(line.split()[1]) for line in [next(file) for _ in range(6)]}
        data = np.loadtxt(file, skiprows=0)
    return header, data

# Function to write ESRI ASCII raster file
def write_ascii_raster(filename, header, data):
    with open(filename, 'w') as file:
        for key, value in header.items():
            file.write(f"{key} {value}\n")
        np.savetxt(file, data, fmt='%f')

# Load crop function raster
irrfunct_header, irrfunct_data = read_ascii_raster('/mnt/c/METAMODEL/irrfunct.asc')

# Load crop coefficients
kc_df = pd.read_csv('/mnt/c/METAMODEL/Kc.csv')

# Map months to their corresponding ascii files
months_files = {4: 'et01971200004.asc', 5: 'et01971200005.asc', 6: 'et01971200006.asc',
                7: 'et01971200007.asc', 8: 'et01971200008.asc', 9: 'et01971200009.asc', 10: 'et01971200010.asc'}

for month, filename in months_files.items():
    # Read the current month's ET0 raster
    et0_header, et0_data = read_ascii_raster(f'/mnt/c/METAMODEL/{filename}')

    # Initialize an empty array for the result
    result_data = np.full_like(et0_data, -9999.0)  # Use no data value for initialization

    for crop_type in np.unique(irrfunct_data):
        if crop_type == -9999.0:
            continue  # Skip no data values
        if crop_type not in kc_df['Code'].values:
            continue  # Skip crop types not in Kc.csv
        kc_value = kc_df.loc[kc_df['Code'] == crop_type, str(month)].values[0]
        mask = irrfunct_data == crop_type
        result_data[mask] = et0_data[mask] * kc_value

    # Write the result to a new ascii raster file
    result_filename = f'/mnt/c/METAMODEL/et0_kc_{month}.asc'
    write_ascii_raster(result_filename, et0_header, result_data)

print("Processing complete.")

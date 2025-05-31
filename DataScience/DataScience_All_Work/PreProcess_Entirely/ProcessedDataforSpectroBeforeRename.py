import os
import re
import pandas as pd
import numpy as np
from pathlib import Path

# === CONFIG ===
raw_base = 'AllData'        # Where your original .txt files are
processed_base = 'ProcessedTextFilesUnamed2'  # Where cleaned .csv files will go

# Columns expected in each file
column_names = [
    "Sensor1_Temp", "Sensor1_Pressure", "Sensor1_Humidity",
    "Sensor2_Temp", "Sensor2_Pressure", "Sensor2_Humidity"
]

# Dynamically detect the zones (subdirectories) in 'AllData'
zones = [zone for zone in os.listdir(raw_base) if os.path.isdir(os.path.join(raw_base, zone))]

# Make sure output folders exist for each zone
for zone in zones:
    Path(f'{processed_base}/{zone}').mkdir(parents=True, exist_ok=True)

# === HELPERS ===

def clean_and_parse_lines(file_path):
    """
    Reads a text file, extracts only numeric values line-by-line, 
    keeps lines with exactly 6 numeric values.
    """
    clean_data = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = re.split(r'[ ,]+', line.strip())
            numeric = [p for p in parts if re.match(r'^-?\d+(\.\d+)?$', p)]
            if len(numeric) == 6:  # Expect 6 values per line
                clean_data.append([float(x) for x in numeric])
    return pd.DataFrame(clean_data, columns=column_names)

def remove_outliers_iqr(df, threshold=1.0):
    """
    Removes rows with any outlier values based on the IQR (Interquartile Range) method.
    """
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    condition = ~((df < (Q1 - threshold * IQR)) | (df > (Q3 + threshold * IQR))).any(axis=1)
    return df[condition]

def process_file(file_path):
    """
    Process one raw .txt file into cleaned DataFrame.
    """
    df = clean_and_parse_lines(file_path)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df = remove_outliers_iqr(df, threshold=1.0)  # Moderate outlier filtering
    return df

# === MAIN FUNCTION ===

def preprocess_all():
    """
    Process all zones dynamically, clean all .txt files, and save cleaned .csv files.
    """
    for zone in zones:
        input_path = os.path.join(raw_base, zone)
        output_path = os.path.join(processed_base, zone)

        print(f"Processing zone: {zone}")
        
        # Check if the input folder exists and contains files
        if not os.path.exists(input_path):
            print(f"{input_path} does not exist!")
            continue
        
        files = [file for file in os.listdir(input_path) if file.endswith(".txt")]
        
        if not files:
            print(f"No .txt files found in {input_path}!")
            continue
        
        # Process each .txt file
        for file in files:
            raw_file_path = os.path.join(input_path, file)
            df_clean = process_file(raw_file_path)

            # Save cleaned file
            out_filename = os.path.splitext(file)[0] + ".csv"
            out_file_path = os.path.join(output_path, out_filename)
            df_clean.to_csv(out_file_path, index=False)

            print(f"Processed: {raw_file_path}")
            print(f"Saved to: {out_file_path}")

# Run this after making the changes
preprocess_all()

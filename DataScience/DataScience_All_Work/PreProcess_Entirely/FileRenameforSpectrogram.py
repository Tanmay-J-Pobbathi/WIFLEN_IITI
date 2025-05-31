import os
import re
import pandas as pd
import numpy as np
from pathlib import Path

# Zones and paths
zones = ['NB', 'EB', 'RB']
processed_base = 'ProcessedTextFilesUnamed2'
spectrogram_base = 'RenamedFilesForSpectogramCode2'

# Create folder structure for spectrograms
for zone in zones:
    Path(f'{spectrogram_base}/{zone}').mkdir(parents=True, exist_ok=True)

def copy_for_spectrograms(daq_csv):
    daq = pd.read_csv(daq_csv)
    daq.columns = [col.strip().replace(" ", "_") for col in daq.columns]  # standardize headers

    for zone in zones:
        time_col = f"{zone}_time"
        link_col = f"{zone}_data_link"

        if time_col not in daq.columns or link_col not in daq.columns:
            print(f"Skipping {zone} - missing columns in DAQ")
            continue

        unique_counter = 1  # Start unique ID counter

        for i, row in daq.iterrows():
            if pd.isna(row[link_col]) or pd.isna(row[time_col]):
                continue

            partial_id = str(int(float(row[link_col])))  # Convert to clean string ID
            raw_time = row[time_col]

            try:
                minutes = int(str(raw_time).split(':')[0])
                seconds = int(str(raw_time).split(':')[1])
                total_min = minutes + (1 if seconds >= 30 else 0)
            except:
                continue

            # Look for any file in processed/<zone>/ that contains the partial ID
            zone_folder = os.path.join(processed_base, zone)
            matched_file = None
            for f in os.listdir(zone_folder):
                if partial_id in f and f.endswith(".csv"):
                    matched_file = f
                    break

            if matched_file:
                unique_id = str(unique_counter).zfill(2)
                source_file = os.path.join(zone_folder, matched_file)
                dest_file = os.path.join(spectrogram_base, zone, f"{zone}_{unique_id}_{total_min}min.csv")
                pd.read_csv(source_file).to_csv(dest_file, index=False)
                print(f"Copied: {source_file} -> {dest_file}")
                unique_counter += 1
            else:
                print(f"No match found in {zone} for partial ID: {partial_id}")

# Example usage:
copy_for_spectrograms("DAQ1.csv")
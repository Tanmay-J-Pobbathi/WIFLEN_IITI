import os
import re
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from tqdm import tqdm

# === CONFIGURATION ===
zones = ['NB', 'EB', 'RB']
input_base = "RenamedFilesForSpectogramCode"
output_base = "Spectrogram_datasetsV7"
segment_duration = 30  # seconds
channel_count = 6
min_variance_threshold = 1e-5  # Increased threshold to filter low-variation signals

# Create output folders
for zone in zones:
    Path(os.path.join(output_base, zone)).mkdir(parents=True, exist_ok=True)

# Function to save spectrogram image
def save_spectrogram_image(signal, fs, save_path):
    nperseg = min(256, len(signal))  # Prevent nperseg > signal length
    if nperseg < 32:
        print(f"Segment too short (len={len(signal)}), skipping.")
        return

    noverlap = nperseg // 2
    f, t, Sxx = spectrogram(signal, fs=fs, nperseg=nperseg, noverlap=noverlap)

    # === DEBUGGING INFO ===
    print(f"[DEBUG] Signal stats -> mean: {np.mean(signal):.4f}, std: {np.std(signal):.4f}, var: {np.var(signal):.6f}")
    print(f"[DEBUG] Spectrogram Sxx -> min: {Sxx.min():.6e}, max: {Sxx.max():.6e}")
    print(f"[DEBUG] nperseg: {nperseg}, len(signal): {len(signal)}, fs: {fs:.2f}")

    plt.figure(figsize=(5, 4))
    plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-8), shading='gouraud', cmap='viridis')  # Adjusted epsilon to 1e-8
    plt.axis('off')
    plt.ylim(0, 3)  # Focus on 0–3 Hz band
    plt.colorbar()  # For debugging: shows power scale
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close()

# Track number of spectrograms created
spectrogram_counts = {zone: 0 for zone in zones}

# === PROCESS EACH ZONE ===
for zone in zones:
    zone_input_folder = os.path.join(input_base, zone)
    zone_output_folder = os.path.join(output_base, zone)

    files = [f for f in os.listdir(zone_input_folder) if f.endswith(".csv")]

    for filename in tqdm(files, desc=f"Processing {zone}", unit="file"):
        filepath = os.path.join(zone_input_folder, filename)
        try:
            df = pd.read_csv(filepath)

            if df.shape[1] != channel_count:
                print(f"Skipping {filename}: expected {channel_count} columns, got {df.shape[1]}.")
                continue

            total_samples = df.shape[0]

            # Extract info from filename using regex
            base_name = os.path.splitext(filename)[0]
            match = re.match(r"([A-Za-z]+)[_]?(\d+)[_]?.*?(\d+)min", base_name)

            if not match:
                print(f"Could not parse information from filename '{filename}'.")
                continue

            zone_name = match.group(1)   # e.g., 'RB'
            unique_id = match.group(2)   # e.g., '10'
            minutes = int(match.group(3))  # e.g., 3

            duration_seconds = minutes * 60
            fs = total_samples / duration_seconds  # Sampling frequency

            print(f"\n[DEBUG] File: {filename}")
            print(f"Total samples: {total_samples}, Duration: {duration_seconds}s, Sampling Rate: {fs:.2f} Hz")

            segment_samples = int(fs * segment_duration)
            num_segments = total_samples // segment_samples

            for seg in range(num_segments):
                start = seg * segment_samples
                end = start + segment_samples
                segment = df.iloc[start:end]

                for ch in range(channel_count):
                    ch_data = segment.iloc[:, ch].values
                    var = np.var(ch_data)

                    if var < min_variance_threshold:
                        print(f"Skipping segment {seg} channel {ch} in {filename} (low variance: {var:.6e}).")
                        continue

                    save_path = os.path.join(
                        zone_output_folder,
                        f"{zone_name}_{unique_id}_{minutes}min_seg{seg}_ch{ch}.png"
                    )
                    save_spectrogram_image(ch_data, fs, save_path)
                    spectrogram_counts[zone] += 1

            print(f"Finished processing {zone}/{filename}")

        except Exception as e:
            print(f"Error processing {zone}/{filename}: {e}")

# === SUMMARY ===
print("\nSpectrogram Generation Summary:")
for zone, count in spectrogram_counts.items():
    print(f"{zone}: {count} images generated.")

print("All spectrograms generated successfully!")

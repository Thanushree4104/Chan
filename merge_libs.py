import os
import pandas as pd

def merge_libs_data(root_folder, output_file):
    all_spectra = []

    for subdir, _, files in os.walk(root_folder):
        for file in files:
            if file.lower().startswith("ch3_lib_") and file.lower().endswith(".csv"):
                file_path = os.path.join(subdir, file)
                try:
                    # Try reading (comma first, then tab)
                    try:
                        df = pd.read_csv(file_path)
                    except:
                        df = pd.read_csv(file_path, sep="\t")

                    # Case 1: Long format (2 columns: Wavelength, Response count)
                    if df.shape[1] == 2 and "Wavelength" in df.columns:
                        df = df.set_index("Wavelength").T  # pivot → wide format
                        df.index = [file]  # label spectrum by filename

                    # Case 2: Wide format already
                    else:
                        drop_cols = [
                            "Time", "Measurement Count", "Operation Mode",
                            "Measurement Type", "Force Reset Status", "Laser Fire Status",
                            "On Off Status", "Laser Energy", "Laser Pump Current", "PRR",
                            "Delay Time", "Number of Pulses", "Integration Time", "X Factor"
                        ]
                        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
                        df.index = [file]

                    # Ensure all columns are strings
                    df.columns = df.columns.astype(str)
                    all_spectra.append(df)

                    print(f"✅ Processed {file_path} with shape {df.shape}")

                except Exception as e:
                    print(f"❌ Error reading {file_path}: {e}")

    if not all_spectra:
        print("❌ No valid spectra found.")
        return None

    # ✅ Merge all spectra (union of wavelengths)
    merged_df = pd.concat(all_spectra, axis=0).fillna(0)

    # ✅ Sort columns numerically (wavelength order)
    merged_df = merged_df.reindex(sorted(merged_df.columns, key=lambda x: float(x)), axis=1)

    # ✅ Save final dataset
    merged_df.to_csv(output_file, index=False)

    print(f"\n🎉 Final merged dataset shape: {merged_df.shape}")
    print(f"📂 Saved to {output_file}")
    return merged_df


# -------------------------
# Run the merge
# -------------------------
root_folder = r"C:\Users\K THANUSHREE\OneDrive - SSN Trust\Cha3\lib-v2\data\calibrated"   # 🔹 update this path
output_file = r"C:\Users\K THANUSHREE\OneDrive - SSN Trust\Cha3\lib-v2\merged_libs_dataset.csv"  # 🔹 where to save

merged_df = merge_libs_data(root_folder, output_file)

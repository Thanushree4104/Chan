import os
from astropy.io import fits
import pandas as pd

def merge_apxs_data(root_folder, output_file):
    all_spectra = []

    for subdir, _, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith(".pha"):   # only APXS .pha files
                file_path = os.path.join(subdir, file)
                try:
                    with fits.open(file_path) as hdul:
                        data = hdul[1].data   # extension 1 usually has spectrum
                        channels = data['CHANNEL']
                        counts = data['COUNTS']

                        # one-row dataframe for this spectrum
                        df = pd.DataFrame([counts], columns=[str(c) for c in channels])
                        df.index = [file]   # label by filename
                        all_spectra.append(df)

                    print(f"✅ Processed {file_path} with shape {df.shape}")

                except Exception as e:
                    print(f"❌ Error reading {file_path}: {e}")

    if not all_spectra:
        print("❌ No valid APXS spectra found.")
        return None

    # merge all spectra into one DataFrame (rows = spectra, columns = channels)
    merged_df = pd.concat(all_spectra, axis=0).fillna(0)

    # sort columns numerically (0,1,2,...,4095)
    merged_df = merged_df.reindex(sorted(merged_df.columns, key=lambda x: int(x)), axis=1)

    # save
    merged_df.to_csv(output_file, index=False)
    print(f"\n🎉 Final merged APXS dataset shape: {merged_df.shape}")
    print(f"📂 Saved to {output_file}")
    return merged_df


# -------------------------
# Run merge
# -------------------------
root_folder = r"C:\Users\K THANUSHREE\OneDrive - SSN Trust\Cha3\apx\data"   # 🔹 update with root path containing .pha files
output_file = r"C:\Users\K THANUSHREE\OneDrive - SSN Trust\Cha3\apx\merged_apxs_dataset.csv"

merged_apxs_df = merge_apxs_data(root_folder, output_file)

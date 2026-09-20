import os
import zipfile
import tempfile
import shutil
import pandas as pd
import numpy as np

# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"D:\adaptive_biomedical_ml"
DATA_DIR = os.path.join(BASE_DIR, "data")

# Change these names only if your ZIP filenames are different
BIDMC_ZIP = os.path.join(
    DATA_DIR,
    "raw",
    "bidmc-ppg-and-respiration-dataset-1.0.0.zip"
)

ENOSE_ZIP = os.path.join(
    DATA_DIR,
    "raw",
    "Dataset_COPD_SMOKERS_CONTROL_AIR.zip"
)

PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

os.makedirs(PROCESSED_DIR, exist_ok=True)


# ============================================================
# HELPER: EXTRACT ZIP
# ============================================================

def extract_zip(zip_path, extract_dir):
    print(f"Extracting: {os.path.basename(zip_path)}")

    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"ZIP file not found:\n{zip_path}")

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)


# ============================================================
# BIDMC PROCESSING
# ============================================================

def process_bidmc():
    print("\nProcessing BIDMC dataset...")

    temp_dir = tempfile.mkdtemp(prefix="bidmc_")

    try:
        extract_zip(BIDMC_ZIP, temp_dir)

        signal_files = []

        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                if file.endswith("_Signals.csv"):
                    signal_files.append(os.path.join(root, file))

        print(f"Found {len(signal_files)} BIDMC signal files.")

        all_features = []

        for file_path in signal_files:

            try:
                df = pd.read_csv(file_path)

                # Normalize column names
                df.columns = [str(c).strip().upper() for c in df.columns]

                features = {
                    "file": os.path.basename(file_path)
                }

                # ------------------------------------------------
                # RESPIRATION
                # ------------------------------------------------
                resp_cols = [
                    c for c in df.columns
                    if "RESP" in c
                ]

                if resp_cols:
                    resp = pd.to_numeric(
                        df[resp_cols[0]],
                        errors="coerce"
                    ).dropna()

                    if len(resp) > 0:
                        features["resp_mean"] = resp.mean()
                        features["resp_std"] = resp.std()
                        features["resp_min"] = resp.min()
                        features["resp_max"] = resp.max()
                        features["resp_range"] = resp.max() - resp.min()

                # ------------------------------------------------
                # ECG
                # ------------------------------------------------
                ecg_cols = [
                    c for c in df.columns
                    if "ECG" in c
                ]

                if ecg_cols:
                    ecg = pd.to_numeric(
                        df[ecg_cols[0]],
                        errors="coerce"
                    ).dropna()

                    if len(ecg) > 0:
                        features["ecg_mean"] = ecg.mean()
                        features["ecg_std"] = ecg.std()
                        features["ecg_min"] = ecg.min()
                        features["ecg_max"] = ecg.max()

                # ------------------------------------------------
                # PLETH / PPG
                # ------------------------------------------------
                pleth_cols = [
                    c for c in df.columns
                    if "PLETH" in c
                ]

                if pleth_cols:
                    pleth = pd.to_numeric(
                        df[pleth_cols[0]],
                        errors="coerce"
                    ).dropna()

                    if len(pleth) > 0:
                        features["ppg_mean"] = pleth.mean()
                        features["ppg_std"] = pleth.std()
                        features["ppg_min"] = pleth.min()
                        features["ppg_max"] = pleth.max()

                all_features.append(features)

            except Exception as e:
                print(f"Could not process {file_path}: {e}")

        if not all_features:
            print("No BIDMC features extracted.")
            return

        result = pd.DataFrame(all_features)

        output_path = os.path.join(
            PROCESSED_DIR,
            "bidmc_features.csv"
        )

        result.to_csv(output_path, index=False)

        print(f"BIDMC features saved to:")
        print(output_path)

        print("\nBIDMC feature preview:")
        print(result.head())

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# ============================================================
# E-NOSE PROCESSING
# ============================================================

def process_enose():

    print("\nProcessing e-nose dataset...")

    temp_dir = tempfile.mkdtemp(prefix="enose_")

    try:
        extract_zip(ENOSE_ZIP, temp_dir)

        target_files = {
            "AIR.CSV": "AIR",
            "CONTROL.CSV": "CONTROL",
            "COPD.CSV": "COPD",
            "SMOKERS.CSV": "SMOKERS"
        }

        loaded_data = []

        for root, dirs, files in os.walk(temp_dir):

            for file in files:

                # Get only the filename, ignoring folders
                filename = os.path.basename(file).strip().upper()

                if filename not in target_files:
                    continue

                file_path = os.path.join(root, file)
                group = target_files[filename]

                print(f"Found: {file_path}")
                print(f"Group: {group}")

                try:
                    df = pd.read_csv(file_path)

                    # Remove completely empty columns
                    df = df.dropna(axis=1, how="all")

                    # Add group label
                    df["group"] = group

                    loaded_data.append(df)

                    print(
                        f"Loaded {filename}: "
                        f"{df.shape[0]} rows x {df.shape[1]} columns"
                    )

                except Exception as e:
                    print(f"Could not load {file_path}: {e}")

        if not loaded_data:
            raise RuntimeError(
                "No e-nose CSV files were successfully loaded."
            )

        # ========================================================
        # COMBINE ALL GROUPS
        # ========================================================

        combined = pd.concat(
            loaded_data,
            ignore_index=True,
            sort=False
        )

        combined_path = os.path.join(
            PROCESSED_DIR,
            "enose_combined.csv"
        )

        combined.to_csv(
            combined_path,
            index=False
        )

        print("\nCombined e-nose dataset saved to:")
        print(combined_path)

        print("\nCombined shape:")
        print(combined.shape)

        print("\nGroup distribution:")
        print(combined["group"].value_counts())

        # ========================================================
        # FIND NUMERIC SENSOR COLUMNS
        # ========================================================

        numeric_columns = combined.select_dtypes(
            include=np.number
        ).columns.tolist()

        sensor_columns = [
            c for c in numeric_columns
            if c != "group"
        ]

        print("\nNumeric sensor columns detected:")

        for column in sensor_columns:
            print("  ", column)

        # ========================================================
        # SENSOR STATISTICS
        # ========================================================

        if sensor_columns:

            summary = combined.groupby("group")[
                sensor_columns
            ].agg(
                ["mean", "std", "min", "max"]
            )

            summary_path = os.path.join(
                PROCESSED_DIR,
                "enose_features.csv"
            )

            summary.to_csv(summary_path)

            print("\ne-Nose feature summary saved to:")
            print(summary_path)

        else:

            print(
                "\nWARNING: No numeric sensor columns detected."
            )

    finally:

        shutil.rmtree(
            temp_dir,
            ignore_errors=True
        )

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("========================================")
    print(" Adaptive Biomedical Monitoring")
    print(" Dataset Preprocessing")
    print("========================================")

    process_bidmc()
    process_enose()

    print("\n========================================")
    print("PREPROCESSING COMPLETE")
    print("========================================")
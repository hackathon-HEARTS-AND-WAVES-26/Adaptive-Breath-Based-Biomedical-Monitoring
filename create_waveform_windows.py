import os
import zipfile
import glob
import numpy as np
import pandas as pd

# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"D:\adaptive_biomedical_ml"

ZIP_PATH = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "bidmc-ppg-and-respiration-dataset-1.0.0.zip"
)

EXTRACT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "bidmc_raw"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "bidmc_waveform_windows.csv"
)

# ============================================================
# SETTINGS
# ============================================================

FS = 125              # BIDMC sampling frequency
WINDOW_SEC = 10       # 10 second window
STEP_SEC = 5          # 5 second overlap

WINDOW_SIZE = FS * WINDOW_SEC
STEP_SIZE = FS * STEP_SEC


# ============================================================
# EXTRACT DATASET
# ============================================================

if not os.path.exists(EXTRACT_DIR):
    print("Extracting BIDMC dataset...")
    os.makedirs(EXTRACT_DIR, exist_ok=True)

    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(EXTRACT_DIR)

    print("Extraction complete.")
else:
    print("BIDMC dataset already extracted.")


# ============================================================
# FIND SIGNAL CSV FILES
# ============================================================

csv_files = glob.glob(
    os.path.join(EXTRACT_DIR, "**", "*Signals.csv"),
    recursive=True
)

print()
print("Signal files found:", len(csv_files))

if len(csv_files) == 0:
    print("\nNo *Signals.csv files found.")
    print("Files available:")

    all_files = glob.glob(
        os.path.join(EXTRACT_DIR, "**", "*"),
        recursive=True
    )

    for f in all_files[:30]:
        print(f)

    raise SystemExit


# ============================================================
# INSPECT FIRST FILE
# ============================================================

print()
print("Inspecting first signal file:")
print(csv_files[0])

sample = pd.read_csv(csv_files[0])

print()
print("Columns found:")
for i, col in enumerate(sample.columns):
    print(i, ":", repr(col))

print()
print("Shape:", sample.shape)


# ============================================================
# AUTOMATIC SIGNAL IDENTIFICATION
# ============================================================

def find_column(columns, keywords):

    for col in columns:
        col_lower = str(col).lower()

        for keyword in keywords:
            if keyword in col_lower:
                return col

    return None


columns = list(sample.columns)

# Respiration
resp_col = find_column(
    columns,
    [
        "resp",
        "respiration",
        "breath"
    ]
)

# PPG
ppg_col = find_column(
    columns,
    [
        "ppg",
        "pleth",
        "plethysmograph"
    ]
)

# ECG fallback
ecg_col = find_column(
    columns,
    [
        "ecg",
        "ekg"
    ]
)

print()
print("Detected signals:")
print("Respiration :", resp_col)
print("PPG         :", ppg_col)
print("ECG         :", ecg_col)


# ============================================================
# BIDMC SOMETIMES USES DIFFERENT COLUMN NAMES
# ============================================================

# If respiration wasn't found, inspect likely numeric columns
if resp_col is None:

    print("\nWARNING: Respiration signal was not automatically detected.")

if ppg_col is None:

    print("\nWARNING: PPG signal was not automatically detected.")


# We need at least one physiological signal.
if resp_col is None and ppg_col is None:

    print("\nNo respiration/PPG columns detected.")
    print("Please send me the column list printed above.")

    raise SystemExit


# ============================================================
# PROCESS ALL RECORDINGS
# ============================================================

all_windows = []

print()
print("Creating temporal windows...")
print("----------------------------------------")


for file_index, file_path in enumerate(csv_files):

    filename = os.path.basename(file_path)

    # Recording ID
    recording = filename.replace("_Signals.csv", "")

    try:
        df = pd.read_csv(file_path)

    except Exception as e:
        print("Could not read:", filename)
        print(e)
        continue


    # --------------------------------------------------------
    # Find columns again for this file
    # --------------------------------------------------------

    columns = list(df.columns)

    current_resp = find_column(
        columns,
        [
            "resp",
            "respiration",
            "breath"
        ]
    )

    current_ppg = find_column(
        columns,
        [
            "ppg",
            "pleth",
            "plethysmograph"
        ]
    )


    # --------------------------------------------------------
    # Need at least one signal
    # --------------------------------------------------------

    if current_resp is None and current_ppg is None:

        print(
            f"[{file_index + 1}/{len(csv_files)}] "
            f"{recording}: no usable signals"
        )

        continue


    # --------------------------------------------------------
    # Convert signals to numeric
    # --------------------------------------------------------

    if current_resp is not None:

        resp = pd.to_numeric(
            df[current_resp],
            errors="coerce"
        ).values

    else:

        resp = None


    if current_ppg is not None:

        ppg = pd.to_numeric(
            df[current_ppg],
            errors="coerce"
        ).values

    else:

        ppg = None


    # --------------------------------------------------------
    # Number of samples
    # --------------------------------------------------------

    lengths = []

    if resp is not None:
        lengths.append(len(resp))

    if ppg is not None:
        lengths.append(len(ppg))

    if len(lengths) == 0:
        continue

    n_samples = min(lengths)


    # --------------------------------------------------------
    # Create windows
    # --------------------------------------------------------

    window_count = 0

    for start in range(
        0,
        n_samples - WINDOW_SIZE + 1,
        STEP_SIZE
    ):

        end = start + WINDOW_SIZE


        # ====================================================
        # RESPIRATION FEATURES
        # ====================================================

        if resp is not None:

            r = resp[start:end]

            r = r[np.isfinite(r)]

            if len(r) == 0:
                continue

            resp_mean = np.mean(r)
            resp_std = np.std(r)
            resp_min = np.min(r)
            resp_max = np.max(r)
            resp_range = resp_max - resp_min

            half = len(r) // 2

            resp_change = (
                np.mean(r[half:])
                -
                np.mean(r[:half])
            )

            resp_variability = (
                resp_std /
                (abs(resp_mean) + 1e-6)
            )

        else:

            resp_mean = np.nan
            resp_std = np.nan
            resp_min = np.nan
            resp_max = np.nan
            resp_range = np.nan
            resp_change = np.nan
            resp_variability = np.nan


        # ====================================================
        # PPG FEATURES
        # ====================================================

        if ppg is not None:

            p = ppg[start:end]

            p = p[np.isfinite(p)]

            if len(p) == 0:
                continue

            ppg_mean = np.mean(p)
            ppg_std = np.std(p)
            ppg_min = np.min(p)
            ppg_max = np.max(p)
            ppg_range = ppg_max - ppg_min

            half = len(p) // 2

            ppg_change = (
                np.mean(p[half:])
                -
                np.mean(p[:half])
            )

            ppg_variability = (
                ppg_std /
                (abs(ppg_mean) + 1e-6)
            )

        else:

            ppg_mean = np.nan
            ppg_std = np.nan
            ppg_min = np.nan
            ppg_max = np.nan
            ppg_range = np.nan
            ppg_change = np.nan
            ppg_variability = np.nan


        # ====================================================
        # COMBINED PHYSIOLOGICAL CHANGE
        # ====================================================

        changes = []

        if np.isfinite(resp_change):
            changes.append(abs(resp_change))

        if np.isfinite(ppg_change):
            changes.append(abs(ppg_change))

        if len(changes) > 0:

            physiological_change = np.mean(changes)

        else:

            physiological_change = np.nan


        # ====================================================
        # SAVE WINDOW
        # ====================================================

        all_windows.append({

            "recording": recording,

            "start_sample": start,

            "start_time_sec":
                start / FS,

            "resp_mean":
                resp_mean,

            "resp_std":
                resp_std,

            "resp_range":
                resp_range,

            "resp_change":
                resp_change,

            "ppg_mean":
                ppg_mean,

            "ppg_std":
                ppg_std,

            "ppg_range":
                ppg_range,

            "ppg_change":
                ppg_change,

            "resp_variability":
                resp_variability,

            "ppg_variability":
                ppg_variability,

            "physiological_change":
                physiological_change
        })

        window_count += 1


    print(
        f"[{file_index + 1}/{len(csv_files)}] "
        f"{recording}: {window_count} windows"
    )


# ============================================================
# CREATE DATAFRAME
# ============================================================

result = pd.DataFrame(all_windows)


# ============================================================
# SAFETY CHECK
# ============================================================

if result.empty:

    print()
    print("ERROR: No temporal windows were created.")
    print()
    print("This means the signal columns were not correctly detected.")
    print("Send me the 'Columns found:' output above.")
    raise SystemExit


# ============================================================
# REMOVE COMPLETELY EMPTY ROWS
# ============================================================

result = result.dropna(
    subset=[
        "resp_mean",
        "ppg_mean"
    ],
    how="all"
)


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    os.path.dirname(OUTPUT_PATH),
    exist_ok=True
)

result.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("========================================")
print("TEMPORAL DATASET CREATED")
print("========================================")

print("Shape:", result.shape)

print()
print("Columns:")
print(result.columns.tolist())

print()
print("Recording distribution:")
print(
    result["recording"]
    .value_counts()
    .head(10)
)

print()
print("Example rows:")
print(result.head())

print()
print("Saved to:")
print(OUTPUT_PATH)

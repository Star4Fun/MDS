#######################################################################################################################
# File: feature_generation.py
#
# Purpose:
#   Window-based feature extraction and label generation from BIS, HR, and ART data.
#   Saves training and test feature matrices in /features folder.
#
# NOTE:
#   - BIS is only used for label generation, not as a feature.
#   - HR and ART are used for feature extraction.
#   - Students will generate their own training and test feature sets with labels.
#######################################################################################################################
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd
import numpy as np
import pathlib

from utils import preprocess_case, WIN_SEC, STEP_SEC, BIS_TARGET_LOW, BIS_TARGET_HIGH, ART_THRESH, UNSTABLE_SHARE
from tqdm import tqdm

def series_slope(series: pd.Series) -> float:
    # if time index: use seconds as x
    if isinstance(series.index, pd.DatetimeIndex) or isinstance(series.index, pd.TimedeltaIndex):
        x = (series.index - series.index[0]).total_seconds()
    else:
        x = np.arange(len(series))

    y = series.values

    # handle all-NaN / too few points
    mask = ~np.isnan(y)
    if mask.sum() < 2:
        return np.nan

    x = x[mask]
    y = y[mask]

    # linear regression via polyfit
    slope, _ = np.polyfit(x, y, 1)
    return slope


def fft_energy(series: pd.Series) -> float:
    y = series.values
    y = y[~np.isnan(y)]
    if len(y) == 0:
        return np.nan

    # remove mean to avoid DC-dominanz
    y = y - y.mean()

    fft_vals = np.fft.rfft(y)
    power = np.abs(fft_vals) ** 2
    return power.sum()


#######################################################################################################################
# Function features_window(hr, art):
# Function to compute numerical features for HR, and ART within a window.
#
# NOTE:
#   - Try to find meaningful features that could help classify stable vs. unstable windows. If needed add additional helper functions.
#
# Input arguments:
#   - [pd.Series] hr: HR values in the window
#   - [pd.Series] art: ART values in the window
# Output argument:
#   - [dict] features: dictionary of feature_name: value
#######################################################################################################################
def features_window(hr, art):
    hr = hr.dropna()
    art = art.dropna()
    q25_hr = hr.quantile(0.25)
    q75_hr = hr.quantile(0.75)
    q25_art = art.quantile(0.25)
    q75_art = art.quantile(0.75)
    return {
        "HR_mean": hr.mean(),
        "HR_median": hr.median(),
        "HR_std": hr.std(),
        "HR_min": hr.min(),
        "HR_max": hr.max(),
        "HR_range": hr.max()-hr.min(),
        "HR_IQR": q75_hr - q25_hr,
        "HR_diff_mean": hr.diff().abs().mean(),
        "HR_slope": series_slope(hr),
        "HR_fft_energy": fft_energy(hr),
        "ART_mean": art.mean(),
        "ART_std": art.std(),
        "ART_min": art.min(),
        "ART_max": art.max(),
        "ART_median": art.median(),
        "ART_range": art.max()-art.min(),
        "ART_IQR": q75_art - q25_art,
        "ART_variation": art.diff().abs().mean(),
        "ART_slope": series_slope(art),
    }

def is_unstable_from_bis(bis: pd.Series, threshold: float = 0.3) -> bool:
    bis = bis.dropna()
    if bis.empty:
        return False
    frac_unstable = ((bis < BIS_TARGET_LOW) | (bis > BIS_TARGET_HIGH)).mean()
    return frac_unstable >= UNSTABLE_SHARE


#######################################################################################################################
# Function process_case(file_path):
# Function to process one patient:
#   - Load CSV
#   - Apply preprocessing
#   - Apply sliding windows
#   - Compute features and labels
#   - Drop windows with too many NaN values
#
# NOTE:
#   - A window is dropped if >50% of its values are NaN in any of the signals.
#
# Input argument:
#   - [Path] file_path: path to patient CSV file
# Output arguments:
#   - [pd.DataFrame] X: feature matrix
#   - [pd.Series] y: label vector
#######################################################################################################################
def process_case(file_path, nan_threshold=0.5):
    df_raw = pd.read_csv(file_path)

    df = preprocess_case(df_raw)
    hr = df['HR']
    art = df['ART']
    bis = df['BIS']

    basename = pathlib.Path(file_path).name
    case_id = (basename or "").split("_")[1] if "_" in (basename or "") else None

    all_features = []
    all_labels = []

    for start in range(0, len(hr) - WIN_SEC + 1, STEP_SEC):
        end = start + WIN_SEC
        hr_win = hr.iloc[start:end]
        art_win = art.iloc[start:end]
        bis_win = bis.iloc[start:end]

        nan_fraction = pd.concat([hr_win, art_win], axis=1).isna().mean().max()
        if nan_fraction > nan_threshold:
            continue

        feature_dict = features_window(hr_win, art_win)
        new_dict = {"case_id": case_id}
        new_dict.update(feature_dict)
        all_features.append(new_dict)

        label = 1 if is_unstable_from_bis(bis_win) else 0
        all_labels.append(label)

    return pd.DataFrame(all_features), pd.Series(all_labels, name="label")

def process_case_helper(file_path, nan_threshold=0.5):
    X, Y = process_case(file_path, nan_threshold)

    if X is None or Y is None:
        return None, None

    if X.empty or Y.empty:
        return None, None

    return X, Y

#######################################################################################################################
# Function process_dataset(split="train"):
# Function to process an entire dataset split (train or test).
#
# NOTE:
#   - Saves X_train.csv and y_train.csv for training.
#   - Saves X_test.csv and y_test.csv for testing.
#   - BIS is only used for label generation, never as a feature.
#
# Input arguments:
#   - [str] split: "train" or "test"
# Output:
#   - saves feature matrix (and labels) to /features folder
#######################################################################################################################
def process_dataset(split="train"):
    assert split in ["train", "test"], "split must be 'train' or 'test'"
    DATA_DIR = Path(f"data/data_{split}")
    files = sorted(DATA_DIR.glob("case_*_BIS_HR_ART_1hz.csv"))
    assert files, "No patient files found in data_train/."

    with ThreadPoolExecutor() as executor:
        results = list(
            tqdm(
                executor.map(process_case_helper, files),
                total=len(files),
                desc="Processing cases",
            )
        )
    x, y = zip(*results)

    x_concated = pd.concat(x, ignore_index=True)
    y_concated = pd.concat(y, ignore_index=True)

    features_dir = Path("features")
    features_dir.mkdir(parents=True, exist_ok=True)

    x_path = features_dir / f"X_{split}.csv"
    y_path = features_dir / f"y_{split}.csv"

    x_concated.to_csv(x_path, index=False)
    y_concated.to_csv(y_path, index=False)


if __name__ == "__main__":
    process_dataset("train")
    process_dataset("test")
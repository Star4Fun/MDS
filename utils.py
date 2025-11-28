#######################################################################################################################
# File: utils.py
#
# Purpose:
#   Shared utility functions for preprocessing, sanity checks, and constants used in BIS/HR/ART data analysis.
#######################################################################################################################

import numpy as np
import pandas as pd

#######################################################################################################################
# Global constants, do not change! Use for preprocessing! 
#######################################################################################################################
BIS_MIN, BIS_MAX = 10.0, 100.0
HR_MIN, HR_MAX = 30.0, 250.0
ART_MIN, ART_MAX = 20.0, 200.0

BIS_TARGET_LOW, BIS_TARGET_HIGH = 40.0, 60.0
ART_THRESH = 65.0
UNSTABLE_SHARE = 0.30

WIN_SEC = 300   # 5 minutes
STEP_SEC = 150  # 50% overlap


#######################################################################################################################
# Function preprocess_case(df):
# Function to clean raw BIS/HR/ART data by applying plausibility checks and imputations.
#
# NOTE:
#   - Implausible values outside physiological ranges are set to NaN.
#   - HR values are updated less frequently (every 1–2 seconds) → forward fill is applied.
#
# Input argument:
#   - [pd.DataFrame] df: raw dataframe with columns ["sec", "BIS", "HR", "ART"]
# Output arguments:
#   - [pd.DataFrame] df: cleaned dataframe
#######################################################################################################################
def preprocess_case(df):
    df = df.copy()
    # TODO set implausible values to NaN
    df.loc[(df["BIS"] < BIS_MIN) | (df["BIS"] > BIS_MAX), "BIS"] = np.nan
    df.loc[(df["HR"]  < HR_MIN)  | (df["HR"]  > HR_MAX),  "HR"]  = np.nan
    df.loc[(df["ART"] < ART_MIN) | (df["ART"] > ART_MAX), "ART"] = np.nan

    # TODO forward fill HR values
    df["HR"] = df["HR"].ffill()

    # um fortlaufende Indexe zu haben
    df = df.reset_index(drop=True)

    return df


#######################################################################################################################
# Function sanity_checks(df, name):
# Function to run basic data integrity checks (monotonic time, duplicates, missing values).
#
# NOTE:
#   - Helps detect corrupted or incomplete files before feature extraction.
#
# Tips:
#   - Can be extended to include checks for unrealistic signal variance or constant values.
#
# Input arguments:
#   - [pd.DataFrame] df: dataframe containing signals
#   - [str] name: case file name
# Output:
#   - prints warning messages if problems are detected
#######################################################################################################################
def sanity_checks(df, name):
    problems = []
    if not df["sec"].is_monotonic_increasing:
        problems.append("sec is not monotonic increasing")
    if df.duplicated(subset=["sec"]).any():
        problems.append("duplicate sec values found")
    for col in ["BIS", "HR", "ART"]:
        if df[col].isna().mean() > 0.2:
            problems.append(f"{col}: >20% missing")
    if problems:
        print(f"[WARN] {name}: " + " | ".join(problems))
    else:
        print(f"[OK]   {name}: basic checks passed")


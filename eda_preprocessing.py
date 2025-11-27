#######################################################################################################################
# File: eda_preprocessing.py
#
# Purpose:
#   Exploratory Data Analysis (EDA) script for BIS, HR, ART.
#   - Loads raw patient data from data_train/
#   - Applies preprocessing and sanity checks (functions from utils.py)
#   - Visualizes signals with thresholds
#
# NOTE:
#   - Preprocessing functions (preprocess_case, sanity_checks) are defined in utils.py
#   - This script is only for exploration and visualization, not for feature generation
#######################################################################################################################

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from utils import preprocess_case, sanity_checks, BIS_TARGET_LOW, BIS_TARGET_HIGH, ART_THRESH


#######################################################################################################################
# Function plot_signals(df, title):
# Function to plot BIS, HR, and ART signals with thresholds.
#
# NOTE:
#   - BIS target range (40–60) is highlighted in green
#   - ART threshold (65 mmHg) is shown as a dashed red line
#   - Threshold constants are imported from utils.py
#
# Input arguments:
#   - [pd.DataFrame] df: patient data with columns ["sec", "BIS", "HR", "ART"]
#   - [str] title: title for the plot
# Output:
#   - None (displays matplotlib plot)
#######################################################################################################################
def plot_signals(df, title):
    time = df["sec"]
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(12,8))
    axes[0].plot(time, df["BIS"])
    axes[0].set_ylabel("BIS")
    axes[0].set_title(title)
    #df["HR"].interpolate() füllt fehlende Werte ->Lücken füllen, damit die Kurve nicht abbricht.
    #limit_direction="both" = auch am Anfang und Ende der Zeitreihe werden fehlende Werte ersetzt.
    axes[1].plot(time, df["HR"].interpolate(limit_direction="both"))
    axes[1].set_ylabel("HR [bpm]")
    axes[2].plot(time, df["ART"])
    axes[2].set_ylabel("ART")
    fig.show()


#######################################################################################################################
# Main execution
#
# Workflow:
#   1. Load one patient file from data_train/
#   2. Run sanity checks (utils.sanity_checks)
#   3. Apply preprocessing (utils.preprocess_case)
#   4. Visualize signals with plot_signals()
#
# TODO:
#   - Highlight periods where BIS < 40 or BIS > 60.
#   - Compute and print descriptive statistics (min, max, mean) for BIS, HR, ART.
#   - Compare preprocessed vs. raw data.
#######################################################################################################################
if __name__ == "__main__":
    DATA_DIR = Path("data/data_train")
    files = sorted(DATA_DIR.glob("case_*_BIS_HR_ART_1hz.csv"))
    assert files, "No patient files found in data_train/."

    # TODO Load first patient file
    df = pd.read_csv(files[0])

    # TODO Run function sanity checks
    utils.sanity_checks(df, files[0])

    # TODO Apply preprocessing
    utils.preprocess_case(df)

    # TODO Plot signals
    plot_signals(df, files[0])

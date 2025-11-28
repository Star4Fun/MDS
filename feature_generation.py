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

from pathlib import Path
import pandas as pd
import numpy as np
from utils import preprocess_case, WIN_SEC, STEP_SEC, BIS_TARGET_LOW, BIS_TARGET_HIGH, ART_THRESH, UNSTABLE_SHARE
from tqdm import tqdm


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
    # TODO implment features
    pass 


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
    # TODO load CSV as dataframe
    
    # TODO apply preprocessing on dataframe
   
    # TODO implement sliding windows
    # Hint:
    #   - You can implement sliding windows using a for-loop.
    #   - The loop should move through the DataFrame in steps of STEP_SEC.
    #   - Each window should have a fixed length WIN_SEC.
    #   - Be careful not to exceed the length of the DataFrame at the end.

    # TODO implment for-loop for sliding windows
    
        # TODO check NaN proportion per column, skip window if too many NaNs

        # TODO compute features and label for the window

        # TODO append to X and y
    pass


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


    # TODO iterate over all patient files in the specified split

        # TODO process one case into features + labels
        
        # TODO Skip completely empty results (no valid windows for this patient)

    # If after all patients nothing valid was found, stop early

    # TODO concatenate all patients into one feature matrix and one label vector
    

    # TODO save feature matrix and labels to CSV
    pass 




if __name__ == "__main__":
    # Example usage
    process_dataset("train")
    process_dataset("test")

    ###################################################################################################################
    # TODO::
    #   - Run both pipelines to generate X_train/y_train and X_test/y_test
    #   - Use only HR and ART as features
    #   - Use BIS only for label generation, not as a feature!
    #   - Train models on training data and evaluate on the test set!
    #   - Experiment with different features, you are free to apply feature selection techniques 
    #   - Please do not change train /test split or label logic!
    ###################################################################################################################

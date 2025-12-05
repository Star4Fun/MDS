#######################################################################################################################
# File: classification_kfold.py
#
# Purpose:
#   Train and evaluate different classifiers on HR/ART feature data.
#   Evaluation follows challenge rules using Group-K-Fold Cross-Validation.
#
# Workflow:
#   1. Load feature matrix (X_train.csv) and labels (y_train.csv).
#   2. Try multiple classifiers.
#   3. Evaluate all models with Group-K-Fold Cross-Validation (case_id = grouping variable).
#   4. Select best model based on mean Accuracy across folds.
#   5. Train best model on full training data.
#   6. Evaluate on X_test locally for reference.
#   7. Save predictions + automatically generate results summary as .txt.
#######################################################################################################################

import pandas as pd
from pathlib import Path

from pandas import Series, DataFrame
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
# Cross Validation imports
from sklearn.model_selection import GroupKFold, cross_val_score


class Perceptron(ClassifierMixin, BaseEstimator):
    def __init__(self, lr=0.1, max_iter=1000):
        self.lr = lr
        self.max_iter = max_iter

    def fit(self, X, y):
        X = np.c_[X, np.ones(len(X))]   # Bias-Term
        self.w_ = np.zeros(X.shape[1])

        for _ in range(self.max_iter):
            misclassified = False
            for xi, yi in zip(X, y):
                y_pred = np.sign(self.w_.dot(xi))

                if y_pred != yi:
                    print(yi, xi)
                    self.w_ += self.lr * yi * xi
                    misclassified = True

            if not misclassified:
                break

        return self

    def predict(self, X):
        X = np.c_[X, np.ones(len(X))]
        return np.sign(X.dot(self.w_))

#######################################################################################################################
# Function load_data(split="train")
#
# Purpose:
#   Load features + labels for training OR test, files generated in feature_generation.py.
#
# Inputs:
#   [str] split: "train" or "test"
#
# Outputs:
#   [pd.DataFrame] X : feature matrix (contains probably case_id!)
#   [pd.Series]   y : labels
#######################################################################################################################
def load_data(split="train"):
    assert split in ["train", "test"], "split must be 'train' or 'test'"
    features_dir = Path("features")
    assert features_dir, f"No features found in {features_dir}/."
    features_dir = Path("features")
    x_path = features_dir / f"X_{split}.csv"
    y_path = features_dir / f"y_{split}.csv"
    x = pd.read_csv(x_path)
    y = pd.read_csv(y_path, index_col=0, squeeze=True)
    return x, y


#######################################################################################################################
# Function cross_validate_model(model, X, y, groups, cv_splits=5)
#
# Purpose:
#   Evaluate classifier with Group-K-Fold Cross-Validation (challenge evaluation method).
#
# Inputs:
#   [sklearn model] model
#   [pd.DataFrame]  X : feature matrix
#   [pd.Series]     y : labels
#   [pd.Series] groups : case_id for Group-K-Fold
#   [int] cv_splits  : number of CV folds, please use 5
#
# Outputs:
#   Returns dict with mean/std for Accuracy and F1 across folds
#######################################################################################################################
def cross_validate_model(model, X, y, groups, cv_splits=5):
    X_nocase = X.drop(columns=["case_id"])
    gkf = GroupKFold(n_splits=cv_splits)

    acc = cross_val_score(model, X_nocase, y, groups=groups, cv=gkf, scoring="accuracy")
    f1  = cross_val_score(model, X_nocase, y, groups=groups, cv=gkf, scoring="f1")

    print(f"CV ({model.__class__.__name__}): ACC={acc.mean():.3f}±{acc.std():.3f},  F1={f1.mean():.3f}±{f1.std():.3f}")

    return {
        "model"   : model,
        "name"    : model.__class__.__name__,
        "mean_acc": acc.mean(),
        "std_acc" : acc.std(),
        "mean_f1" : f1.mean(),
        "std_f1"  : f1.std()
    }


#######################################################################################################################
# Function evaluate_on_test(model, X_train, y_train, X_test, y_test)
#
# Purpose:
#   Train best model fully using all training data and evaluate on held-out test set.
#   This function performs the final model fitting AFTER Cross-Validation.
#
# Inputs:
#   [sklearn model] model       : the selected best model from Cross-Validation
#   [pd.DataFrame]  X_train     : full training feature matrix (contains case_id)
#   [pd.Series]     y_train     : training labels
#   [pd.DataFrame]  X_test      : hold-out test feature matrix (contains case_id)
#   [pd.Series]     y_test      : test labels for evaluation
#
# Outputs:
#   [float]        acc          : accuracy on hold-out test set
#   [float]        f1           : F1 score on hold-out test set
#   [np.ndarray]   y_pred_test  : predicted test labels (used for saving .csv)
#######################################################################################################################
def evaluate_on_test(model, X_train, y_train, X_test, y_test):

    X_train_nocase = X_train.drop(columns=["case_id"])
    X_test_nocase  = X_test.drop(columns=["case_id"])

    # Train final model on full training data (after CV selection)
    model.fit(X_train_nocase, y_train)

    # Predict on test set, used for metrics AND saving submission file
    y_pred_test = model.predict(X_test_nocase)

    # Metrics for local reference only (not the challenge score!)
    acc = accuracy_score(y_test, y_pred_test)
    f1  = f1_score(y_test, y_pred_test)

    print("\nEvaluation on hold-out TEST set (not used for challenge ranking)")
    print(f"Test ACC={acc:.3f},  Test F1={f1:.3f}\n")

    return acc, f1, y_pred_test



#######################################################################################################################
# Main execution
#######################################################################################################################
if __name__ == "__main__":

    # Load data
    X_train, y_train = load_data("train")
    X_test, y_test   = load_data("test")

    # Required for Group-K-Fold, depends on how you created the feature matrix!
    groups = X_train["case_id"]

    clf = Perceptron(lr=0.1)
    clf.fit(X_train, y_train)

    # TODO Define classifiers
    classifiers = [
        clf
    ]


    ###############################################################################################################
    # Train & evaluate via Group-K-Fold CV
    ###############################################################################################################
    results = []
    # cross-validate all models
    for clf in classifiers:
        r = cross_validate_model(clf, X_train, y_train, groups, cv_splits=5)
        results.append(r)


    ###############################################################################################################
    # Select best model = highest mean Accuracy across folds (challenge metric)
    ###############################################################################################################
    best = max(results, key=lambda x: x["mean_acc"])
    best_model = best["model"]

    print("\nBest model selected:", best["name"])
    print(f"mean_ACC={best['mean_acc']:.3f}   mean_F1={best['mean_f1']:.3f}")
    print("----------------------------------------------------------------------------------")


    ###############################################################################################################
    # Final evaluation on TEST set + prediction generation
    ###############################################################################################################
    test_acc, test_f1, y_pred_test = evaluate_on_test(
        best_model, X_train, y_train, X_test, y_test
    )

    pred_path = Path("features/y_pred_test.csv")
    pd.Series(y_pred_test).to_csv(pred_path, index=False)


    ###############################################################################################################
    # Write summary results to .txt for upload/presentation
    ###############################################################################################################
    report = Path("features/model_results.txt")

    with report.open("w") as f:
        f.write("RESULT SUMMARY\n")
        f.write("----------------------------------------------------------------------------------\n")
        f.write("Cross-Validation (Group-K-Fold) results:\n")
        for r in results:
            f.write(f"{r['name']}: ACC={r['mean_acc']:.4f}±{r['std_acc']:.4f},  "
                    f"F1={r['mean_f1']:.4f}±{r['std_f1']:.4f}\n")

        f.write("\nSelected best model:\n")
        f.write(f"{best['name']}: ACC={best['mean_acc']:.4f}, F1={best['mean_f1']:.4f}\n")

        f.write("\nLocal TEST evaluation:\n")
        f.write(f"ACC={test_acc:.4f}, F1={test_f1:.4f}\n")



    print(f"\nSaved: {report}")
    print(f"Predictions stored: {pred_path}")
    print("\nDone.")

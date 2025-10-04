"""
main.py
-------------------------------------------------------------------------------
 Implemented from scratch:
    - Entropy, Gini index, Misclassification error
    - Information Gain
    - Chi-square pre-pruning (α configurable)
    - Tree construction and classification

    Used libraries so far:
    - pandas (for data handling)
    - numpy (for math)
    - scipy.stats.chi2.ppf (chi-square)
    - sklearn.model_selection.train_test_split (for splitting data)

Everything else (metrics, impurity, gain, etc.) was implemented by hand.
-------------------------------------------------------------------------------
"""

import time
import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split  # allowed
import decisionTree as DT  # our implementation


# -------------------------------
# simple helper functions
# -------------------------------
def impute_missing_values(df: pd.DataFrame) -> None:
    """
    replace missing values:
        - numeric columns: median
        - categorical columns: most frequent value (mode)
    ensures the tree can split all rows cleanly
    """
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            mode = df[col].mode(dropna=True)
            df[col] = df[col].fillna(mode.iloc[0] if len(mode) else "**MISSING**")


def accuracy(y_true, y_pred):
    """Compute simple accuracy = (# correct) / (total)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float((y_true == y_pred).mean())


def balanced_accuracy(y_true, y_pred):
    """
    balanced accuracy = (recall of positive + recall of negative) / 2
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # recall for class 1
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0  # recall for class 0
    return 0.5 * (tpr + tnr)


def confusion_matrix(y_true, y_pred):
    """
    Simple 2x2 confusion matrix for binary classification:
        [[TN, FP],
         [FN, TP]]
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))
    return np.array([[tn, fp], [fn, tp]])


# main logic
def main():
    parser = argparse.ArgumentParser(description="Decision Tree runner w/ optional Kaggle submission mode")
    parser.add_argument("--submit", action="store_true", help="enable Kaggle submission mode (train on all data + predict test.csv)")
    parser.add_argument("--data_csv", type=str, default="train.csv", help="training data with labels")
    parser.add_argument("--test_csv", type=str, default="test.csv", help="unlabeled test data for submission")
    parser.add_argument("--sample_sub", type=str, default="sample_submission.csv", help="Kaggle sample submission format")
    args = parser.parse_args()

    DATA_CSV = args.data_csv
    TARGET = "isFraud"

    print("loading dataset")
    df = pd.read_csv(DATA_CSV)
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found in dataset!")

    # Separate features (X) and labels (y)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # Handle missing data
    impute_missing_values(X)

    # if submission mode: train on full train.csv and predict test.csv
    if args.submit:
        print("\nkaggle mode on")
        print("training on ALL of train.csv, no validation split.")
        train_df = pd.concat([X.reset_index(drop=True),
                              y.reset_index(drop=True).rename(TARGET)], axis=1)

        tree = DT.decisionTree(
            train_df,
            impurity="entropy",
            alpha=0.1,
            max_depth=14,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features=None,
            numeric_thresholds=32
        )

        start_time = time.time()
        root = tree.DT_construction()
        print(f"tree built successfully in {time.time() - start_time:.2f} seconds.\n")

        # load test.csv and impute missing data
        test_df = pd.read_csv(args.test_csv)
        impute_missing_values(test_df)

        print("predicting on test.csv")
        y_pred_test = [tree.DT_classify(row, root) for _, row in test_df.iterrows()]

        # load sample submission for correct format
        sample = pd.read_csv(args.sample_sub)
        if len(sample) != len(y_pred_test):
            print(f"length mismatch! sample_sub has {len(sample)}, predictions have {len(y_pred_test)}")

        sample.iloc[:, -1] = y_pred_test  # put predictions into last column
        sample.to_csv("submission.csv", index=False)
        print("\nsubmission.csv written successfully ready for kaggle upload")

        return  # done with submission mode

    # otherwise: normal training/testing workflow
    print("splitting dataset into train, validation, and test sets")
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25,
        random_state=42, stratify=y_train_val
    )

    print(f"train size: {len(X_train)}, val size: {len(X_val)}, test size: {len(X_test)}")

    # 3.train Decision Tree
    print("\ntraining decision tree")
    train_df = pd.concat([X_train.reset_index(drop=True),
                          y_train.reset_index(drop=True).rename(TARGET)], axis=1)

    # parameters can be tuned as needed
    tree = DT.decisionTree(
        train_df,
        impurity="ginis",        # 'gini' | 'misclassification' | 'entropy'
        alpha=0.01,                 # chi-square pruning threshold
        max_depth=14,              # limit tree depth to avoid overfitting
        min_samples_split=10,
        min_samples_leaf=5,
        max_features=None,         # use all features per split
        numeric_thresholds=32      # quantile based thresholds per numeric feature
    )

    start_time = time.time()
    root = tree.DT_construction()
    print(f"tree built successfully in {time.time() - start_time:.2f} seconds.")

    # 4. Evaluate Performance
    print("\nevaluating tree on validation and test sets")

    def predict(tree, root, X):
        return [tree.DT_classify(row, root) for _, row in X.iterrows()]

    # validation predictions
    y_pred_val = predict(tree, root, X_val)
    y_pred_test = predict(tree, root, X_test)

    # validation metrics
    acc_val = accuracy(y_val, y_pred_val)
    bacc_val = balanced_accuracy(y_val, y_pred_val)
    cm_val = confusion_matrix(y_val, y_pred_val)

    # test metrics
    acc_test = accuracy(y_test, y_pred_test)
    bacc_test = balanced_accuracy(y_test, y_pred_test)
    cm_test = confusion_matrix(y_test, y_pred_test)

    # 5. Print Results
    print("\nRESULTS")
    print(f"Validation Accuracy: {acc_val:.4f} | Balanced Accuracy: {bacc_val:.4f}")
    print(f"Validation Confusion Matrix:\n{cm_val}\n")
    print(f"Test Accuracy: {acc_test:.4f} | Balanced Accuracy: {bacc_test:.4f}")
    print(f"Test Confusion Matrix:\n{cm_test}")

# only run main if file executed directly
if __name__ == "__main__":
    main()

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

import os
import time
import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split  # allowed
import decisionTree as DT  # our implementation
import randomForest as RandomForest


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





def main():
    TARGET = "isFraud"
    #DATA_CSV = "train.csv"
    #TEST_CSV = "test.csv"
    #SAMPLE_SUB = "sample_submission.csv"
    DATA_CSV = input("Enter the file path for training data(e.g train.csv): ")
    TEST_CSV = input("Enter the file path for training data(e.g test.csv): ")
    SAMPLE_SUB = input("Enter the file path for training data(e.g sample_submission.csv): ")

    print("Loading dataset")
    df = pd.read_csv(DATA_CSV)
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found!")

    #split features and labels
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    impute_missing_values(X)

    #split for local validation
    print("Splitting dataset (train/val/test)")
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
    )
    print(f"Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")


    #select the model here

    USE_FOREST = True  # change to False to use a single Decision Tree


    if not USE_FOREST:
        print("\nTraining Decision Tree")
        train_df = pd.concat([X_train.reset_index(drop=True),
                              y_train.reset_index(drop=True).rename(TARGET)], axis=1)

        tree = DT.decisionTree(
            train_df,
            impurity="entropy",       # 'gini', 'misclassification', or 'entropy'
            alpha=0.1,                # Chi-square pruning threshold
            max_depth=14,             # Depth limit to reduce overfitting
            min_samples_split=10,
            min_samples_leaf=5,
            max_features=None,
            numeric_thresholds=32
        )

        start = time.time()
        root = tree.DT_construction()
        print(f"Tree built in {time.time() - start:.2f}s")

        #predictions
        y_pred_val = [tree.DT_classify(row, root) for _, row in X_val.iterrows()]
        y_pred_test = [tree.DT_classify(row, root) for _, row in X_test.iterrows()]

    else:
        print("\nTraining Random Forest")
        forest = RandomForest.RandomForest(
            alpha=0.9,                # Chi-square pruning threshold
            numTrees=5,
            maxDepth=15,
            impurity="ginis",          # 'gini', 'misclassification', or 'entropy'
            xInput=X_train,
            yInput=y_train,
            minSampleSplit=5,
            minSampleLeaf=1,
            maxFeatures=50,
            numericThresholds=32
        )

        start = time.time()
        forest.trainTrees()
        print(f"Forest built in {time.time() - start:.2f}s")

        y_pred_val = [forest.predict(row) for _, row in X_val.iterrows()]
        y_pred_test = [forest.predict(row) for _, row in X_test.iterrows()]


    #metrics

    print("\nValidation Results")
    print(f"Accuracy: {accuracy(y_val, y_pred_val):.4f}")
    print(f"Balanced Accuracy: {balanced_accuracy(y_val, y_pred_val):.4f}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_val, y_pred_val)}")

    print("\nTest Results")
    print(f"Accuracy: {accuracy(y_test, y_pred_test):.4f}")
    print(f"Balanced Accuracy: {balanced_accuracy(y_test, y_pred_test):.4f}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred_test)}")


    #write submission file
    print("\nWriting Kaggle submission file")
    test_df = pd.read_csv(TEST_CSV)
    impute_missing_values(test_df)

    if USE_FOREST:
        y_pred_submission = [forest.predict(row) for _, row in test_df.iterrows()]
    else:
        y_pred_submission = [tree.DT_classify(row, root) for _, row in test_df.iterrows()]

    #match sample format
    sample = pd.read_csv(SAMPLE_SUB)
    sample.iloc[:, -1] = y_pred_submission
    sample.to_csv("submission.csv", index=False)
    print("submission.csv written successfully.")


if __name__ == "__main__":
    main()
import pandas as pd
import numpy
import time
from scipy import stats
from sklearn.model_selection import train_test_split

from decisionTree import decisionTree


sample = pd.read_csv('sample_submission.csv')
testD = pd.read_csv('test.csv')
trainD = pd.read_csv('train.csv')

"D = one of the data sets"

class RandomForest:
    def __init__(self, alpha=None, numTrees=5, maxDepth=None, impurity="ginis", xInput=None, yInput=None, minSampleSplit = 10, minSampleLeaf=5, maxFeatures=None, numericThresholds=32):
        self.numTrees = numTrees
        self.alpha = alpha
        self.xInput = xInput
        self.yInput = yInput
        self.impurity = impurity
        self.maxDepth = maxDepth
        self.train_df = None
        self.min_sample_split = minSampleSplit
        self.min_sample_leaf = minSampleLeaf
        self.maxFeatures = maxFeatures
        self.numericThresholds = numericThresholds
        self.trees = []
        self.roots = []

        print("Creating " + str(numTrees) + " tree(s) in a random forest using " + str(impurity) + " method for impurity measure")


    def trainTrees(self):

        for i in range(0, self.numTrees):
            print("\nGenerating training data for tree " + str(i+1))
            X_train_val, X_test, y_train_val, y_test = train_test_split(
                self.xInput, self.yInput, test_size=0.2, random_state=42,shuffle=True, stratify=self.yInput
            )

            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val, test_size=0.25,
                random_state=42, stratify=y_train_val, shuffle=True
            )
            self.train_df = pd.concat([X_train.reset_index(drop=True),
                                  y_train.reset_index(drop=True).rename("isFraud")], axis=1)
            print("\tTraining data generated.")
            tree = decisionTree(
                self.train_df,
                self.impurity,  # 'gini' | 'misclassification' | 'entropy'
                alpha=self.alpha,  # chi-square pruning threshold
                max_depth=self.maxDepth,  # limit tree depth to avoid overfitting
                min_samples_split=self.min_sample_split,
                min_samples_leaf=self.min_sample_leaf,
                max_features=self.maxFeatures,  # use all features per split
                numeric_thresholds=self.numericThresholds
            )
            print("\tTraining tree " + str(i+1) + " of " + str(self.numTrees) + "...")
            start_time = time.time()
            root = tree.DT_construction()
            print(f"Training done in {time.time() - start_time:.3f} seconds")

            self.trees.append(tree)
            print("\tTree appended.")
            self.roots.append(root)
            print("\tRoot appended.")


    def predict(self,X):
        predictions = []
        for i in range(0, self.numTrees):
            predictions.append(self.trees[i].DT_classify(X, self.roots[i]))

        total = sum(predictions)
        result = round(total)
        return result

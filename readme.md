# Project 1: Random Forest

## How to run

### Hyperparameter Setting
To change parameters please manually change anything of interest via lines 132-138 or 152-161
depending on if you are running a single tree or forest respectively.

Tree:
```angular2html
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
```
Forest:
```angular2html
     forest = RandomForest.RandomForest(
            alpha=0.9,               # Chi-square pruning threshold
            numTrees=5,
            maxDepth=15,
            impurity="gini",          # 'gini', 'misclassification', or 'entropy'
            xInput=X_train,
            yInput=y_train,
            minSampleSplit=5,
            minSampleLeaf=1,
            maxFeatures=50,
            numericThresholds=32
        )
```

To pick between a tree or forest please change the boolean on line 123.
```angular2html
    USE_FOREST = True  # change to False to use a single Decision Tree

```
After picking parameters simply run the program and it will provide a submission.csv file
There will also be a need to input the file paths for the csv files.
based on the sample_submission.csv format. 

```angular2html
python3 main.py --submit --data_csv my_train.csv --test_csv my_test.csv --sample_sub my_sample.csv

```




### Contributors

- Jesus Aguirre | Implemented Decision Tree Class, Node Class
- Carly Salazar | Implemented Decision Tree Class, Main.py
- Benjamin Webster | Implemented Decision Tree Class, Main.py
- Gregory Ziter-Glass | Implemented Random Forest Class, Hyper-parameter tuning

### Kaggle 
Balanced Accuracy Score: 75.10%

### File Manifest
- DecisionTree.py
  - Contains decision trees functions
  - Contains node class
  - Implements entropy, gini, misclassification, chi-square pruning, and information gain
- Main
  - Loads dataset
  - Trains and evaluates the Decision Tree
  - Includes optional Kaggle submission mode
- randomForest.py
  - Logic for creating multiple trees for a random forest

[//]: # (### Experiments)

[//]: # (- Compare and contrast the trees built by IG with entropy, gini index, )

[//]: # (and misclassification error, in terms of their structural properties and )

[//]: # (accuracy. Include for example maximum depth, average depth, average accuracy, etc)

[//]: # (- Use chi-square as a termination rule with $\alpha = 0.01, 0.05, 0.1, 0.25, 0.5, 0.9$ compare and contrast the resulting trees)

[//]: # ()
[//]: # ()
[//]: # (### Need to implement from scratch:)

[//]: # ()
[//]: # (- Implement decision tree with information gain)

[//]: # (- Implement a random forest)

[//]: # (- Information Gain)

[//]: # (  - entropy)

[//]: # (  - gini index)

[//]: # (  - Misclassifcation error)

[//]: # ()
[//]: # (### Can use: )

[//]: # ()
[//]: # (- pandas to deal with the data)

[//]: # (- `train_test_split` or other data split methods from `sklearn.model_selection`)

[//]: # (- any visualization tool or analysis library to obtain insights from your results)

[//]: # (- `scipy.stats.chi2.ppf`)

[//]: # (- basic numpy or scipy &#40;but not for the functions that you need to code from scratch&#41;)

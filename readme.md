# Project 1: Random Forest

## How to run

```angular2html
To change parameters please manually change anything of interest via lines 127-134 or 148-157 
depending on if you are running a single tree or forest respectively. 

To pick between a tree or forest please change the boolean on line 118.

After picking paramters simply run the program and it will provide a submission.csv file
based on the sample_submission.csv format.
```


### Contributors

- Jesus | Implemented Decision Tree Class, Node Class
- Carly Salazar | Implemented Decision Tree Class, Edited hyper-parameters
- Ben | Tree class, Chi-square implementation, Optmization, Dealt with missing Data, Main class set up
- Greg | Random forrest implrmentation, Hyper-parameter setting

### Kaggle 
Kaggle score etc

### File Manifest
- DecisionTree.py
  - Contains decision trees functions
  - Contains node class
  - Implements entropy, gini, misclassification, chi-square pruning, and information gain
- Main
  - Loads dataset
  - Trains and evaluates the Decision Tree
  - Includes optional Kaggle submission mode
  

### Experiments
- Compare and contrast the trees built by IG with entropy, gini index, 
and misclassification error, in terms of their structural properties and 
accuracy. Include for example maximum depth, average depth, average accuracy, etc
- Use chi-square as a termination rule with $\alpha = 0.01, 0.05, 0.1, 0.25, 0.5, 0.9$ compare and contrast the resulting trees


### Need to implement from scratch:

- Implement decision tree with information gain
- Implement a random forest
- Information Gain
  - entropy
  - gini index
  - Misclassifcation error

### Can use: 

- pandas to deal with the data
- `train_test_split` or other data split methods from `sklearn.model_selection`
- any visualization tool or analysis library to obtain insights from your results
- `scipy.stats.chi2.ppf`
- basic numpy or scipy (but not for the functions that you need to code from scratch)

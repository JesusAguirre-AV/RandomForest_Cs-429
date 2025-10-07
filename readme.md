# Project 1: Random Forest

## How to run

```angular2html
Running on its own will:
-Prompts to select between creating a random forrest or single decision tree
-Load train.csv
-Split into training, validation, and test sets
-Build and evaluate a single decision tree or to a random forrest depending on selection
-Results are printed to the console


To train on all of train.csv and make predictions for test.csv, run:
'python main.py --submit' or give paramter of --submit

This will: 
-Train on the entire training dataset (no validation split)
-Load test.csv
-Read sample_submission.csv for the required output format
-Write predictions to a new file: submission.csv

To run with other parameters follow this layout: 
'python main.py --data_csv my_train.csv --test_csv my_test.csv --sample_sub my_sample.csv"

Where,

Argument	Default	                Description
--submit	(off)	                Enables Kaggle submission mode(generates csv file and runs with test.csv)
--data_csv	train.csv	        Training data (with labels)
--test_csv	test.csv	        Unlabeled test data
--sample_sub	sample_submission.csv	Sample submission file for Kaggle


```


### Contributors

- Jesus | Implemented Decision Tree Class, Node Class
- Carly Salazar | Implemented Decision Tree Class, Edited hyper-parameters
- Ben | Chi-square implementation, main class set up
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

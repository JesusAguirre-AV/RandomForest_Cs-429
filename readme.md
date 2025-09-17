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

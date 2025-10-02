import pandas as pd
import numpy as np
import math
from scipy.stats import chi2

class decisionTree:
    def __init__(self, D):
        """
        Initialize the decision tree with dataset.
        """
        self.D = D

    '''
    MAIN DECISION TREE CREATION FUNCTIONS
    - Used to recursively create the tree
    - Used to classify the data given a vector x (features)
    '''

    def DT_construction(self):
        """
        This constructs the decision tree and returns the head node of the tree.
        """
        return self.DT_construct(self.D)

    def DT_construct(self, D):
        """
        Recursively construct tree.
        Parameters:
            D - dataset
        Return: the head node of the tree.
        """
        t = self.newNode()
        t.label = self.representativeClass(D)
        # Is the current dataset under the same label
        if self.impure(D):
            criterion = self.splitCriterion(D)
            # If there is not a good stopping criterion
            if criterion is None:
                return t

            t.criterion = criterion
            # Category splits
            if isinstance(criterion, str):
                t.split_values = D[criterion].unique()
        else:
            return t
        Ds = self.decompose(D, criterion)
        for d in Ds:
            # We make the next branches here
            self.addSuccessor(t, self.DT_construct(d))
        return t

    def DT_classify(self, x, t):
        """
        New instances and classification.
        Parameters:
            x - row of data to be classified
            t - Root(?) node of tree
        Return: Classfies each row of data as fraud or not
        """
        if self.isLeaf(t):
            return t.label
        else:
            # todo fix split successor
            return self.DT_classify(x, self.splitSuccessor(t, x))


    '''
    SPLITTING FUNCTIONS
    - Used for determining how to split the data into different classifiers
    - Decomposes the data into different classifiers
    - Split the data into the different groups
    '''

    def splitCriterion(self, dSet):
        """
        Split Criterion via information gain (with entropy), gini index, and chi square.
        The feature is categorical if it is a string.
        Feature is numerical if it is a number.
        They will vary based on this
        TODO: Take into consideration missing data!!! This assumes there are no missing values.
        Todo: add chi square at the end for prepruning or something
        Parameters:
            dSet - dataset
        Returns: The feature that produces the best split criterion with information gain.
        """
        # Used to determine which feature produces the most information gain
        maxInfoGain = -1
        bestCriterionFeature = None
        bestSplitValue = None

        # Only the features, does not include ID or class declaration(Y)
        # features = dSet.drop(columns=["isFraud", "TransactionID"])
        features = dSet.drop(columns=["isFraud"])

        for feature_name in features.columns:
            # Handle Categorical Data
            if pd.api.types.is_string_dtype(features[feature_name]):
                dSetSubsets = self.decompose(dSet, feature_name)

                # Ensure that subsets are more than size 1
                if len(dSetSubsets) <= 1:
                    continue

                infoGain = self.informationGain(dSet, dSetSubsets)
                # keep track of the best information gain
                if infoGain > maxInfoGain:
                    maxInfoGain = infoGain
                    bestCriterionFeature = feature_name

            # Handle Numeric Data
            else:
                # Unique values to split between
                uniqueNumericVals = np.sort(dSet[feature_name].unique())

                for i in range(len(uniqueNumericVals) - 1):
                    # value between each unique value, for splitting
                    splitVal = (uniqueNumericVals[i] + uniqueNumericVals[i+1]) / 2

                    # Subset left of the split value (values less than the split value)
                    leftSubSet = dSet[dSet[feature_name] <= splitVal]
                    # Subset right of the split value (values more than the split value)
                    rightSubSet = dSet[dSet[feature_name] > splitVal]

                    infoGain = self.informationGain(dSet, [leftSubSet, rightSubSet])
                    # keep track of the best information gain
                    if infoGain > maxInfoGain:
                        maxInfoGain = infoGain
                        bestSplitVal = splitVal
                        bestCriterionFeature = (feature_name, bestSplitVal)

        # todo add chi square at the end, i think?
        return bestCriterionFeature

    def decompose(self, dSet, criterion):
        """
        Splits up the data into smaller subsets of data based on criterion.
        Parameters:
            dSet - dataset
            criterion - either numerical (featureName, dataValSplit) or categorical (featureName)
        Returns: A list of subsets of the dataset based on the criterion.
        """
        # Empty subset of decomposed D sets, subsets of D
        dSetSubsets = []

        # Criteria is Categorical, only decomposing by feature
        if isinstance(criterion, str):
            split_column = criterion

            # Create a subset for each unique value in the column
            for value in dSet[split_column].unique():
                subset = dSet[dSet[split_column] == value]
                dSetSubsets.append(subset)

        # Criteria is Numerical, contains feature name and data split
        elif isinstance(criterion, tuple):
            split_column, threshold = criterion

            # Create the "left" subset (less than or equal to the threshold)
            subset_left = dSet[dSet[split_column] <= threshold]
            dSetSubsets.append(subset_left)

            # Create the "right" subset (greater than the threshold)
            subset_right = dSet[dSet[split_column] > threshold]
            dSetSubsets.append(subset_right)
        return dSetSubsets

    def splitSuccessor(self, node, x):
        """
        Finds the correct node for data point
        Parameters:
            node - node
            dataRow - A row of data that needs to be classified
        Returns: A node for data point to visit
        """
        criterion = node.criterion

        # Handle categorical features
        if isinstance(criterion, str):
            feature_name, threshold = criterion
            # If the data point's value is less than or equal, go to the first child (left)
            if x[feature_name] <= threshold:
                return node.successors[0]
            # else go to the second child (right)
            else:
                return node.successors[1]

        # Handle numerical features
        if isinstance(criterion, tuple):
            feature_name = criterion
            feature_value = x[feature_name]

            try:
                # Find the index of the data point's category in the list we saved on the node
                # For example, if split_values is ['W', 'H', 'C'] and the value is 'H', the index is 1
                idx = np.where(node.split_values == feature_value)[0][0]
                # Return the child node at that same index
                return node.successors[idx]
            except IndexError:
                return node.successors[0]


    '''
    IMPURITY/INFO GAIN FUNCTIONS
    - Used to describe the impurity and represention of each class
    - Used to calculate impurity of the data via Gini Index, entropy, and misclassification error
    - Information gain is used to determine how well a feature splits the data
    - Chi-square is used as a termination criterion
    '''

    def impure(self, dSet):
        """
        If all the sets have the same class then they should pass the following loop and return pure.
        If there is only one unique value in the dataset, the data is pure. If there are more values, then
        the data is not pure.
        Parameters:
            dSet - dataset
        Return: If dataset is impure (True) or pure (False)
        """
        # Number of unique values in the dataset
        numUnique = dSet['isFraud'].nunique()
        if numUnique == 1: return False
        elif numUnique >= 2: return True

    def representativeClass(self, dSet):
        """
        The majority class (isFraud, !isFraud), the class that is >0.50
        Parameters:
            dSet - dataset
        Returns: The class that is the most representative of the given dataset.
        """
        # The majority class, (isFraud = 0, !isFraud = 1)
        majorityClass = dSet["isFraud"].mode()[0]
        return majorityClass

    def chiSquare(self, dSet, criterion, alpha=0.5):
        """
        Used for termination criteria.
        Parameters:
            dSet - dataset
        Returns: something?
        """
        # todo: implement the chi square thing correctly
        return NotImplemented

    def entropy(self, dSet):
        """
        Measures impurity using entropy.
        0 -> The most pure
        1 -> The least pure
        Entropy = - SUM(p_i log2 (p_i))
        Parameters:
            dSet - dataset
        Returns: A impurity rating, [0,1]
        """
        probs = self.probabilityList(dSet)
        return -self.sumF(lambda x: x * (math.log2(x)) if x > 0 else 0, probs)

    def giniIdex(self, dSet):
        """
        Measures impurity with gini index
        Parameters:
            dSet - dataset
        Returns: A gini index, [0,1]
        """
        probs = self.probabilityList(dSet)
        return 1 - (self.sumF(lambda x: x ** 2, probs))

    def missclassicationError(self, dSet):
        """
        A method of determining impurity
        Parameters:
            dSet - dataset
        Returns: A misclassification error, [0,1]
        todo: implement this please
        """
        return NotImplemented

    def informationGain(self, dSet, dSubSets):
        """
        A metric that determines how much information was gain based on split.
        Can use gini index or entropy for impurity.
        information gain = impurity(dSet) - SUM ((subset_i Samples/ Total Samples) * impurity(dSubSet_i))
        Parameters:
            dSet - dataset
        Returns: The amount of information gained, [0,1]
        TODO: handle all three types of impurity
        """
        data_entropy = self.entropy(dSet)
        weightedSubsetEntopy = 0

        totalSamples = len(dSet)

        for subset in dSubSets:
            subset_samples = len(subset)
            if subset_samples == 0:
                continue

            subset_entropy = self.entropy(subset)
            proportion = subset_samples / totalSamples
            weightedSubsetEntopy += (proportion * subset_entropy)

        infoGain = data_entropy - weightedSubsetEntopy
        return infoGain

    def sumF(self, function, list):
        """
        Passing an argument to sum the results
        Parameters:
            function - a function
            list - a list
        Returns: The sum of the result
        """
        sum = 0
        for l in list:
            sum += function(l)
        return sum

    def probabilityList(self, dSet):
        """
        Calculates the probability of each class
        """
        probs = []

        classTarget = dSet["isFraud"]
        total = len(classTarget)
        classCounts = classTarget.value_counts()

        if total == 0:
            return []

        # Find the probability for
        for count in classCounts:
            prob = count/total
            probs.append(prob)
        return probs


    '''
    NODE/TREE FUNCTIONS
    - creates new nodes
    - add a successor node
    - determines if leaf
    '''

    def newNode(self):
        """
        Creates a new node for decision tree.
        """
        newT = Node()
        return newT

    def addSuccessor(self, n, s):
        """
        Adding a successor/child node to the current node.
        Parameters:
            n - current node
            s - successor node
        """
        n.successors.append(s)

    def isLeaf(self, n):
        """
        Determine if the node is leaf.
        Parameters:
            n - current node
        Return: Return if node is leaf (True) or not (False)
        """
        return not (not n.successors)

class Node:
    """
    Class that contains the information about a node.
    """
    # todo: This is the node class that contains its successors (cosider the information gain and purity as well)
    def __init__(self):
        """
        Class for the nodes of a decision tree.
        """
        self.successors = []
        self.label = ""
        self.prob = 0
        self.criterion = None
        self.split_values = None
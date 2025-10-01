import pandas as pd
import numpy as np
from sklearn.metrics.cluster import entropy

from main import criterion

"Here, we write the functions that were mentioned in the slides"
import math
"import pandas as r"

"sample = r.read_csv('sample_sub.csv')"
"testD = r.read_csv('test.csv')"
"trainD = r.read_csv('train.csv')"

"D = one of the data sets"
class decisionTree:
    def __init__(self, D):
        """
        Initialize the decision tree with dataset.
        """
        self.D = D

    ##### MAIN DECISION TREE CREATION FUNCTIONS #####

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
        TODO: work on this one
        """
        t = self.newNode()
        t.label = self.representativeClass(D)
        # Is the current dataset under the same label
        if self.impure(D):
            criterion = self.splitCriterion(D)
            t.criterion = criterion
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


    ### SPLITTING FUNCTIONS ###

    def splitCriterion(self, dSet):
        """
        Split Criterion via information gain (with entropy), gini index, and chi square.
        The feature is categorical if it is a string.
        Feature is numerical if it is a number.
        They will vary based on this
        TODO: Take into consideration missing data!!! This assumes there are no missing values.
        Parameters:
            dSet - dataset
        Returns: The feature that produces the best split criterion with information gain.
        """
        # Used to determine which feature produces the most information gain
        maxInfoGain = -1
        bestCriterionFeature = None
        bestSplitValue = None

        # Only the features, does not include ID or class declaration(Y)
        features = dSet.drop(columns=["isFraud", "TransactionID"])

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
                    rightSubSet = dSet[dSet[feature_name] >= splitVal]

                    infoGain = self.informationGain(dSet, [leftSubSet, rightSubSet])
                    # keep track of the best information gain
                    if infoGain > maxInfoGain:
                        maxInfoGain = infoGain
                        bestSplitVal = splitVal
                        bestCriterionFeature = (feature_name, bestSplitVal)
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
        TODO: work on this
        """
        criterion = node.criterion

        # Handle categorical features
        if isinstance(criterion, str):
            return NotImplemented

        # Handle numerical features
        if isinstance(criterion, tuple):
            return NotImplemented

        # for s in node.succesors:
        #     for t in object.traits:
        #         # TODO: Going through successors and seeing what has matching traits, how to read object
        return NotImplemented


    ##### IMPURITY/INFO GAIN FUNCTIONS #####

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
        todo: what exactly does this return and does it matter?
        Parameters:
            dSet - dataset
        Returns: The class that is the most representative of the given dataset.
        """
        # The majority class, (isFraud = 0, !isFraud = 1)
        majorityClass = dSet["isFraud"].mode()[0]
        return majorityClass

    def chiSquare(self, dSet, alpha):
        """
        Used for termination criteria.
        Parameters:
            dSet - dataset
        Returns: something?
        """
    #   todo: implement the chi square thing correctly
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
        # "sum = 0"
        # "for d in dSet:"
        # "TAB sum += prob(d) * (math.log2(prob(d)))"
        # "return -sum"

    def giniIdex(self, dSet):
        """
        Measures impurity with gini index
        Parameters:
            dSet - dataset
        Returns: A gini index, [0,1]
        """
        probs = self.probabilityList(dSet)
        return 1 - (self.sumF(lambda x: x ** 2, probs))
        # "sum = 0"
        # "for d in dSet:"
        # "TAB sum += (prob(d) ** 2)"
        # "return (1-sum)"

    def missclassicationError(self, dSet):
        return NotImplemented

    def informationGain(self, dSet, dSubSets):
        """
        A metric that determines how much information was gain based on split.
        Can use gini index or entropy for impurity.
        information gain = impurity(dSet) - SUM ((subset_i Samples/ Total Samples) * impurity(dSubSet_i))
        Parameters:
            dSet - dataset
        Returns: The amount of information gained, [0,1]
        TODO: which impurity to use, which are the samples
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

        # Find the probability for
        for count in classCounts:
            prob = count/total
            probs.append(prob)


    ##### NODE/TREE FUNCTIONS #####

    def newNode(self):
        """
        Creates a new node for decision tree.
        """
        newT = node()
        return newT

    def addSuccessor(self, n, s):
        """
        Adding a successor/child node to the current node.
        todo: does this mean child?
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

class node:
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
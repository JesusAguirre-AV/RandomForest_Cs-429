import pandas as pd
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
            # splitCriterion is figuring out what to split it with
            # todo: fix and implement split criteria
            criterion = self.splitCriterion(D)
        else:
            return t
        Ds = self.decompose(D, criterion)
        for d in Ds:
            # We make the next branches here
            self.addSuccessor(t, self.DT_construct(d))
        return t

    def DT_classify(self, x,t):
        """
        New instances and classification.
        """
        if self.isLeaf(t):
            return t.label
        else:
            # todo: fix this
            return self.DT_classify(x, self.splitSuccessor(t, x))
        # "splitSuccsessor is choosing what direction to go down"

    def newNode(self):
        """
        Creates a new node.
        """
        newT = node()
        return newT

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
        # checker = dSet[0]
        # for d in dSet:
        #     if not classMatch(checker, d):
        #         return Truex
        # return False

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

    def splitSuccessor(self, node, object):
        """
        TODO: Electing what branch to go down"
        """
        for s in node.succesors:
            for t in object.traits:
                "TODO: Going through successors and seeing what has matching traits, how to read object"

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


    def splitCriterion(self, dSet):
        """
        Split Criterion via information gain (with entropy), gini index, and chi square.
        Parameters:
            dSet - dataset
        Returns: a split criterion or something.
        TODO: Using intelligence gain, decide what criteria to split
        """
        #all gini/info gain/chi
        return NotImplemented

    def chiSquare(self, dSet):
        """
        Used for termination criteria.
        Parameters:
            dSet - dataset
        Returns: something?
        """
    #   todo: implement the chi square thing correctly
        return NotImplemented

    def decompose(self, dSet, criterion):
        """
        TODO: assuming we make a set of sublists based on criterion, not sure
        Splits up the data into smaller subsets of data based on criterion.
        Parameters:
            dSet - dataset
            criterion - criterion (Entropy or Gini)
        Returns: A list of subsets of the dataset based on the criterion.
        """
        # Empty subset of decomposed D sets, subsets of D
        Ds = []
        # one for numerical data
        # one for categorical data
        # todo: what is the difference between a categorical and numerical split
        return Ds

    # def classMatch(dA, dB):
    #     "TODO: check to see if the classes between the two datapoints match"

    def impurity(self, dSet):
        """
        Impurity
        Parameters:
            dSet - dataset
        Returns: A impurity rating, [0,1]
        """
        return sumF(lambda x: x * (math.log2(x)), self.probabilityList(dSet))
        "sum = 0"
        "for d in dSet:"
        "TAB sum += prob(d) * (math.log2(prob(d)))"
        "return -sum"

    def giniIdex(self, dSet):
        """
        The gini index
        Parameters:
            dSet - dataset
        Returns: A gini index, [0,1]
        """
        return 1 - (sumF(lambda x: x ** 2, probabilityList(dSet)))
        "sum = 0"
        "for d in dSet:"
        "TAB sum += (prob(d) ** 2)"
        "return (1-sum)"

    def informationGain(self, dSet):
        """
        A metric that determines how much information was gain based on split.
        Parameters:
            dSet - dataset
        Returns: The amount of information gained, [0,1]
        TODO: which impurity to use, which are the samples
        """
        # todo: implement entropy
        # todo probably redo this
        result = impurity(dSet) - sumF((sWithValue(dSet)/samples(dSet)) * sumF(impurity, dWithValues(dSet)))

    def sumF(self, function, list):
        """
        passing an argument to sum the results
        """
        sum = 0
        for l in list:
            sum += function(l)
        return sum

    def probabilityList(self, dSet):
        probs = []
        for d in dSet:
            probs.append(prob(d))

class node:
    # todo: This is the node class that contains its successors (cosider the information gain and purity as well)
    def __init__(self):
        """
        Class for the nodes of a decision tree.
        """
        self.successors = []
        self.label = ""
        self.prob = 0
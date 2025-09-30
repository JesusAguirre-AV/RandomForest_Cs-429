# decisionTree.py
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
        Initialize the decision tree with dataset
        """
        self.D = D

    def DT_construction(self):
        """
        This constructs the decision tree and returns the head node of the tree.
        """
        return self.DT_construct(self.D)

    def DT_construct(self, D):
        """
        Creating tree. (Code from Notes)
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

    def DT_classify(x,t):
        """
        New instances and classification.
        """
        if isLeaf(t):
            return t.label
        else:
            return DT_classify(x, splitSuccessor(t,x))
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
        n.successors.append(s)

    def isLeaf(n):
        return not (not n.successors)

    def splitSuccessor(node, object):
        """
        TODO: Electing what branch to go down"
        """
        for s in node.succesors:
            for t in object.traits:
                "TODO: Going through successors and seeing what has matching traits, how to read object"

    def representativeClass(self, dSet):
        """
        The majority class (isFraud, !isFraud), the class that is >0.50
        TODO: Finding classification from the dataset, Data sets are read with the column name, these will be the attributes
        """
        # The majority class, (isFraud = 0, !isFraud = 1)
        majorityClass = dSet["isFraud"].mode()[0]
        return majorityClass


    def splitCriterion(self, dSet):
        """
        Split Criterion via information gain (with entropy), gini index, and chi square.
        TODO: Using intelligence gain, decide what criteria to split
        """
    #     call gini/info gain/chi

    def chiSquare(self, dSet):
        """
        A split criteria
        """



    def decompose(self, dSet, criterion):
        "TODO: assuming we make a set of sublists based on criterion, not sure"

        return {}

    def classMatch(dA, dB):
        "TODO: check to see if the classes between the two datapoints match"

    def impurity(dSet):
        """
        Impurity
        """
        return sumF(lambda x: x * (math.log2(x)), probabilityList(dSet))
        "sum = 0"
        "for d in dSet:"
        "TAB sum += prob(d) * (math.log2(prob(d)))"
        "return -sum"

    def giniIdex(dSet):
        """
        The gini index
        """
        return 1 - (sumF(lambda x: x ** 2, probabilityList(dSet)))
        "sum = 0"
        "for d in dSet:"
        "TAB sum += (prob(d) ** 2)"
        "return (1-sum)"

    def informationGain(dSet):
        """
        TODO: which impurity to use, which are the samples
        """
        result = impurity(dSet) - sumF((sWithValue(dSet)/samples(dSet)) * sumF(impurity, dWithValues(dSet)))

    def sumF(function, list):
        """
        passing an argument to sum the results
        """
        sum = 0
        for l in list:
            sum += function(l)
        return sum

    def probabilityList(dSet):
        probs = []
        for d in dSet:
            probs.append(prob(d))

"This is the node class that contains its successors (cosider the information gain and purity as well)"
class node:
    def __init__(self):
        self.successors = []
        self.label = ""
        self.prob = 0
#slideCode.py
"Here, we write the functions that were mentioned in the slides"
import math
import pandas as r

sample = r.read_csv('sample_sub.csv')
testD = r.read_csv('test.csv')
trainD = r.read_csv('train.csv')

"D = one of the data sets"

"Creating tree"
def DT_construct(D):
    t = newNode()
    t.label = representativeClass(D)
    "Is the current dataset under the same label"
    if impure(D):
        "splitCriterion is figuring out what to split it with"
        criterion = splitCriterion(D)
    else:
        return t
    Ds = decompose(D, criterion)
    for d in Ds:
        "We make the next branches here"
        addSuccessor(t, DT_construct(d))
    return t

"New instances and classification"
def DT_classify(x,t):
    if isLeaf(t):
        return t.label
    else: return DT_classify(x, splitSuccessor(t,x))
    "splitSuccsessor is choosing what direction to go down"

def newNode():
    newT = node()
    return newT

def impure(dSet):
    "If all the sets have the same class then they should pass the following loop and return pure"
    checker = dSet[0]
    for d in dSet:
        if not classMatch(checker, d):
            return True
    return False

def addSuccessor(n, s):
    n.successors.append(s)

def isLeaf(n):
    return not (not n.successors)

def splitSuccessor(node, object):
    "TODO: Electing what branch to go down"

def representativeClass(dSet):
    "TODO: Finding classification from the dataset, Data sets are read with the column name, these will be the atributes"

def splitCriterion(dSet):
    "TODO: Using intelligence gain, decide what criteria to split"

def decompose(dSet, c):
    "TODO: assuming we make a set of sublists based on criterion, not sure"

def classMatch(dA, dB):
    "TODO: check to see if the classes between the two datapoints match"

def impurity(dSet):
    sum = 0
    for d in dSet:
        sum += prob(d) * (math.log2(prob(d)))
    return -sum

def giniImpurity(dSet):
    sum = 0
    for d in dSet:
        sum += (prob(d) ** 2)
    return (1-sum)

def informationGain():
    "TODO: which impurity to use, which are the samples"

"This is the node class that contains its successors (cosider the information gain and purity as well)"
class node:
    def __init__(self):
        self.successors = []
        self.label = ""
        self.prob = 0
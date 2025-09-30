#forestCode.py
# import slideCode
import pandas as r
import numpy
from scipy import stats


sample = r.read_csv('sample_sub.csv')
testD = r.read_csv('test.csv')
trainD = r.read_csv('train.csv')

"D = one of the data sets"

trees=[]

def createForest(v):
    "TODO: create a forest based on what may be required"
    for i in range(v):
        trees.append(slideCode.DT_construct(d))

def searchForest(o):
    "TODO: Run the forrest and find the mode of the results and return it."
    resultVotes=[]
    for tree in trees:
        resultVotes.append(slideCode.DT_calssify(o,tree))
    npA = numpy.array(resultVotes)
    return stats.mode(npA).mode

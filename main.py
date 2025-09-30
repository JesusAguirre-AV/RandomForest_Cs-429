import decisionTree as DT
import pandas as pd

training_data = pd.read_csv('train.csv')
print(training_data)

# testingggg!!


d = [1,1,1,1,1,1]
# d = [0]
df = pd.DataFrame(d, columns=['womp'])
modeww = df['womp'].mode()[0]
modeww = df['womp'].nunique()

print("unique : ")
print(modeww)

y = training_data["isFraud"].mode()[0]

print("moajority 0: \n")
print(y)


# create decision tree

tree = DT.decisionTree(training_data)
head_node = tree.DT_construction()
# import decisionTree as DT
import pandas as pd

training_data = pd.read_csv('train.csv')
# print(training_data)

# testingggg!!


# d = [1,1,1,1,1,1]
# # d = [0]
# df = pd.DataFrame(d, columns=['womp'])
# modeww = df['womp'].mode()[0]
# modeww = df['womp'].nunique()
#
# print("unique : ")
# print(modeww)
#
# y = training_data["isFraud"].mode()[0]
#
# print("moajority 0: \n")
# print(y)

features = training_data.drop(columns=["isFraud", "TransactionID"])

# for feature_name in features.columns:
#     # val = features[feature_name]
#     # if pd.api.types.is_numeric_dtype(features[feature_name]):
#     #     print(f"{feature_name}: numeric")
#
#     if pd.api.types.is_string_dtype(features[feature_name]):
#         print(f"{feature_name}: categorical")
#     else:
#         print(f"{feature_name}: numeric")

criterion = "C14"

if isinstance(criterion, str):
    print(f"{criterion}: is a string")
else:
    print(f"{criterion}: is not a string")

mask = training_data.isin(['NotFound']).any(axis=1)
training_cleaned = training_data[~mask]

pd.set_option('display.max_columns', None)

# print(training_data)
# print("clean data")
# print(training_cleaned)



#
# for feature in features.columns:
#     # print(feature)
#     val = (features[feature].mode() & features[feature].mode() != "NotFound")
#     print(f"{feature} mode: {val}")




# print(features)


# create decision tree

# tree = DT.decisionTree(training_data)
# head_node = tree.DT_construction()
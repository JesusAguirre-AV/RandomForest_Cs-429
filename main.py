import decisionTree as DT
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, balanced_accuracy_score

'''
##### FLAGS #####
Impurity Options
Gini - G
Entropy - E
Misclassification - M
'''
# Set impurity method
impurity_type = 'G'
# set alpha level
alpha = 0.1
# submission file name
out_file = 'submission.csv'

'''
Training Data
- uses training/testing validation splitting
'''

training_data = pd.read_csv('train.csv')
# Temp data which removes rows with missing data
# TODO: change this since it needs to have missing data
mask = training_data.isin(['NotFound']).any(axis=1)
training_cleaned = training_data[~mask]
# full_data = training_cleaned
full_data = training_data

# Test set only using a few features
# features_to_use = ['TransactionID', 'ProductCD', 'card1', 'card4', 'card6', 'addr1', 'isFraud']
features_to_use = ['TransactionID', 'TransactionAmt', 'addr1', 'isFraud']
data = full_data[features_to_use].copy()
# data = full_data.copy()


# Handle missing values
for col in data.columns:
    if pd.api.types.is_numeric_dtype(data[col]):
        # Fill missing numbers with the median
        data[col] = data[col].fillna(data[col].median())
    else:
        # Fill missing strings/categories with the mode
        data[col] = data[col].fillna(data[col].mode()[0])

# Separate features (x) from the target (y)
y = data['isFraud']
x = data.drop(columns=['isFraud', 'TransactionID'])

# Training and testing the data with the training dataset
# Testing data = 0.20, Validation = 0.25, Training = 0.55
# todo: use the validation sets for something
X_train_val, X_test, y_train_val, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

# Full training data, including isFraud Category
train_df = pd.concat([X_train, y_train], axis=1)

print(f"Yay training! Training with {len(train_df)} samples.")

# create tree object
tree = DT.decisionTree(train_df)

# construct the decision tree
print("Building the decision tree...")
root_node = tree.DT_construction()
print("Decision tree has been built!!")

predictions = []
predictions_results = []
print(f"Making predictions on {len(X_test)} test samples...")

# Loop through each row in the test set
for index, row in X_test.iterrows():
    # yay classify
    prediction = tree.DT_classify(row, root_node)
    predictions.append(prediction)

    # used for csv creation
    predictions_results.append({
        'TransactionID': index,
        'isFraud': prediction
    })

bal_accuracy = balanced_accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions)

print("\n--- Model Performance ---")
print(f"Accuracy: {bal_accuracy:.4f}")
print("\nClassification Report:")
print(report)

# Create a submission CSV
submission_df = pd.DataFrame(predictions_results)
submission_df.to_csv(
    out_file, # 'submission.csv'
    index=False,
    columns=['TransactionID', 'isFraud']
)
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv("drug.csv")

print(dataset.isnull().sum())
dataset.dropna(inplace=True)

print(dataset.dtypes)

lbl_en = LabelEncoder()
dataset.Sex = lbl_en.fit_transform(dataset.Sex)
dataset.BP = lbl_en.fit_transform(dataset.BP)
dataset.Cholesterol = lbl_en.fit_transform(dataset.Cholesterol)
dataset.Drug = lbl_en.fit_transform(dataset.Drug)
print(dataset.dtypes)

# First experiment: Training and Testing with Fixed Train-Test Split Ratio
target_feature = dataset[["Drug"]]
features = dataset[['Age', 'Sex', 'BP', 'Cholesterol']]

best_acc = 0
best_model = None
best_exp = 0

for i in range(5):
    X_train, X_test, y_train, y_test = train_test_split(features, target_feature, test_size=0.3, random_state=i)

    model = DecisionTreeClassifier(criterion="entropy")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    tree_size = model.tree_.node_count
    print(f"Experiment {i + 1}: Tree Size = {tree_size}, Accuracy = {accuracy}")

    if accuracy > best_acc:
        best_acc = accuracy
        best_model = model
        best_exp = i + 1

print(f"The Best Model is Experiment no.{best_exp} with accuracy: {best_acc}")


# Second Experiment: Training and Testing with a Range of Train-Test Split Ratios
train_sizes = np.arange(0.3, 0.8, 0.1)

mean_accuracies = []
max_accuracies = []
min_accuracies = []
mean_tree_sizes = []
max_tree_sizes = []
min_tree_sizes = []

for train_size in train_sizes:
    accuracies = []
    tree_sizes = []

    for i in range(5):
        X_train, X_test, y_train, y_test = train_test_split(
            features, target_feature, test_size=1 - train_size, random_state=i
        )

        model = DecisionTreeClassifier(criterion="entropy")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        tree_size = model.tree_.node_count

        accuracies.append(accuracy)
        tree_sizes.append(tree_size)

    mean_accuracies.append(np.mean(accuracies))
    max_accuracies.append(np.max(accuracies))
    min_accuracies.append(np.min(accuracies))
    mean_tree_sizes.append(np.mean(tree_sizes))
    max_tree_sizes.append(np.max(tree_sizes))
    min_tree_sizes.append(np.min(tree_sizes))

# Report
report = pd.DataFrame({
    'Train Size': train_sizes,
    'Mean Accuracy': mean_accuracies,
    'Max Accuracy': max_accuracies,
    'Min Accuracy': min_accuracies,
    'Mean Tree Size': mean_tree_sizes,
    'Max Tree Size': max_tree_sizes,
    'Min Tree Size': min_tree_sizes
})

print(report)

# Plotting
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(train_sizes, mean_accuracies, label='Mean Accuracy')
plt.fill_between(train_sizes, min_accuracies, max_accuracies, alpha=0.3, label='Accuracy Range')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_sizes, mean_tree_sizes, label='Mean Tree Size')
plt.fill_between(train_sizes, min_tree_sizes, max_tree_sizes, alpha=0.3, label='Tree Size Range')
plt.xlabel('Training Set Size')
plt.ylabel('Number of Nodes')
plt.legend()

plt.tight_layout()
plt.show()

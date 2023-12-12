# ML-DecisionTree-Algorithm
[Decision Trees using Scikit-learn][2 experiments]:
The objective is to build a model that can predict the appropriate medication/Drug type
(target variable) for patients suffering from a specific illness based on their Age, Sex, Blood
Pressure, and Cholesterol levels. [Drag.csv]
Assignment Tasks:

1. Data Preprocessing: Perform data preprocessing steps by handling missing data [count how
many missing values occurs + handle empty cell by your own way] and encoding categorical
variables.

2. First experiment: Training and Testing with Fixed Train-Test Split Ratio:
 
    • Divide the dataset into a training set and a testing set (30% of the samples).

    • Repeat this experiment five times with different random splits of the data into training
and test sets.

    • Report the sizes and accuracies of the decision trees in each experiment, Compare the
results of different models and select the one that achieves the highest overall
performance.

3. Second experiment: Training and Testing with a Range of Train-Test Split Ratios:
Consider training set sizes in the range of 30% to 70% (increments of 10%). Start with a
training set size of 30% and increase it by 10% until you reach 70%.
For each training set size:

    • Run the experiment with five different random seeds.

    • Calculate the mean, maximum, and minimum accuracy at each training set size.

    • Measure the mean, maximum, and minimum tree size.

    • Store the statistics in a report.

    • Create two plots: one showing accuracy against training set size and another showing
the number of nodes in the final tree against training set size.

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# Define the number of models
num_models = 6

# Create arrays for storing test and combined accuracy for each model
test_accuracy_percent = np.zeros(num_models)
combined_accuracy_percent = np.zeros(num_models)

# Read the dataset
df = pd.read_csv('heart1.csv')
df_numpy = df.to_numpy()

# Separate the features and the target variable
X = df_numpy[:, 0:13]
y = df_numpy[:, 13]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=18)

def train_and_evaluate_classifier(classifier, normalize=False):
    """
    Train and evaluate a classifier.
    
    Args:
        classifier: The classifier to train and evaluate.
        normalize (bool): Whether to normalize the features.
    
    Returns:
        test_acc_per (float): Test accuracy percentage.
        combined_acc_per (float): Combined accuracy percentage.
    """
    if normalize:
        # Normalize the features if needed
        sc = StandardScaler()
        X_train_sc = sc.fit_transform(X_train)
        X_test_sc = sc.transform(X_test)
    else:
        X_train_sc = X_train
        X_test_sc = X_test
    
    # Fit the classifier on the training data
    classifier.fit(X_train_sc, y_train)
    
    # Combine the training and testing data
    X_combined = np.vstack((X_train_sc, X_test_sc))
    y_combined = np.hstack((y_train, y_test))
    
    # Predict the target variable for the testing data
    y_pred = classifier.predict(X_test_sc)
    
    # Calculate the test accuracy
    test_acc = accuracy_score(y_test, y_pred)
    
    # Predict the target variable for the combined data
    y_combined_pred = classifier.predict(X_combined)
    
    # Calculate the combined accuracy
    combined_acc = accuracy_score(y_combined, y_combined_pred)
    
    # Round the accuracies to 2 decimal places
    test_acc_per = round(test_acc * 100, 2)
    combined_acc_per = round(combined_acc * 100, 2)
    
    return test_acc_per, combined_acc_per

# Initialize classifiers with their respective parameters
classifiers = [
    Perceptron(max_iter=15, tol=1e-3, eta0=0.001, fit_intercept=True, random_state=18, verbose=False),
    LogisticRegression(C=10, solver='liblinear', multi_class='ovr', random_state=18),
    SVC(kernel='linear', C=1, random_state=18),
    DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=18),
    RandomForestClassifier(criterion='entropy', n_estimators=10, random_state=18, n_jobs=1),
    KNeighborsClassifier(n_neighbors=1, p=1, metric='minkowski')
]

# Train and evaluate each classifier
for i, classifier in enumerate(classifiers):
    test_accuracy_percent[i], combined_accuracy_percent[i] = train_and_evaluate_classifier(classifier, True)

# Create a pandas dataframe for the accuracy table
accuracy_table = pd.DataFrame(
    list(zip(test_accuracy_percent, combined_accuracy_percent)),
    index=['Perceptron', 'Logistic Regression', 'Support Vector', 'Decision Tree', 'Random Forest', 'K Nearest Neighbor'],
    columns=['Test Accuracy in %', 'Combined Accuracy in %']
)

# Display the accuracy information in a different plain text format
print("Accuracy Information\n")
for classifier, test_acc, combined_acc in zip(accuracy_table.index, accuracy_table['Test Accuracy in %'], accuracy_table['Combined Accuracy in %']):
    print(f"Classifier: {classifier}")
    print(f"Test Accuracy: {test_acc}%")
    print(f"Combined Accuracy: {combined_acc}%")
    print("\n")

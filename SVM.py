import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from src.kernels.rbf import RBF
from src.svm import SVM

# Prepare read buffer
file_path = 'wdbc.data'
inputs = []
labels = []

# Read data from file
with open(file_path, 'r') as file:
    for line in file:
        elements = line.strip().split(',')
        diagnosis = elements[1]
        input = list(map(float, elements[2:]))

        if diagnosis == 'M':
            label = 1
        elif diagnosis== 'B':
            label = 0

        inputs.append(input)
        labels.append(label)

# Remap data #1 - vectorize
inputs = np.array(inputs)
labels = np.array(labels)

def normalizeInputs(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    normalized_X = (X - mean) / std
    return normalized_X

def normalizeLabels(y):
    return np.where(y == 1, 1, -1)

# Remap data #2 - normalize
inputs = normalizeInputs(inputs)
labels = normalizeLabels(labels)

# Split data
X_train, X_test, y_train, y_test = train_test_split(inputs, labels, test_size=0.5, random_state=42)

# Train & repeat
for gamma in [1e+0, 2e+0, 5e+0, 1e+1, 2e+1, 5e+1]:
    for coherence in [1e-1, 2e-1, 5e-1, 1e+0, 2e+0, 5e+0]:
        for tolerance in [1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4]:
            svm = SVM(RBF(gamma), coherence, tolerance)
            svm.train(X_train, y_train)

            prediction = svm.predict(X_test)

            print(f"gamma: {gamma} | coherence: {coherence} | tolerance: {tolerance}")
            print(f"Accuracy: {accuracy_score(prediction, y_test)}")
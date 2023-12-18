import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from src.kernels.rfb import RFB
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
for gamma in [0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 5.0]:
    for coherence in [0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 5.0]:
        svm = SVM(RFB(gamma), coherence)
        svm.train(X_train, y_train)

        prediction = svm.predict(X_test)

        print("gamma: {} | coherence: {}".format(gamma, coherence))
        print("Accuracy:", accuracy_score(prediction, y_test))
import numpy as np

# Wczytywanie danych z pliku .data
data_file_path = 'wdbc.data'
data = []
labels = []


with open(data_file_path, 'r') as file:
    for line in file:
        elements = line.strip().split(',')
        diagnosis = elements[1]
        features = list(map(float, elements[2:]))
        if diagnosis == 'M':
            label = 1
        elif diagnosis== 'B':
            label = 0

        data.append(features)
        labels.append(label)

X = np.array(data)
y = np.array(labels).reshape(-1, 1)

# Normalizacja danych
def normalize_data(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    normalized_X = (X - mean) / std
    return normalized_X

X = normalize_data(X)

# Podział danych na zbiór treningowy, testowy i walidacyjny
def custom_train_test_val_split(X, y, test_size=0.2, val_size=0.3):
    np.random.seed(42)

    # Tasowanie indeksów
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    # Podział na zbiór treningowy, testowy i walidacyjny
    test_start = int(len(X) * (1 - test_size))
    val_start = int(test_start - len(X) * val_size)

    train_indices = indices[:val_start]
    val_indices = indices[val_start:test_start]
    test_indices = indices[test_start:]

    X_train, X_val, X_test = X[train_indices], X[val_indices], X[test_indices]
    y_train, y_val, y_test = y[train_indices], y[val_indices], y[test_indices]

    return X_train, X_val, X_test, y_train, y_val, y_test

X_train, X_val, X_test, y_train, y_val, y_test = custom_train_test_val_split(X, y)

# Implementacja MLP
class MLP:
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        self.weights_input_hidden1 = np.random.rand(input_size, hidden_size1) # inicjalizacja wag dla połączenia warstwy wejściowej i ukrytej
        self.weights_hidden1_hidden2 = np.random.rand(hidden_size1, hidden_size2) # inicjalizacja wag dla połączenia warstwy ukrytej 1 i ukrytej 2
        self.weights_hidden2_output = np.random.rand(hidden_size2, output_size)  # inicjalizacja wag dla połączenia warstwy ukrytej 1 i ukrytej 2

    # funkcja aktywująca neurony
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # pochodna funkcji sigmoidalnej, odpowiadająca za propagację wsteczną - aktualizuje wagi
    def sigmoid_derivative(self, x):
        return x * (1 - x)

    # def relu_activation_function(self, x):
    #         return np.maximum(x,0)
    #
    # def relu_derivative(self,x):
    #         return np.where(x > 0, 1, 0)


    def train(self, X, y, epochs=1000, learning_rate=0.1):
        for epoch in range(epochs):
            # Forward pass i obliczanie przewidywanego wejścia
            hidden1_layer_output = self.sigmoid(np.dot(X, self.weights_input_hidden1))
            hidden2_layer_output = self.sigmoid(np.dot(hidden1_layer_output,self.weights_hidden1_hidden2))
            predicted_output = self.sigmoid(np.dot(hidden2_layer_output, self.weights_hidden2_output))

            # propagacja wsteczna, aktualizowanie wag
            output_delta = (y - predicted_output) * self.sigmoid_derivative(predicted_output)
            hidden2_layer_delta = output_delta.dot(self.weights_hidden2_output.T) * self.sigmoid_derivative(hidden2_layer_output)
            hidden1_layer_delta = hidden2_layer_delta.dot(self.weights_hidden1_hidden2.T) * self.sigmoid_derivative(hidden1_layer_output)


            # aktualizacja wag
            self.weights_hidden2_output += hidden2_layer_output.T.dot(output_delta) * learning_rate
            self.weights_hidden1_hidden2 += hidden1_layer_output.T.dot(hidden2_layer_delta) * learning_rate
            self.weights_input_hidden1 += X.T.dot(hidden1_layer_delta) * learning_rate

    def predict(self, X):
        hidden1_layer_output = self.sigmoid(np.dot(X, self.weights_input_hidden1))
        hidden2_layer_output = self.sigmoid(np.dot(hidden1_layer_output,self.weights_hidden1_hidden2))
        predicted_output = self.sigmoid(np.dot(hidden2_layer_output, self.weights_hidden2_output))
        return predicted_output

# Dobór optymalnych hiperparametrów za pomocą walidacji krzyżowej
def cross_validation(X, y, hidden_size1, hidden_size2, learning_rate, epochs, n_splits=10):
    np.random.seed(7)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    fold_size = len(X) // n_splits
    accuracies = []

    for i in range(n_splits):
        test_indices = np.arange(i * fold_size, (i + 1) * fold_size)
        train_indices = np.concatenate([np.arange(0, i * fold_size), np.arange((i + 1) * fold_size, len(X))])

        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        mlp = MLP(X_train.shape[1], hidden_size1, hidden_size2, 1)
        mlp.train(X_train, y_train, epochs=epochs, learning_rate=learning_rate)

        predictions = mlp.predict(X_test)
        accuracy = np.mean((predictions > 0.5) == y_test)
        accuracies.append(accuracy)

    return np.mean(accuracies)

# Przeszukiwanie przestrzeni hiperparametrów
best_accuracy = 0
best_hidden_size1 = 0
best_hidden_size2 = 0
best_learning_rate = 0
best_epochs = 0

for hidden_size1 in [5, 50]:
    for hidden_size2 in [10, 100]:
        for learning_rate in [0.01, 0.5]:
            for epochs in [500, 1000, 1500]:
                accuracy = cross_validation(X_val, y_val, hidden_size1, hidden_size2, learning_rate, epochs)
                print(f"Hidden Size 2: {hidden_size2}, Hidden Size 1: {hidden_size1}, Learning Rate: {learning_rate}, Epochs: {epochs}, Accuracy: {accuracy}")

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_hidden_size1 = hidden_size1
                    best_hidden_size2 = hidden_size2
                    best_learning_rate = learning_rate
                    best_epochs = epochs

# Utworzenie i trenowanie modelu z optymalnymi hiperparametrami
best_mlp = MLP(X_train.shape[1], best_hidden_size1, best_hidden_size2, 1)
best_mlp.train(X_train, y_train, epochs=best_epochs, learning_rate=best_learning_rate)

# Testowanie na zbiorze testowym
predictions = best_mlp.predict(X_test)

print(f"\nBest epochs amount: {best_epochs}, Best learning rate: {best_learning_rate} for Best hidden1 size: {best_hidden_size1} and Best hidden2 size: {best_hidden_size2}")

# Policzenie dokładności
accuracy = np.mean((predictions > 0.5) == y_test)
print(f"\nAccuracy: {round(accuracy,4)}")

# Macierz pomyłek
TP=0
TN=0
FP=0
FN=0
for i,j in zip(predictions,y_test):
    if ((i>0.5) and j==1):
        TP = TP + 1
    elif ((i>0.5) and j!=1):
        FP = FP + 1
    elif ((i<=0.5) and j!=0):
        FN = FN + 1
    elif ((i<=0.5) and j==0):
        TN = TN + 1
print(f"\nConfussion matrix:\n {TP} {FP} \n {FN} {TN}")


# Obliczenie precyzji
if TP+FP != 0:
    precision = TP/(TP+FP)
    print(f"\nPrecision: {round(precision,4)}")
else:
    print("Cant calculate precission")

# Obliczanie recall
recall = TP/(TP+FN)
print(f"\nRecall: {round(recall,4)}")

# Oblicznie miary F1 - średnia harmoniczna
if (TP+FP) != 0:
    beta=1
    f1Measure = ((1 + beta * beta) * precision * recall) / (beta * beta * precision + recall)
    print(f"\nF1-Measure: {round(f1Measure,4)}")
else:
    print("Cant calculate F-Measure")
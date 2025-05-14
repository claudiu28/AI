import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns


def sigmoid(x):
    # Prevent overflow
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return np.where(x > 0, 1, 0)


def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-15  # Prevent log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss


class CustomANN:
    def __init__(self, input_size, hidden_layers, output_size):
        self.z_values = []
        self.activations = []
        self.layers = [input_size] + hidden_layers + [output_size]
        self.weights = []
        self.biases = []

        for i in range(len(self.layers) - 1):
            limit = np.sqrt(6 / (self.layers[i] + self.layers[i + 1]))
            w = np.random.uniform(-limit, limit, (self.layers[i], self.layers[i + 1]))
            b = np.zeros((1, self.layers[i + 1]))

            self.weights.append(w)
            self.biases.append(b)

        self.training_history = {'loss': [], 'accuracy': []}

    def forward_propagation(self, x):
        self.activations = [x]
        self.z_values = []

        for i in range(len(self.weights)):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.z_values.append(z)

            if i == len(self.weights) - 1:  # Output layer
                activation = sigmoid(z)
            else:
                activation = relu(z)

            self.activations.append(activation)

        return self.activations[-1]

    def backward_propagation(self, x, y, learning_rate):
        m = x.shape[0]
        output = self.activations[-1]

        y_reshaped = y.reshape(-1, 1)
        delta = (output - y_reshaped) * sigmoid_derivative(output)

        for i in reversed(range(len(self.weights))):
            # Calculate gradients
            dw = np.dot(self.activations[i].T, delta) / m
            db = np.sum(delta, axis=0, keepdims=True) / m

            self.weights[i] -= learning_rate * dw
            self.biases[i] -= learning_rate * db

            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * relu_derivative(self.activations[i])

    def train(self, x, y, epochs=1000, learning_rate=0.01, batch_size=32,x_val=None, y_val=None, verbose=10):
        n_samples = x.shape[0]
        n_batches = n_samples // batch_size

        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            x_shuffled = x[indices]
            y_shuffled = y[indices]

            epoch_loss = 0
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size

                x_batch = x_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]

                output = self.forward_propagation(x_batch)
                batch_loss = binary_cross_entropy(y_batch, output)
                self.backward_propagation(x_batch, y_batch, learning_rate)

                epoch_loss += batch_loss

            if n_samples % batch_size != 0:
                start_idx = n_batches * batch_size
                x_batch = x_shuffled[start_idx:]
                y_batch = y_shuffled[start_idx:]

                output = self.forward_propagation(x_batch)
                batch_loss = binary_cross_entropy(y_batch, output)
                self.backward_propagation(x_batch, y_batch, learning_rate)

                epoch_loss += batch_loss
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            self.training_history['loss'].append(avg_loss)

            train_pred = self.predict(x)
            train_accuracy = accuracy_score(y, train_pred)
            self.training_history['accuracy'].append(train_accuracy)


    def predict(self, x):
        output = self.forward_propagation(x)
        return (output > 0.5).astype(int).flatten()

    def predict_proba(self, x):
        return self.forward_propagation(x).flatten()


def load_sepia_dataset(dataset_path="../../sepia_dataset"):
    features = []
    labels = []

    normal_dir = os.path.join(dataset_path, "normal")
    for filename in os.listdir(normal_dir):
        if filename.endswith('.jpg'):
            img_path = os.path.join(normal_dir, filename)
            image = cv2.imread(img_path)

            hist_r = cv2.calcHist([image], [0], None, [64], [0, 256]).flatten()
            hist_g = cv2.calcHist([image], [1], None, [64], [0, 256]).flatten()
            hist_b = cv2.calcHist([image], [2], None, [64], [0, 256]).flatten()

            feature = np.concatenate([hist_r, hist_g, hist_b])
            features.append(feature)
            labels.append(0)

    sepia_dir = os.path.join(dataset_path, "sepia")
    for filename in os.listdir(sepia_dir):
        if filename.endswith('.jpg'):
            img_path = os.path.join(sepia_dir, filename)
            image = cv2.imread(img_path)

            hist_r = cv2.calcHist([image], [0], None, [64], [0, 256]).flatten()
            hist_g = cv2.calcHist([image], [1], None, [64], [0, 256]).flatten()
            hist_b = cv2.calcHist([image], [2], None, [64], [0, 256]).flatten()

            feature = np.concatenate([hist_r, hist_g, hist_b])
            features.append(feature)
            labels.append(1)

    return np.array(features), np.array(labels)


def split_data(inputs, outputs, train_ratio=0.8, seed=5):
    np.random.seed(seed)
    n_samples = len(inputs)
    indices = np.arange(n_samples)

    n_train = int(train_ratio * n_samples)
    train_indices = np.random.choice(indices, n_train, replace=False)
    test_indices = np.array([i for i in indices if i not in train_indices])

    train_inputs = inputs[train_indices]
    train_outputs = outputs[train_indices]
    test_inputs = inputs[test_indices]
    test_outputs = outputs[test_indices]

    return train_inputs, train_outputs, test_inputs, test_outputs


def normalisation(train_data, test_data):
    scaler = StandardScaler()

    if len(train_data.shape) == 1:
        train_data = train_data.reshape(-1, 1)
    if len(test_data.shape) == 1:
        test_data = test_data.reshape(-1, 1)

    scaler.fit(train_data)
    normalised_train_data = scaler.transform(train_data)
    normalised_test_data = scaler.transform(test_data)

    return normalised_train_data, normalised_test_data, scaler


def plot_confusion_matrix(labels_true, labels_pred):
    cm = confusion_matrix(labels_true, labels_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',xticklabels=['Normal', 'Sepia'],yticklabels=['Normal', 'Sepia'])
    plt.title('Confusion Matrix - Custom ANN')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('confusion_matrix_custom.png')
    plt.show()


def main():
    if not os.path.exists("../../sepia_dataset"):
        print("Dataset not found!")
        return

    features, labels = load_sepia_dataset()
    print(f"Dataset: {len(features)} images")

    feature_train, label_train, feature_test, label_test = split_data(features, labels)

    # Normalize f
    feature_train_scaled, feature_test_scaled, scaler = normalisation(feature_train, feature_test)

    ann = CustomANN(input_size=192, hidden_layers=[100, 50], output_size=1)

    ann.train(feature_train_scaled, label_train,epochs=1000,learning_rate=0.01,batch_size=32,x_val=feature_test_scaled, y_val=label_test,verbose=100)

    print("\nTest model..")
    predictions = ann.predict(feature_test_scaled)

    correct_predictions = sum(1 for i in range(len(label_test)) if label_test[i] == predictions[i])
    accuracy = correct_predictions / len(label_test)

    print(f"\nAccuracy: {accuracy * 100:.2f}%")

    plot_confusion_matrix(label_test, predictions)

if __name__ == "__main__":
    main()
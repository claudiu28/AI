import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns

def load_sepia_dataset(dataset_path="../../sepia_dataset"):
    feature = []  # feature -> per imagine ori feature
    labels = []  # labels -> normal / sepia

    normal_dir = os.path.join(dataset_path, "normal")
    for filename in os.listdir(normal_dir):
        if filename.endswith('.jpg'):
            img_path = os.path.join(normal_dir, filename)
            image = cv2.imread(img_path)

            # caracteristici -> extrase per imagine
            hist_r = cv2.calcHist([image], [0], None, [64], [0, 256]).flatten()
            hist_g = cv2.calcHist([image], [1], None, [64], [0, 256]).flatten()
            hist_b = cv2.calcHist([image], [2], None, [64], [0, 256]).flatten()

            features = np.concatenate([hist_r, hist_g, hist_b])
            feature.append(features)
            labels.append(0)

    sepia_dir = os.path.join(dataset_path, "sepia")
    for filename in os.listdir(sepia_dir):
        if filename.endswith('.jpg'):
            img_path = os.path.join(sepia_dir, filename)
            image = cv2.imread(img_path)

            # Extrage aceleași features
            hist_r = cv2.calcHist([image], [0], None, [64], [0, 256]).flatten()
            hist_g = cv2.calcHist([image], [1], None, [64], [0, 256]).flatten()
            hist_b = cv2.calcHist([image], [2], None, [64], [0, 256]).flatten()

            features = np.concatenate([hist_r, hist_g, hist_b])
            feature.append(features)
            labels.append(1)

    return np.array(feature), np.array(labels)

def classifier():

    if not os.path.exists("../../sepia_dataset"):
        return

    features, labels = load_sepia_dataset()
    print(f"Imagini total: {len(features)}")


    # test -> 80% 20%
    feature_train, feature_test, label_train, label_test = train_test_split(features, labels, test_size=0.2, random_state=50, stratify = labels)
    print(f"Training: {len(feature_train)} imagini")
    print(f"Testing: {len(label_test)} imagini")

    # normalizing
    scaler = StandardScaler()
    features_train_scaled = scaler.fit_transform(feature_train)
    features_test_scaled = scaler.transform(feature_test)

    # learning model
    print("\nANN:")
    model = MLPClassifier(hidden_layer_sizes=(100, 50),activation='relu',max_iter=1000,random_state=50,verbose=1)
    model.fit(features_train_scaled, label_train)

    # test model
    print("\nTest ANN:")
    label_prediction = model.predict(features_test_scaled)

    # accuracy
    accuracy = accuracy_score(label_test, label_prediction)
    print(f"\nAccuracy: {accuracy * 100:.2f}%")

    print(classification_report(label_test, label_prediction, target_names=["normal", "sepia"]))

    # matrice de confuzie:
    print("\nMatricea de confuzie:")
    cm = confusion_matrix(label_test, label_prediction)

    # diagrma confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',xticklabels=['Normal', 'Sepia'],yticklabels=['Normal', 'Sepia'])
    plt.title('CM -> Clasificator Sepia')
    plt.xlabel('Predictie')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()
    return model, scaler

def main():
    model, scaler = classifier()

if __name__ == "__main__":
    main()
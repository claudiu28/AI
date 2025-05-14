import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
import gensim
import kagglehub
from win32comext.adsi.demos.scp import verbose


def load_data(file_path):
    texts = []
    sentiments = []
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            if len(row) >= 2:
                texts.append(row[0])
                sentiments.append(row[1])
    return texts, sentiments


def bow_features(train_texts, test_texts):
    vectorizer = CountVectorizer()
    train = vectorizer.fit_transform(train_texts)
    test = vectorizer.transform(test_texts)

    feature_names = vectorizer.get_feature_names_out()

    print("Vocab size:", len(feature_names), "words")
    print("Train data size:", train.shape[0])
    print("Test data size:", test.shape[0])
    print("Some words of the vocab:", feature_names[:10])

    return train, test, vectorizer


def tfidf_features(train_texts, test_texts):
    vectorizer = TfidfVectorizer()
    train = vectorizer.fit_transform(train_texts)
    test = vectorizer.transform(test_texts)

    feature_names = vectorizer.get_feature_names_out()

    print("Vocab size:", len(feature_names), "words")
    print("Train data size:", train.shape[0])
    print("Test data size:", test.shape[0])
    print("Some words of the vocab:", feature_names[:10])

    return train, test, vectorizer


def load_or_download_word2vec():
    model_dir = kagglehub.dataset_download("leadbest/googlenewsvectorsnegative300")
    model_path = os.path.join(model_dir, "GoogleNews-vectors-negative300.bin")
    return gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)


def word2vec_features(model, texts):
    features = []
    for text in texts:
        words = text.split()
        vectors = [model[w] for w in words if len(w) > 2 and w in model.index_to_key]
        if vectors:
            avg = np.mean(vectors, axis=0)
        else:
            avg = np.zeros(model.vector_size)
        features.append(avg)

    if len(features) > 0:
        print(f"W2V size: {model.vector_size}")
        first_vec = features[0]
        print(f"First text vector stats: mean={np.mean(first_vec):.4f}, std={np.std(first_vec):.4f}")
        print(f"First 5 dim: {first_vec[:5]}")
    return np.array(features)


def more_features(texts):
    positive = ['good', 'great', 'excellent', 'happy', 'love', 'best', 'favorite',
                'wonderful', 'perfect', 'awesome', 'beautiful', 'amazing', 'fantastic',
                'eco-friendly', 'environmental', 'green', 'sustainable', 'proud']

    negative = ['bad', 'terrible', 'awful', 'hate', 'worst', 'poor', 'horrible',
                'disappointing', 'waste', 'failure', 'broken', 'useless', 'annoying']

    features = []
    feature_names = [
        'text_length', 'word_count', 'avg_word_length', 'special_chars',
        'uppercase_ratio', 'positive_words', 'negative_words',
        'exclamations', 'questions'
    ]

    for text in texts:
        f = {
            'text_length': len(text),
            'word_count': len(text.split()),
            'avg_word_length': np.mean([len(w) for w in text.split()]) if text.split() else 0,
            'special_chars': sum(not c.isalnum() and not c.isspace() for c in text),
            'uppercase_ratio': sum(c.isupper() for c in text) / len(text) if len(text) > 0 else 0,
            'positive_words': sum(w in positive for w in text.lower().split()),
            'negative_words': sum(w in negative for w in text.lower().split()),
            'exclamations': text.count('!'),
            'questions': text.count('?')
        }
        features.append(list(f.values()))

    if features:
        words = texts[0].lower().split()
        pos_found = [w for w in words if w in positive]
        neg_found = [w for w in words if w in negative]
        print("\nPositive words:", pos_found if pos_found else "Nothing pos")
        print("Negative words:", neg_found if neg_found else "Nothing neg")

    return np.array(features)


def classifier(feature_train, feature_test, label_train, label_test, label_encoder):
    model = MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=300, random_state=50, activation='relu', verbose = 1)
    model.fit(feature_train, label_train)
    label_prediction = model.predict(feature_test)

    print("Accuracy:", model.score(feature_test, label_test))

    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(label_test, label_prediction), annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    return model


def predict_sentiment(tfidf_vectorizer, word2vec_model, model, text, label_encoder):
    print(f"Text: {text}")
    tfidf = tfidf_vectorizer.transform([text]).toarray()
    w2v = word2vec_features(word2vec_model, [text])
    mf = more_features([text])
    combine = np.hstack([tfidf, w2v, mf])
    predicted_class = model.predict(combine)
    predicted_label = label_encoder.inverse_transform(predicted_class)[0]
    print(f"\nPredicted Sentiment: {predicted_label}")


def main():
    data_path = os.path.join(os.getcwd(), "../data", "reviews_mixed.csv")
    texts, labels = load_data(data_path)

    # 80% train, 20% test
    feature_train_texts, feature_test_texts, labels_train, labels_test = train_test_split(texts, labels, test_size=0.2,
                                                                                          random_state=50)

    label_encoder = LabelEncoder()
    labels_train = label_encoder.fit_transform(labels_train)
    labels_test = label_encoder.transform(labels_test)

    bow_train, bow_test, bow_vectorizer = bow_features(feature_train_texts, feature_test_texts)
    tfidf_train, tfidf_test, tfidf_vectorizer = tfidf_features(feature_train_texts, feature_test_texts)

    word2vec_model = load_or_download_word2vec()
    w2v_train = word2vec_features(word2vec_model, feature_train_texts)
    w2v_test = word2vec_features(word2vec_model, feature_test_texts)

    more_features_train = more_features(feature_train_texts)
    more_features_test = more_features(feature_test_texts)

    future_train = hstack([tfidf_train, csr_matrix(w2v_train), csr_matrix(more_features_train)])
    future_test = hstack([tfidf_test, csr_matrix(w2v_test), csr_matrix(more_features_test)])

    print(f"TF-IDF features: {tfidf_train.shape[1]}")
    print(f"Word2Vec features: {w2v_train.shape[1]}")
    print(f"Custom features: {more_features_train.shape[1]}")

    model = classifier(future_train, future_test, labels_train, labels_test, label_encoder)

    text = "By choosing a bike over a car, I'm reducing my environmental footprint. Cycling promotes eco-friendly transportation, and I'm proud to be part of that movement."
    predict_sentiment(tfidf_vectorizer, word2vec_model, model, text, label_encoder)


if __name__ == "__main__":
    main()

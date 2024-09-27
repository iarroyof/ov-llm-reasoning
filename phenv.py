import math
from collections import Counter
from typing import List, Dict, Tuple, Any
from joblib import Parallel, delayed, parallel_backend
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

class PhraseEntropyViews:
    def __init__(self):
        # Download necessary NLTK data
        nltk.download('punkt', quiet=True)

    @staticmethod
    def tokenize_sentence(sentence: str) -> List[str]:
        """Tokenize a sentence into words and subwords."""
        words = word_tokenize(sentence.lower())
        subwords = []
        for word in words:
            subwords.extend([word[i:j] for i in range(len(word)) for j in range(i + 1, len(word) + 1)])
        return words + subwords

    @staticmethod
    def calculate_entropy(tokens: List[str]) -> float:
        """Calculate the entropy of a list of tokens."""
        freq = Counter(tokens)
        probs = [count / len(tokens) for count in freq.values()]
        return -sum(p * math.log2(p) for p in probs)

    @classmethod
    def process_batch(cls, batch: List[Dict[str, str]], output_type='vector', token_weight=0.7) -> List[Dict[str, Any]]:
        """Process a batch of sentences and return their entropies."""
        entropies = []
        for item in batch:
            sentence = item['sentence']
            tokens = cls.tokenize_sentence(sentence)
            word_lengths = [len(word) for word in word_tokenize(sentence)]
            
            token_entropy = cls.calculate_entropy(tokens)
            length_entropy = cls.calculate_entropy(word_lengths)
            
            if output_type == 'float':
                combined_entropy = token_weight * token_entropy + (1 - token_weight) * length_entropy
            elif output_type == 'vector':
                combined_entropy = [token_weight * token_entropy, (1 - token_weight) * length_entropy]
                
            entropies.append({'entropy': combined_entropy, 'label': item['label']})

        return entropies

    def fit_entropies(self, batched_dataset: List[List[Dict[str, str]]], output_file: str='out_entropies', return_results: bool=False):
        all_entropies = []
        with Parallel(n_jobs=-1) as parallel:
            batch_entropies = parallel(delayed(self.process_batch)(batch, output_type='vector') for batch in batched_dataset)
            with open(output_file, 'ab') as f:
                np.savez(f, *batch_entropies)
        if return_results:
            all_entropies.extend(batch_entropies)
            return all_entropies

    @staticmethod
    def plot_scatter(result):
        x_coords, y_coords, colors, labels = [], [], [], []
        for batch_entropy in result:
            for item in batch_entropy:
                x_coords.append(item['entropy'][0])
                y_coords.append(item['entropy'][1])
                colors.append('blue' if item['label'] == 'high_info' else 'red')
                labels.append(item['label'])

        plt.figure(figsize=(8, 6))
        plt.scatter(x_coords, y_coords, c=colors)
        plt.xlabel('Token Entropy')
        plt.ylabel('Length Entropy')
        plt.title('Scatter Plot of Entropy Values')
        plt.show()

        return x_coords, y_coords, labels

    @staticmethod
    def train_mlp(x_coords, y_coords, labels):
        dataset = pd.DataFrame({'x': x_coords, 'y': y_coords, 'label': labels})
        dataset['label'] = dataset['label'].map({'low_info': 0, 'high_info': 1})

        X = dataset[['x', 'y']]
        y = dataset['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        mlp = MLPClassifier(hidden_layer_sizes=(10, 5), activation='relu', solver='adam', random_state=42)
        mlp.fit(X_train, y_train)

        y_pred = mlp.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"MLP Accuracy: {accuracy}")

        return mlp, X, y

    @staticmethod
    def plot_mlp_decision_boundary(model, X, y, x_coords, y_coords, colors):
        title='MLP Decision Boundary'
        xx, yy = np.meshgrid(np.arange(X['x'].min()-1, X['x'].max()+1, 0.1),
                             np.arange(X['y'].min()-1, X['y'].max()+1, 0.1))

        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.figure(figsize=(8, 6))
        plt.contourf(xx, yy, Z, alpha=0.4)
        plt.scatter(x_coords, y_coords, c=colors)
        plt.xlabel('Token Entropy')
        plt.ylabel('Length Entropy')
        plt.title(title)
        plt.show()

    @staticmethod
    def train_svm(x_coords, y_coords, labels):
        dataset = pd.DataFrame({'x': x_coords, 'y': y_coords, 'label': labels})
        dataset['label'] = dataset['label'].map({'low_info': 0, 'high_info': 1})

        X = dataset[['x', 'y']]
        y = dataset['label']

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Initialize and train the online SVM
        svm = SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001, max_iter=1000, tol=1e-3, random_state=42)
        svm.fit(X_scaled, y)

        # Evaluate the model
        y_pred = svm.predict(X_scaled)
        accuracy = accuracy_score(y, y_pred)
        print(f"Online SVM Accuracy: {accuracy}")

        return svm, scaler, X, y

    @staticmethod
    def plot_svm_decision_boundary(svm, scaler, X, y, x_coords, y_coords, colors):
        xx, yy = np.meshgrid(np.arange(X['x'].min()-1, X['x'].max()+1, 0.1),
                             np.arange(X['y'].min()-1, X['y'].max()+1, 0.1))

        # Scale the mesh
        mesh_scaled = scaler.transform(np.c_[xx.ravel(), yy.ravel()])

        Z = svm.predict(mesh_scaled)
        Z = Z.reshape(xx.shape)

        plt.figure(figsize=(8, 6))
        plt.contourf(xx, yy, Z, alpha=0.4)
        plt.scatter(x_coords, y_coords, c=colors)
        plt.xlabel('Token Entropy')
        plt.ylabel('Length Entropy')
        plt.title('Scatter Plot of Entropy Values with Online SVM Decision Boundary')
        plt.show()
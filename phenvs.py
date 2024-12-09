import math
from collections import Counter
import itertools
from typing import List, Dict, Tuple, Any
from joblib import Parallel, delayed, parallel_backend
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.decomposition import IncrementalPCA
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import silhouette_score
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

from pdb import set_trace as st

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
    def calculate_entropy(tokens: List[str], sample_len: int=None) -> float:
        """Calculate the entropy of a list of tokens."""
        
        freq = Counter(tokens)
        probs = [count / (sample_len if sample_len else len(tokens)) for count in freq.values()]
        return -sum(p * math.log2(p) for p in probs)

    @classmethod
    def process_batch(cls, batch: List[Dict[str, str]], output_type: str='vector', token_weight: float=0.7, batch_view: bool=False) -> List[Dict[str, Any]]:
        """Process a batch of sentences and return their entropies."""
        entropies = []
        token_batch_sample_len = 0
        length_batch_sample_len = 0
        batch_tokens = []
        batch_lengths = []
        for item in batch:
            #st()
            #sentence = item['sentence']
            sentence = item[0]
            tokens = cls.tokenize_sentence(sentence)
            word_lengths = [len(word) for word in word_tokenize(sentence)]
            if batch_view:
                token_batch_sample_len += len(tokens)
                length_batch_sample_len += len(word_lengths)
                batch_tokens.append(tokens)
                batch_lengths.append(word_lengths)
            token_entropy = cls.calculate_entropy(tokens)
            length_entropy = cls.calculate_entropy(word_lengths)

            if output_type == 'float':
                combined_entropy = token_weight * token_entropy + (1 - token_weight) * length_entropy
            elif output_type == 'vector':
                combined_entropy = [token_weight * token_entropy, (1 - token_weight) * length_entropy]
                
            #entropies.append({'entropy': combined_entropy, 'label': item['label']})
            entropies.append({'entropy': combined_entropy})

        if batch_view:
            for i, (tokens, lengths) in enumerate(zip(batch_tokens, batch_lengths)):
                token_entropy = cls.calculate_entropy(tokens, token_batch_sample_len)
                length_entropy = cls.calculate_entropy(lengths, length_batch_sample_len)
                
                if output_type == 'float':
                    combined_entropy = token_weight * token_entropy + (1 - token_weight) * length_entropy
                    entropies[i]['entropy'] = entropies[i]['entropy'] + combined_entropy
                elif output_type == 'vector':
                    combined_entropy = [token_weight * token_entropy, (1 - token_weight) * length_entropy]
                    entropies[i]['entropy'].extend(combined_entropy)

        return entropies

    def fit_entropies(self,
            batched_dataset: List[List[Dict[str, str]]],
            output_file: str='out_entropies',
            return_results: bool=False,
            output_type: str='vector',
            batch_view: bool=True,
            n_jobs: int=-1):
        all_entropies = []
        with Parallel(n_jobs=n_jobs) as parallel:
            batch_entropies = parallel(delayed(self.process_batch)(batch, output_type=output_type, batch_view=batch_view) for batch in batched_dataset)
            with open(output_file, 'ab') as f:
                np.savez(f, *batch_entropies)
        if return_results:
            all_entropies.extend(batch_entropies)
            return all_entropies

    @staticmethod
    def plot_scatter(result, batch_or_sample: str='batch', return_coords: bool=False):
        x_coords, y_coords, colors, labels = [], [], [], []
        for batch_entropy in result:
            for item in batch_entropy:
                x_coords.append(item['entropy'][0 if batch_or_sample == 'sample' else 2])
                y_coords.append(item['entropy'][1 if batch_or_sample == 'sample' else 3])
                colors.append('blue' if item['label'] == 'high_info' else 'red')
                labels.append(item['label'])

        plt.figure(figsize=(8, 6))
        plt.scatter(x_coords, y_coords, c=colors)
        plt.xlabel('Token Entropy')
        plt.ylabel('Length Entropy')
        plt.title(f'Scatter Plot of {batch_or_sample}-based Entropy Values')
        plt.show()
        
        if return_coords:
            return x_coords, y_coords, labels

    @staticmethod
    def train_mlp(batched_dataset, test_size: float=0.2, n_epochs: int=10):
        mlp = MLPClassifier(hidden_layer_sizes=(10, 5), activation='relu', solver='adam', random_state=42)

        for e in range(n_epochs):
            for i, batch in enumerate(batched_dataset):
                # Crear un DataFrame a partir del batch actual
                data = []
                for sample in batch:
                    data.append({f'dim_{i}': coord for i, coord in enumerate(sample['entropy'])})
                    data[-1]['label'] = sample['label']
                df = pd.DataFrame(data)

                # Mapear las etiquetas a valores numéricos
                df['label'] = df['label'].map({'low_info': 0, 'high_info': 1})

                # Separar en caracteristicas y etiquetas
                X = df.drop('label', axis=1)
                y = df['label']

                # Dividir en conjuntos de entrenamiento y prueba
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42)
                # Entrenar el modelo usando partial_fit
                mlp.partial_fit(X_train, y_train, classes=[0, 1])

                # Evaluar el modelo en el conjunto de prueba
                y_pred = mlp.predict(X_test)
                accuracy += accuracy_score(y_test, y_pred)
                f1 += f1_score(y_test, y_pred)
                
            print(f"Avg batch Accuracy at epoch {e}: {accuracy/i}")
            print(f"Avg batch F1-score at epoch {e}: {f1/i}")

        return mlp

    @staticmethod
    def train_mlp(batched_dataset, n_epochs: int=10, mode: str='2dembed', plot: bool=False):
        n_components = 2
        # Inicializar el clasificador MLP
        mlp = MLPClassifier(hidden_layer_sizes=(10, 5), activation='relu', solver='adam', random_state=42)
        # Inicializar el codificador de etiquetas
        label_encoder = LabelEncoder()
        #pca = MiniBatchDictionaryLearning(
        #    n_components=n_components, batch_size=len(batched_dataset[0]), random_state=42)
        pca = IncrementalPCA(n_components=n_components)
        if mode == 'full':
            # Entrenar el modelo con todos los datos de entrenamiento
            avg_f1 = 0
            
            for epoch in range(n_epochs):
                f1 = 0
                batch_test_preds = []
                batch_X_test = []
                batch_y_test = []
                for b, batch in enumerate(batched_dataset):
                    split_index = int(0.8 * len(batch))
                    X_train, X_test = ([d['entropy'] for d in batch][:split_index],
                        [d['entropy'] for d in batch][split_index:])
                    y_train, y_test = ([1 if d['label'] == 'high_info' else 0 for d in batch][:split_index],
                        [1 if d['label'] == 'high_info' else 0 for d in batch][split_index:])
                    
                    mlp.partial_fit(X_train, y_train, classes=[0, 1])
                        # Evaluar el modelo
                    y_pred = mlp.predict(X_test)

                    f1 += f1_score(y_test, y_pred, average='macro')
                    if plot:
                        pca.partial_fit(X_train)
                        batch_X_test.append(X_test)
                        batch_y_test.append(y_test)
                        batch_test_preds.append(y_pred)

                print(f"Epoch {epoch} F1 Score: {f1/b}")

            if plot:
                import itertools
                # Visualización
                X_embedded = pca.transform(np.array(list(itertools.chain(*batch_X_test))))
                y_test = np.array(list(itertools.chain(*batch_y_test)))
                y_pred = np.array(list(itertools.chain(*batch_test_preds)))
                
                plt.figure(figsize=(10, 8))        
           
                correct_class_0 = (y_test == 0) & (y_test == y_pred)
                correct_class_1 = (y_test == 1) & (y_test == y_pred)
                incorrect_class_0 = (y_test == 0) & (y_test != y_pred)
                incorrect_class_1 = (y_test == 1) & (y_test != y_pred)
                
                plt.scatter(X_embedded[correct_class_0, 0], X_embedded[correct_class_0, 1],
                    c='red', label='Classified Class 0')
                plt.scatter(X_embedded[correct_class_1, 0], X_embedded[correct_class_1, 1],
                    c='blue', label='Classified Class 1')
                plt.scatter(X_embedded[incorrect_class_0, 0], X_embedded[incorrect_class_0, 1],
                    c='orange', label='Misclassified Class 0')
                plt.scatter(X_embedded[incorrect_class_1, 0], X_embedded[incorrect_class_1, 1],
                    c='cyan', label='Misclassified Class 1')
                
                plt.title("Visualization of test data")
                plt.xlabel("Feature 1")
                plt.ylabel("Feature 2")
                plt.legend()
            
        elif mode == '2dembed':
            
            for epoch in range(n_epochs):
                f1 = 0
                batch_test_preds = []
                batch_X_test = []
                batch_y_test = []            
                for b, batch in enumerate(batched_dataset):
                    split_index = int(0.8 * len(batch))
                    X = np.array([d['entropy'] for d in batch])
                    pca.partial_fit(X)
                    X_embedded = pca.transform(X)
                    X_train, X_test = (X_embedded[:split_index], X_embedded[split_index:])
                    y_train, y_test = ([1 if d['label'] == 'high_info' else 0 for d in batch][:split_index],
                        [1 if d['label'] == 'high_info' else 0 for d in batch][split_index:])
                    
                    mlp.partial_fit(X_train, y_train, classes=[0, 1])
                        # Evaluar el modelo
                    y_pred = mlp.predict(X_test)

                    f1 += f1_score(y_test, y_pred, average='macro')
                    if plot:
                        batch_X_test.append(X_test.tolist())
                        batch_y_test.append(y_test)
                        batch_test_preds.append(y_pred)

                print(f"Epoch {epoch} F1 Score: {f1/b}")
                
            if plot:
                import itertools
                # Visualización
                X_test = np.array(list(itertools.chain(*batch_X_test)))
                y_test = np.array(list(itertools.chain(*batch_y_test)))
                y_pred = np.array(list(itertools.chain(*batch_test_preds)))
            
                plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis', alpha=0.7)
                
                x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
                y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
                xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
                
                Z = mlp.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)
                
                plt.contourf(xx, yy, Z, alpha=0.2, cmap='viridis')
                
                plt.title("Decision function visualization")
                plt.xlabel("Feature 1")
                plt.ylabel("Feature 2")
            
                plt.colorbar(ticks=[0, 1], label='Label')
            
        plt.show()
    
        return mlp


    @staticmethod
    def cluster_entropy_levels(batched_dataset, n_epochs: int=10, mode: str='2dembed', plot: bool=False, batches_to_plot: int=5):
        n_components = 2
        n_clusters = 2
        # Inicializar el clasificador MLP
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters, random_state=42, batch_size=len(batched_dataset[0]))
        # Inicializar el codificador de etiquetas
        label_encoder = LabelEncoder()
        pca = IncrementalPCA(n_components=n_components)
        if mode == 'full':
            # Entrenar el modelo con todos los datos de entrenamiento
            for epoch in range(n_epochs):
                batch_preds = []
                batch_X = []
                validation_scores = []
                for b, batch in enumerate(batched_dataset):
                    split_index = int(0.8 * len(batch))
                    X_train, X_test = ([d['entropy'] for d in batch][:split_index],
                        [d['entropy'] for d in batch][split_index:])

                    kmeans.partial_fit(X_train)
                        # Evaluar el modelo
                    y_pred_test = kmeans.predict(X_test)
                    if len(np.unique(y_pred_test)) > 1:
                        score = silhouette_score(X_test, y_pred_test)
                        print(f"Validation Silhouette Score: {score}")
                    else:
                        score = 0.0
                        print(f"Validation Hard-Fixed Null score: {score}")
                    
                    validation_scores.append(score)                    
                    # Reassign new labels based on distances from the origin
                    # '0' for the nearest (lower entropy) and '1' for the fartest (high entropy)
                    kmeans.partial_fit(X_test)
                    y_pred_train = kmeans.predict(X_train)
                    centroids = kmeans.cluster_centers_
                    distances = np.linalg.norm(centroids, axis=1)
                    # '0' for the nearest cluster (lower entropy) and '1' for the fartest one (high entropy)
                    sorted_indices = np.argsort(distances)[::-1]
                    label_map = {old_label: new_label
                        for new_label, old_label in enumerate(sorted_indices)}
                    
                    y_pred = np.concatenate([y_pred_train, y_pred_test])
                    y_pred = np.array([label_map[label] for label in y_pred])
                    
                    if plot and b < batches_to_plot:
                        # To plot, it is needed to project the full data into a 2D space using PCA
                        X_batch = np.concatenate([X_train, X_test])
                        pca.partial_fit(X_batch)
                        batch_X.append(X_batch)
                        batch_preds.append(y_pred)
                print(f"Epoch average validation Score {np.mean(validation_scores)}")
            if plot:
                # Visualización
                X_embedded = pca.transform(np.array(list(itertools.chain(*batch_X))))
                y_pred = np.array(list(itertools.chain(*batch_preds)))
                plt.figure(figsize=(10, 8))
                plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y_pred, cmap='viridis', alpha=0.7)
                plt.title("Visualization of test data")
                plt.xlabel("Feature 1")
                plt.ylabel("Feature 2")
                legend_elements = [Line2D([0], [0], marker='o', color='w',
                    label='Low entropy', markerfacecolor='purple', markersize=10),
                                  Line2D([0], [0], marker='o', color='w',
                    label='High entropy', markerfacecolor='yellow', markersize=10)]
                plt.legend(handles=legend_elements)
                        
        elif mode == '2dembed':
            
            for epoch in range(n_epochs):
                batch_preds = []
                batch_X = []
                batch_y = []            
                validation_scores = []
                for b, batch in enumerate(batched_dataset):
                    split_index = int(0.8 * len(batch))
                    X = np.array([d['entropy'] for d in batch])
                    pca.partial_fit(X)
                    X_embedded = pca.transform(X)
                    X_train, X_test = (X_embedded[:split_index], X_embedded[split_index:])
                    #y_train, y_test = (
                    #    [1 if d['label'] == 'high_info' else 0 for d in batch][:split_index],
                    #    [1 if d['label'] == 'high_info' else 0 for d in batch][split_index:])
                    
                    kmeans.partial_fit(X_train)
                        # Evaluar el modelo
                    y_pred_test = kmeans.predict(X_test)
                    if len(np.unique(y_pred_test)) > 1:
                        score = silhouette_score(X_test, y_pred_test)
                        print(f"Validation Silhouette Score: {score}")
                    else:
                        score = 0.0
                        print(f"Validation Hard-Fixed Null score: {score}")
                        
                    validation_scores.append(score)
                    kmeans.partial_fit(X_test)
                    y_pred_train = kmeans.predict(X_train)
                    centroids = kmeans.cluster_centers_
                    distances = np.linalg.norm(centroids, axis=1)
                    # '0' for the nearest cluster (lower entropy) and '1' for the fartest one (high entropy)
                    sorted_indices = np.argsort(distances)[::-1]
                    label_map = {old_label: new_label
                        for new_label, old_label in enumerate(sorted_indices)}

                    y_pred = np.concatenate([y_pred_train, y_pred_test])
                    
                    y_pred = np.array([label_map[label] for label in y_pred])
                    
                    if plot and b < batches_to_plot:
                        # To plot, the full data is already a 2D PCA space 
                        X_batch = np.concatenate([X_train, X_test])
                        batch_X.append(X_batch)
                        batch_preds.append(y_pred)
                print(f"Epoch average validation Score {np.mean(validation_scores)}")
            if plot:
                # Visualización
                X = np.array(list(itertools.chain(*batch_X)))
                y_pred = np.array(list(itertools.chain(*batch_preds)))
            
                plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', alpha=0.7)
                
                x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
                y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
                xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
                
                Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)
                
                plt.contourf(xx, yy, Z, alpha=0.2, cmap='viridis')
                
                plt.title("Decision function visualization")
                plt.xlabel("Feature 1")
                plt.ylabel("Feature 2")
                cbar = plt.colorbar(ticks=[0, 1], label='Label')
                cbar.set_ticklabels(['Low Entropy', 'High Entropy'])
            
        plt.show()
    
        
        return kmeans, 
        

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

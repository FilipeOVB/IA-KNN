import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import time

# Função para calcular a distância Euclidiana
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

# Função KNN (hardcore)
def knn(X_train, y_train, X_test, k):
    predictions = []
    for test_sample in X_test:
        distances = [euclidean_distance(test_sample, x) for x in X_train]
        k_nearest_neighbors = np.argsort(distances)[:k]  # Índices dos k vizinhos mais próximos
        k_nearest_labels = [y_train[i] for i in k_nearest_neighbors]
        # Votação majoritária
        most_common = Counter(k_nearest_labels).most_common(1)[0][0]
        predictions.append(most_common)
    return predictions

# Carregar o dataset iris
df = pd.read_csv('iris.csv')

# Separar as características e o rótulo
X = df.drop(columns=['Id', 'Species']).values
y = df['Species'].values

# Dividir os dados em treino e teste (80% treino, 20% teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Testar o KNN para diferentes valores de k
for k in [1, 3, 5, 7]:
    start_time = time.time()  # Inicia a contagem do tempo
    
    y_pred = knn(X_train, y_train, X_test, k)
    
    elapsed_time = time.time() - start_time  # Calcula o tempo gasto
    print(f"K={k}")
    print(f"Tempo de execução: {elapsed_time:.4f} segundos")
    print(f"Acurácia: {accuracy_score(y_test, y_pred)}")
    print(f"Precisão: {precision_score(y_test, y_pred, average='macro')}")
    print(f"Revocação: {recall_score(y_test, y_pred, average='macro')}")
    print(f"Matriz de confusão:\n{confusion_matrix(y_test, y_pred)}\n")

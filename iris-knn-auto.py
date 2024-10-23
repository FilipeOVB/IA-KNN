import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import time

# Carregar o dataset iris
df = pd.read_csv('iris.csv')

# Separar as características e o rótulo
X = df.drop(columns=['Id', 'Species']).values
y = df['Species'].values

# Dividir os dados em treino e teste (80% treino, 20% teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Testar o KNN para diferentes valores de k
for k in [1, 3, 5, 7]:
    start_time = time.time()  # Iniciar a contagem de tempo
    
    # Implementar o KNN utilizando a biblioteca sklearn
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)  # Treinar o classificador
    y_pred = knn.predict(X_test)  # Fazer as previsões
    
    elapsed_time = time.time() - start_time  # Calcular o tempo gasto
    print(f"K={k}")
    print(f"Tempo de execução: {elapsed_time:.4f} segundos")
    print(f"Acurácia: {accuracy_score(y_test, y_pred)}")
    print(f"Precisão: {precision_score(y_test, y_pred, average='macro')}")
    print(f"Revocação: {recall_score(y_test, y_pred, average='macro')}")
    print(f"Matriz de confusão:\n{confusion_matrix(y_test, y_pred)}\n")

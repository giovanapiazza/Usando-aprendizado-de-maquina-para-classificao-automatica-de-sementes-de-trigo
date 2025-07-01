# Baixar e descompactar o conjunto de dados
!wget -c https://archive.ics.uci.edu/static/public/236/seeds.zip
!unzip -u seeds.zip

# Importando as bibliotecas necessárias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from zipfile import ZipFile
import requests
from sklearn import tree, model_selection
from sklearn.metrics import classification_report, confusion_matrix
import graphviz
import random



# Convertendo as classes para strings
data['Classe'] = data['Classe'].astype(str)

# Ler o arquivo de dados e adicionar os nomes das colunas
data = pd.read_csv('seeds_dataset.txt', delim_whitespace=True, header=None)
data.columns = ['Área', 'Perímetro', 'Compacidade', 'Comprimento do Grão', 'Largura do Grão',
                'Coeficiente de Assimetria', 'Comprimento do Sulco do Grão', 'Classe'] # A última coluna é a classe
data.head()  # Exibir as primeiras 5 amostras

# Armazenando os inputs na matriz X e os outputs no array y
X = data.iloc[:, :-1]  # características (todas as colunas, exceto a última)
y = data.iloc[:, -1]   # alvo (a última coluna)
print(X.describe())
print("\n", y.value_counts(), "\n")

nomes_classes = list(set(y))

# Dividindo o conjunto de dados em conjuntos de treinamento e teste
train_x, test_x, train_y, test_y = model_selection.train_test_split(X, y, train_size=0.76, stratify=y)

print('As dimensões do conjunto de dados de treinamento (inputs) são: ', train_x.shape)
print('As dimensões do conjunto de dados de treinamento (outputs) são: ', train_y.shape)
print('As dimensões do conjunto de dados de teste (inputs) são: ', test_x.shape)
print('As dimensões do conjunto de dados de teste (outputs) são: ', test_y.shape)

# Visualizando os dados
print(data.head())

# 3. Criar uma árvore de decisão
X = data.iloc[:, :-1]  # Todas as colunas, exceto a última
y = data['Classe']  # A última coluna

# Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Criando o classificador
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Plotando a árvore de decisão
plt.figure(figsize=(12, 8))
plot_tree(clf, filled=True, feature_names=X.columns, class_names=np.unique(y).astype(str))  # Garantindo que class_names sejam strings
plt.title("Árvore de Decisão")
plt.show()

# Gerar boxplot da área
plt.figure(figsize=(10, 6))
sns.boxplot(x='Classe', y='Área', data=data)
plt.title("Boxplot da Área por Classe")
plt.show()

# Gerar boxplot da perímetro
plt.figure(figsize=(10, 6))
sns.boxplot(x='Classe', y='Perímetro', data=data)
plt.title("Boxplot da Perímetro por Classe")
plt.show()

# Gerar boxplot da compacidade
plt.figure(figsize=(10, 6))
sns.boxplot(x='Classe', y='Compacidade', data=data)
plt.title("Boxplot da Compacidade por Classe")
plt.show()

# Gerar boxplot da comprmento do grão
plt.figure(figsize=(10, 6))
sns.boxplot(x='Classe', y='Comprimento do Grão', data=data)
plt.title("Boxplot da Comprimento do Grão por Classe")
plt.show()

# Gerar boxplot da largura do grão
plt.figure(figsize=(10, 6))
sns.boxplot(x='Classe', y='Largura do Grão', data=data)
plt.title("Boxplot da Largura do Grão por Classe")
plt.show()

# Gerar boxplot da coeficiente de assimetria
plt.figure(figsize=(10, 6))
sns.boxplot(x='Classe', y='Coeficiente de Assimetria', data=data)
plt.title("Boxplot da Coeficiente de Assimentria por Classe")
plt.show()

# Gerar boxplot da comprimento do sulco do grão
plt.figure(figsize=(10, 6))
sns.boxplot(x='Classe', y='Comprimento do Sulco do Grão', data=data)
plt.title("Boxplot da Comprimento do Sulco do Grão por Classe")
plt.show()

# Realizar clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Adicionando os clusters ao DataFrame
data['Cluster'] = clusters

# Visualizar os clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Área', y='Perímetro', hue='Cluster', data=data, palette='viridis')
plt.title("Clusters de Sementes")
plt.show()

# 6. Gerar uma matriz de confusão
y_pred = clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred, labels=np.unique(y))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y))
disp.plot(cmap='Blues')
plt.title("Matriz de Confusão")
plt.show()

# 7. Mostrar a variedade testada
variedades = data['Classe'].unique()
print("Variedades testadas:")
print(variedades)

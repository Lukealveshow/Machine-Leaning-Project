# Importação de bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report

# Configuração para exibição de gráficos
%matplotlib inline

# Leitura do dataset
dataset = pd.read_csv('/content/venture_capital.csv')

# Instalação e utilização da biblioteca dataprep para análise exploratória
!pip install dataprep
from dataprep.eda import create_report
create_report(dataset)

# Exibição do dataset
dataset

# Remoção da coluna 'Startup'
dataset = dataset.drop(columns='Startup')

# Separação das variáveis independentes (X) e dependentes (y)
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

# Divisão dos dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

# Padronização dos dados
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Implementação do K-Nearest Neighbors (KNN)
knn_model = KNeighborsClassifier(n_neighbors=30)
knn_model.fit(X_train, y_train)

# Predições com o KNN
y_pred_knn = knn_model.predict(X_test)

# Avaliação do modelo KNN
print('\n-------------------------\n')
print('Valores reais com Dados de teste: \n', y_test)
print('\n-------------------------\n')
print('Valores Preditivos com a Máquina Preditiva com Dados de Teste: \n', y_pred_knn)
print('Acurácia da Máquina (KNN) = ', metrics.accuracy_score(y_test, y_pred_knn) * 100)

# Matriz de confusão e relatório de classificação para o KNN
cm_knn = confusion_matrix(y_test, y_pred_knn)
cr_knn = classification_report(y_test, y_pred_knn)
print(cm_knn)
print(cr_knn)

# Implementação do Support Vector Classifier (SVC)
svc_model = SVC(kernel='rbf', random_state=7)
svc_model.fit(X_train, y_train)

# Predições com o SVC
y_pred_svc = svc_model.predict(X_test)

# Avaliação do modelo SVC
print('\n---------------\n')
print('Valores Predetidos para o conjunto de Testes (SVC): \n', y_pred_svc)
print('Acurácia da Máquina (SVC) = ', metrics.accuracy_score(y_test, y_pred_svc) * 100)

# Comparação entre valores atuais e preditos
print('\nAtual vs Predito \n-------------------------\n')
error_df = pd.DataFrame({'Atual': y_test, 'Predito': y_pred_svc})
print(error_df)

# Matriz de confusão e relatório de classificação para o SVC
cm_svc = confusion_matrix(y_test, y_pred_svc)
cr_svc = classification_report(y_test, y_pred_svc)
print(cm_svc)
print(cr_svc)

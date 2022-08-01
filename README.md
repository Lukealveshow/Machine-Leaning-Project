# Importando Bibliotecas Essenciais
import pandas as pd                                       
import numpy as np                                       
import matplotlib.pyplot as plt                           
import seaborn as sns                                    
from sklearn.preprocessing import StandardScaler          
from sklearn.model_selection import train_test_split      
from sklearn.neighbors import KNeighborsClassifier        
from sklearn.svm import SVC                               
from sklearn import metrics                            
%matplotlib inline

dataset = pd.read_csv('/content/venture_capital.csv')
dataset
!pip install dataprep
from dataprep.eda import create_report
create_report(dataset)
dataset
dataset = dataset.drop(columns='Startup')
dataset
X = dataset.iloc[:,:-1].values 
y = dataset.iloc[:,-1].values
y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 7)
X_train
X_test
y_train
y_test
dataset.describe()
from sklearn.preprocessing import StandardScaler # Função de Padronização para deixar os Dados na mesma Escala
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_train
X_test
from sklearn.neighbors import KNeighborsClassifier
Maquina_Preditiva = KNeighborsClassifier(n_neighbors=30)
Maquina_Preditiva
Maquina_Preditiva = Maquina_Preditiva.fit(X_train, y_train)
y_pred = Maquina_Preditiva.predict(X_test) 
print('\n-------------------------\n')
print('Valores reais com Dados de teste: \n', y_test)
print('\n-------------------------\n')
print('Valores Preditivos com a Máquina Preditiva com Dados de Teste: \n', y_pred)
from sklearn import metrics
print("Acurácia da Máquina = ", metrics.accuracy_score(y_test, y_pred) *100)
print('\nReal vs Predito \n---------------------\n')
error_df = pd.DataFrame({'Real': y_test,'Predito': y_pred})
error_df
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import metrics

print('Acurácia da Máquina = ', metrics.accuracy_score(y_test, y_pred) * 100)
cm = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred)
print(cm)
print(cr)
from sklearn.svm import SVC
Maquina_Preditiva = SVC(kernel='rbf',random_state=7)
Maquina_Preditiva
Maquina_Preditiva.fit(X_train, y_train)
y_pred = Maquina_Preditiva.predict(X_test)
print('\n---------------\n')
print('Valores Predetidos para o conjunto de Testes: \n', y_pred)
print('\n---------------\n')
print('Valores Atuais para o conjunto de Testes: \n', y_pred)
from sklearn import metrics
print('Prediction Accuracy: ', metrics.accuracy_score(y_test, y_pred))

# Comparando valores Reais com Preditos pela Máquina
print('\nAtual vs Predito \n-------------------------\n')
error_df = pd.DataFrame({'Atual': y_test, 'Predito': y_pred})
error_df
from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred)
print(cm)
print(cr)

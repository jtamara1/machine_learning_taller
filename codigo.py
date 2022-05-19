import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from warnings import simplefilter
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve,roc_auc_score, f1_score, precision_score, recall_score,classification_report

from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)

#Se procesa dataset bank
bank = pd.read_csv('./Bank Marketing/bank-full.csv')
bank.job = bank.job.replace(['management', 'technician', 'entrepreneur', 'blue-collar',
       'unknown', 'retired', 'admin.', 'services', 'self-employed',
       'unemployed', 'housemaid', 'student'], [0,1,2,3,4,5,6,7,8,9,10,11])
bank.marital = bank.marital.replace(['married', 'single', 'divorced'], [0,1,2])
bank.education = bank.education.replace(['tertiary', 'secondary', 'unknown', 'primary'], [0,1,2,3])
bank.default = bank.default.replace(["no","yes"],[0,1])
bank.housing = bank.housing.replace(["no","yes"],[0,1])
bank.loan = bank.loan.replace(["no","yes"],[0,1])
bank.contact = bank.contact.replace(['unknown', 'cellular', 'telephone'],[0,1,2])
bank.month = bank.month.replace(['may', 'jun', 'jul', 'aug', 'oct', 'nov', 'dec', 'jan', 'feb',
       'mar', 'apr', 'sep'], [0,1,2,3,4,5,6,7,8,9,10,11])
bank.poutcome = bank.poutcome.replace(['unknown', 'failure', 'other', 'success'],[0,1,2,3])
bank.y = bank.y.replace(["no","yes"],[0,1])
rangos = [0, 8, 15, 18, 25, 40, 60, 100]
nombres = ['1', '2', '3', '4', '5', '6', '7']
bank.age = pd.cut(bank.age, rangos, labels=nombres)
bank.dropna(axis=0,how='any', inplace=True)

#Se procesa dataset weather
weather = pd.read_csv("./weatherAUS/weatherAUS.csv")
weather = weather.drop(['Date'], 1)
weather.Location = weather.Location.replace(['Albury', 'BadgerysCreek', 'Cobar', 'CoffsHarbour', 'Moree', 'Newcastle', 'NorahHead', 'NorfolkIsland', 'Penrith', 'Richmond', 'Sydney', 'SydneyAirport', 'WaggaWagga', 'Williamtown', 'Wollongong', 'Canberra', 'Tuggeranong', 'MountGinini', 'Ballarat', 'Bendigo', 'Sale', 'MelbourneAirport', 'Melbourne', 'Mildura', 'Nhil', 'Portland', 'Watsonia', 'Dartmoor', 'Brisbane', 'Cairns', 'GoldCoast', 'Townsville', 'Adelaide', 'MountGambier', 'Nuriootpa', 'Woomera', 'Albany', 'Witchcliffe', 'PearceRAAF', 'PerthAirport', 'Perth', 'SalmonGums', 'Walpole', 'Hobart', 'Launceston', 'AliceSprings', 'Darwin', 'Katherine', 'Uluru'] , [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48])
weather.WindGustDir = weather.WindGustDir.replace(['SSW', 'S', 'NNE', 'WNW', 'N', 'SE', 'ENE', 'NE', 'E', 'SW', 'W', 'WSW', 'NNW', 'ESE', 'SSE', 'NW'], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
weather.WindDir9am = weather.WindDir9am.replace(['ENE', 'SSE', 'NNE', 'WNW', 'NW', 'N', 'S', 'SE', 'NE', 'W', 'SSW', 'E', 'NNW', 'ESE', 'WSW', 'SW'],[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
weather.WindDir3pm = weather.WindDir3pm.replace(['SW', 'SSE', 'NNW', 'WSW', 'WNW', 'S', 'ENE', 'N', 'SE', 'NNE', 'NW', 'E', 'ESE', 'NE', 'SSW', 'W'],[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
weather.RainToday = weather.RainToday.replace(['No', 'Yes'],[0, 1])
weather.RainTomorrow = weather.RainTomorrow.replace(['No', 'Yes'],[0, 1])
weather.dropna(axis=0,how='any', inplace=True)

def metricas(modelo, nombre, dataset):
    print("*"*50)
    print(f"MODELO {nombre}")
    kfold = KFold(n_splits=10)
    cvscores = [] 
    for train, test in kfold.split(x_train, y_train):
        modelo.fit(x_train[train], y_train[train])
        scores = modelo.score(x_train[test], y_train[test])
        cvscores.append(scores)
    y_pred = modelo.predict(x_test)
    accuracy_entrenamiento = accuracy_score(modelo.predict(x_train), y_train)
    accuracy_validation = np.mean(cvscores)
    accuracy_test = accuracy_score(y_pred, y_test)
    matriz_confusion = confusion_matrix(y_test, y_pred)
    F1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    actual = pd.DataFrame({"modelo":[nombre],
                           "accuracy_validacion":[accuracy_validation],
                           "accuracy_test":[accuracy_test],
                           "accuracy_entrenamiento":[accuracy_entrenamiento],
                           "recall":[recall],
                           "precision":[precision],
                           "F1":[F1]})
    
    #Punto 3
    print("Punto 3")
    print(classification_report(y_test, y_pred))
    #Punto 4
    print("Punto 4")
    print(actual)
    #Punto 5
    heatmap = sns.heatmap(matriz_confusion)
    fig = heatmap.get_figure()
    print(heatmap)
    fig.savefig(f"heatmap_{nombre}_{dataset}.png")
    plt.clf()
    #Punto 6
    print("Punto 6")
    print("Y test")
    print(y_test)
    print("Y pred")
    print(y_pred)
    return accuracy_validation, accuracy_test, y_pred, accuracy_entrenamiento, matriz_confusion

random = RandomForestClassifier()
knn = KNeighborsClassifier()
arbol = DecisionTreeClassifier()
ada = AdaBoostClassifier()
probabilistico = GaussianNB()

data_train = bank[:36168]
data_test = bank[36168:]

x = np.array(data_train.drop(['y'], 1))
y = np.array(data_train.y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

x_test_out = np.array(data_test.drop(['y'], 1))
y_test_out = np.array(data_test.y)

print("Dataset bank")

metricas(arbol, "Arbol de decisión", "bank")
metricas(ada, "Adaboost","bank")
metricas(random, "Random Forest", "bank")
metricas(probabilistico, "Gaussian Naive Bayes", "bank")
metricas(knn, "KNN", "bank")

random = RandomForestClassifier()
knn = KNeighborsClassifier()
arbol = DecisionTreeClassifier()
ada = AdaBoostClassifier()
probabilistico = GaussianNB()

data_train = weather[:45136]
data_test = weather[45136:]

x = np.array(data_train.drop(['RainTomorrow'], 1))
y = np.array(data_train.RainTomorrow)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

x_test_out = np.array(data_test.drop(['RainTomorrow'], 1))
y_test_out = np.array(data_test.RainTomorrow)

print("Dataset bank")

metricas(arbol, "Arbol de decisión", "weather")
metricas(ada, "Adaboost", "weather")
metricas(random, "Random Forest", "weather")
metricas(probabilistico, "Gaussian Naive Bayes", "weather")
metricas(knn, "KNN", "weather")

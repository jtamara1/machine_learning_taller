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
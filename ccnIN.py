import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error
from math import sqrt

# Keras specific
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense

with open('INDIA_ALL_CL.data', 'r') as f:
    s = f.read()
with open('INDIA_ALL_CL.data', 'w') as f:
    s = s.replace(',\n', '\n').replace(',\r\n', '\r\n')
    f.write(s)

with open('INDIA_ALL_COTTON.data', 'r') as f:
    s = f.read()
with open('INDIA_ALL_COTTON.data', 'w') as f:
    c = s.replace(',\n', '\n').replace(',\r\n', '\r\n')
    f.write(c)

with open('INDIA_ALL_OILCROPS.data', 'r') as f:
    s = f.read()
with open('INDIA_ALL_OILCROPS.data', 'w') as f:
    c = s.replace(',\n', '\n').replace(',\r\n', '\r\n')
    f.write(c)

dataCL = pd.read_csv('INDIA_ALL_CL.data')
dataCL = pd.DataFrame(dataCL)
dataCL.drop('year', axis=1)
dataCL.insert(loc=0, column='row_num', value=np.arange(len(dataCL)))

# dataCL.drop('TCI', axis=1)

dataCOTT = pd.read_csv('INDIA_ALL_COTTON.data')
dataCOTT = pd.DataFrame(dataCOTT)
dataCOTT.drop('year', axis=1)
dataCOTT.insert(loc=0, column='row_num', value=np.arange(len(dataCOTT)))
meansTC = dataCOTT['CTCI']

dataOIL = pd.read_csv('INDIA_ALL_OILCROPS.data')
dataOIL = pd.DataFrame(dataOIL)
dataOIL.drop('year', axis=1)
dataOIL.insert(loc=0, column='row_num', value=np.arange(len(dataOIL)))
meansTC = dataOIL['OTCI']

df6 = dataCOTT.merge(dataOIL, how='left')
#dataCL.merge(dataCOTT, how='left').merge(dataOIL, how='left')
print(df6)

droughtLevel = []
for x in meansTC:
    if 0 <= x < 6:
        droughtLevel.append(5)
    elif 6 <= x < 16:
        droughtLevel.append(4)
    elif 16 <= x < 26:
        droughtLevel.append(3)
    elif 26 <= x < 35:
        droughtLevel.append(2)
    elif 35 <= x <= 40:
        droughtLevel.append(1)
    elif x > 40:
        droughtLevel.append(0)

df = df6.drop('row_num', axis=1)
dfCL = dataCL.drop('row_num', axis=1)
# dfCL1 = dataCL.drop('TCI', axis=1)

X = dfCL
print(X)

y = pd.DataFrame(droughtLevel, columns=['droughtLevel'])
print(X.head())
print(y.head())

print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.450, random_state=40)
print(X_train.dtypes)
print(X_test.dtypes)
print(y_train.dtypes)
print(y_test.dtypes)

mlp = MLPClassifier(hidden_layer_sizes=(8, 8, 8), activation='relu', solver='adam', max_iter=1500)
mlp.fit(X_train, y_train)

predict_train = mlp.predict(X_train)
predict_test = mlp.predict(X_test)

print(confusion_matrix(y_train, predict_train))
print(classification_report(y_train, predict_train))

# model = Sequential()
# model.add(Dense(50, activation='relu', input_dim=7))
# model.add(Dense(40, activation='relu'))
# model.add(Dense(30, activation='relu'))
# model.add(Dense(20, activation='relu'))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(5, activation='relu'))
# model.add(Dense(6, activation='softmax'))
#
# # Compile the model
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])
#
# model.fit(X_train, y_train, epochs=5320)
#
# pred_train = model.predict(X_train)
# print(pred_train)
# print(np.sqrt(mean_squared_error(y_train, pred_train)))

# pred = model.predict(X_test)
# print(np.sqrt(mean_squared_error(y_test, pred)))

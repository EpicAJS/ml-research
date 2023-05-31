import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

with open('AFRICA_CL.data', 'r') as f:
    s = f.read()
with open('AFRICA_CL.data', 'w') as f:
    s = s.replace(',\n', '\n').replace(',\r\n', '\r\n')
    f.write(s)

dataCL = pd.read_csv('AFRICA_CL.data')
dataCL = pd.DataFrame(dataCL)
dataCL.drop('year', axis=1)
dataCL.insert(loc=0, column='row_num', value=np.arange(len(dataCL)))

meansTC = dataCL['VCI']

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

df = dataCL.drop('VCI', axis=1)

X = df
print(X)

y = pd.DataFrame(droughtLevel, columns=['droughtLevel'])

print(X.head())
print(y.head())

print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = svm.SVC()
model.fit(X_train, y_train)

print(model)

predictions = model.predict(X_test)
acc = accuracy_score(y_test, predictions)

print('predictions: ', predictions)
print('actual: ', y_test)
print('accuracy: ', acc)

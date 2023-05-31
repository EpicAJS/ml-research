from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

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

X = dataOIL
print(X)

y = pd.DataFrame(droughtLevel, columns=['droughtLevel'])
print(X.head())
print(y.head())

print(X.shape)
print(y.shape)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

neighbors = np.arange(1, 20)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

# Loop over K values
for i, k in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    # Compute training and test data accuracy
    train_accuracy[i] = knn.score(X_train, y_train)
    test_accuracy[i] = knn.score(X_test, y_test)

# Generate plot
plt.plot(neighbors, test_accuracy, label='Testing dataset Accuracy')
plt.plot(neighbors, train_accuracy, label='Training dataset Accuracy')
plt.xticks(range(0, 21))

plt.legend()
plt.xlabel('n_neighbors')
plt.ylabel('Accuracy')
plt.show()

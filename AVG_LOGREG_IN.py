import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from matplotlib import pyplot as plt

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

meansTC = dataCL['TCI']
dataCL.drop('TCI', axis=1)

dataCOTT = pd.read_csv('INDIA_ALL_COTTON.data')
dataCOTT = pd.DataFrame(dataCOTT)
dataCOTT.drop('year', axis=1)
dataCOTT.insert(loc=0, column='row_num', value=np.arange(len(dataCOTT)))

dataOIL = pd.read_csv('INDIA_ALL_OILCROPS.data')
dataOIL = pd.DataFrame(dataOIL)
dataOIL.drop('year', axis=1)
dataOIL.insert(loc=0, column='row_num', value=np.arange(len(dataOIL)))

df6 = dataCL.merge(dataCOTT, how='left').merge(dataOIL, how='left')
print(df6)

droughtLevel = []
for x in meansTC:
    if 0 <= x < 6:
        droughtLevel.append("Exceptional")
    elif 6 <= x < 16:
        droughtLevel.append("Extreme")
    elif 16 <= x < 26:
        droughtLevel.append("Severe")
    elif 26 <= x < 35:
        droughtLevel.append("Moderate")
    elif 35 <= x <= 40:
        droughtLevel.append("Abnormally Dry")
    elif x > 40:
        droughtLevel.append("No Drought")

df = df6.drop('row_num', axis=1)
X = dataCL

y = pd.DataFrame(droughtLevel, columns=['droughtLevel'])


print(X.head())
print(y.head())

print(X.shape)
print(y.shape)

model = LogisticRegression()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# train
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Predictions: ", predictions)
print("Accuracy: ", model.score(X_test, y_test))
print(model.predict_proba(X_test))

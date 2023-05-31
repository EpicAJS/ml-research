from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
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

df6 = dataCL.merge(dataCOTT, how='left')#.merge(dataOIL, how='left')
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

df7 = df6.drop('row_num', axis=1)
X = dataCL

y = pd.DataFrame(droughtLevel, columns=['droughtLevel'])


print(X.head())
print(y.head())

print(X.shape)
print(y.shape)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

m_depth = np.arange(1, 32)
train_accuracy = np.empty(len(m_depth))
test_accuracy = np.empty(len(m_depth))

# Loop over K values
for i, k in enumerate(m_depth):
    clf = RandomForestClassifier(n_estimators=51, max_depth=k)
    clf.fit(X_train, y_train)

    # Compute training and test data accuracy
    train_accuracy[i] = clf.score(X_train, y_train)
    test_accuracy[i] = clf.score(X_test, y_test)

# Generate plot
plt.plot(m_depth, test_accuracy, label='Testing dataset Accuracy')
plt.plot(m_depth, train_accuracy, label='Training dataset Accuracy')

plt.legend()
plt.xlabel('max_depth')
plt.ylabel('Accuracy')
plt.show()

#
# # Training the model on the training dataset
# # fit function is used to train the model using the training sets as parameters
# clf.fit(X_train, y_train)
#
# # performing predictions on the test dataset
# y_pred = clf.predict(X_test)
#
# print()
#
# # using metrics module for accuracy calculation
# print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred))
#
# print(clf.predict([[10.5, 15, 30, 18]]))
#
# #feature_imp = pd.Series(clf.feature_importances_,).sort_values(ascending = False)
# #feature_imp

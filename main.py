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

# Remove commas from data
with open('USA_TCI_CL.txt', 'r') as f:
    s = f.read()
with open('USA_TCI_CL.txt', 'w') as f:
    c = s.replace(',\n', '\n').replace(',\r\n', '\r\n')
    f.write(c)

with open('USA_VCI_CL.data', 'r') as f:
    s = f.read()
with open('USA_VCI_CL.data', 'w') as f:
    c = s.replace(',\n', '\n').replace(',\r\n', '\r\n')
    f.write(c)

with open('USA_VHI_CL.data', 'r') as f:
    s = f.read()
with open('USA_VHI_CL.data', 'w') as f:
    c = s.replace(',\n', '\n').replace(',\r\n', '\r\n')
    f.write(c)

dataVC = pd.read_csv('USA_VCI_CL.data')
dataVC = pd.DataFrame(dataVC)
dataVC = dataVC.drop('year', axis=1)
dataVC = dataVC.drop('week', axis=1)

dataVC.to_numpy()
print(dataVC)

meanVC = []
avgVC = 0
for col, row in dataVC.iterrows():
    print(f"Index: {col}")
    avgVC = ((float(f"{row['vh1']}") / 100) * 1) + ((float(f"{row['vh2']}") / 100) * 6) + \
            ((float(f"{row['vh3']}") / 100) * 11) + ((float(f"{row['vh4']}") / 100) * 16) + \
            ((float(f"{row['vh5']}") / 100) * 21) + ((float(f"{row['vh6']}") / 100) * 26) + \
            ((float(f"{row['vh7']}") / 100) * 31) + ((float(f"{row['vh8']}") / 100) * 36) + \
            ((float(f"{row['vh9']}") / 100) * 41) + ((float(f"{row['vh10']}") / 100) * 46) + \
            ((float(f"{row['vh11']}") / 100) * 51) + ((float(f"{row['vh12']}") / 100) * 56) + \
            ((float(f"{row['vh13']}") / 100) * 61) + ((float(f"{row['vh14']}") / 100) * 66) + \
            ((float(f"{row['vh15']}") / 100) * 71) + ((float(f"{row['vh16']}") / 100) * 76) + \
            ((float(f"{row['vh17']}") / 100) * 81) + ((float(f"{row['vh18']}") / 100) * 86) + \
            ((float(f"{row['vh19']}") / 100) * 91) + ((float(f"{row['vh20']}") / 100) * 96) + \
            ((float(f"{row['vh21']}") / 100) * 101)
    print(avgVC)
    meanVC.append(avgVC)
print(meanVC)

dataTC = pd.read_csv('USA_TCI_CL.txt')
dataTC = pd.DataFrame(dataTC)
dataTC = dataTC.drop('year', axis=1)
dataTC = dataTC.drop('week', axis=1)

dataTC.to_numpy()
print(dataTC)

meanTC = []
avgTC = 0
for col, row in dataTC.iterrows():
    print(f"Index: {col}")
    avgTC = ((float(f"{row['vh1']}") / 100) * 1) + ((float(f"{row['vh2']}") / 100) * 6) + \
            ((float(f"{row['vh3']}") / 100) * 11) + ((float(f"{row['vh4']}") / 100) * 16) + \
            ((float(f"{row['vh5']}") / 100) * 21) + ((float(f"{row['vh6']}") / 100) * 26) + \
            ((float(f"{row['vh7']}") / 100) * 31) + ((float(f"{row['vh8']}") / 100) * 36) + \
            ((float(f"{row['vh9']}") / 100) * 41) + ((float(f"{row['vh10']}") / 100) * 46) + \
            ((float(f"{row['vh11']}") / 100) * 51) + ((float(f"{row['vh12']}") / 100) * 56) + \
            ((float(f"{row['vh13']}") / 100) * 61) + ((float(f"{row['vh14']}") / 100) * 66) + \
            ((float(f"{row['vh15']}") / 100) * 71) + ((float(f"{row['vh16']}") / 100) * 76) + \
            ((float(f"{row['vh17']}") / 100) * 81) + ((float(f"{row['vh18']}") / 100) * 86) + \
            ((float(f"{row['vh19']}") / 100) * 91) + ((float(f"{row['vh20']}") / 100) * 96) + \
            ((float(f"{row['vh21']}") / 100) * 101)
    print(avgTC)
    meanTC.append(avgTC)
print(meanTC)

conditions = [meanVC, meanTC]
conditions = np.array(conditions).reshape(2057, 2)

df = pd.DataFrame(conditions, columns=['meansVC', 'meansTC'])

print(df)
df['indexAvg'] = df.iloc[:, 0:1].mean(axis=1)

dataVH = pd.read_csv('USA_VHI_CL.data')
dataVH = pd.DataFrame(dataVH)
dataVH = dataVH.drop('weekVH', axis=1)
dataVH = dataVH.drop('year', axis=1)

dataVH.to_numpy()
print(dataVH)

meanVH = []
avgVH = 0
for col, row in dataVH.iterrows():
    print(f"Index: {col}")
    avgVH = ((float(f"{row['vh1']}") / 100) * 1) + ((float(f"{row['vh2']}") / 100) * 6) + \
            ((float(f"{row['vh3']}") / 100) * 11) + ((float(f"{row['vh4']}") / 100) * 16) + \
            ((float(f"{row['vh5']}") / 100) * 21) + ((float(f"{row['vh6']}") / 100) * 26) + \
            ((float(f"{row['vh7']}") / 100) * 31) + ((float(f"{row['vh8']}") / 100) * 36) + \
            ((float(f"{row['vh9']}") / 100) * 41) + ((float(f"{row['vh10']}") / 100) * 46) + \
            ((float(f"{row['vh11']}") / 100) * 51) + ((float(f"{row['vh12']}") / 100) * 56) + \
            ((float(f"{row['vh13']}") / 100) * 61) + ((float(f"{row['vh14']}") / 100) * 66) + \
            ((float(f"{row['vh15']}") / 100) * 71) + ((float(f"{row['vh16']}") / 100) * 76) + \
            ((float(f"{row['vh17']}") / 100) * 81) + ((float(f"{row['vh18']}") / 100) * 86) + \
            ((float(f"{row['vh19']}") / 100) * 91) + ((float(f"{row['vh20']}") / 100) * 96) + \
            ((float(f"{row['vh21']}") / 100) * 101)
    print(avgVH)
    meanVH.append(avgVH)
print(meanVH)

droughtLevel = []
for x in meanVH:
    if 0 <= x < 6:
        droughtLevel.append(5)
    if 6 <= x < 16:
        droughtLevel.append(4)
    if 16 <= x < 26:
        droughtLevel.append(3)
    if 26 <= x < 35:
        droughtLevel.append(2)
    if 35 <= x <= 40:
        droughtLevel.append(1)
    if x > 40:
        droughtLevel.append(0)

X = df

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

mlp = MLPClassifier(hidden_layer_sizes=(18, 18, 18), activation='relu', solver='adam', max_iter=1000)
mlp.fit(X_train, y_train)

predict_train = mlp.predict(X_train)
predict_test = mlp.predict(X_test)

print(confusion_matrix(y_train, predict_train))
print(classification_report(y_train, predict_train))

# model = Sequential()
# model.add(Dense(500, activation='relu', input_dim=3))
# model.add(Dense(100, activation='relu'))
# model.add(Dense(50, activation='relu'))
# model.add(Dense(3, activation='softmax'))
#
# # Compile the model
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])
#
# model.fit(X_train, y_train, epochs=20)
#
# pred_train = model.predict(X_train)
# print(pred_train)
# print(np.sqrt(mean_squared_error(y_train, pred_train)))

# pred = model.predict(X_test)
# print(np.sqrt(mean_squared_error(y_test, pred)))

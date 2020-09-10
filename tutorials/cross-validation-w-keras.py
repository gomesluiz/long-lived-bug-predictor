import pandas as pd 
import os 
import numpy as np
from scipy.stats import zscore 

from sklearn import metrics
from sklearn.model_selection    import train_test_split
from sklearn.model_selection    import RepeatedKFold

from tensorflow.keras.models    import Sequential
from tensorflow.keras.layers    import Dense, Activation

df = pd.read_csv(
    "https://data.heatonresearch.com/data/t81-558/jh-simple-dataset.csv",
    na_values=['NA','?'])

# generate dummies for job
df = pd.concat([df, pd.get_dummies(df['job'], prefix="job")], axis=1)
df.drop('job', axis=1, inplace=True)

# generate dummies for area
df = pd.concat([df, pd.get_dummies(df['area'], prefix="area")], axis=1)
df.drop('area', axis=1, inplace=True)

# missing values for income
med = df['income'].median()
df['income'] = df['income'].fillna(med)

df['income'] = zscore(df['income'])
df['aspect'] = zscore(df['aspect'])
df['save_rate'] = zscore(df['save_rate'])
df['age'] = zscore(df['age'])
df['subscriptions'] = zscore(df['subscriptions'])

x_columns = df.columns.drop('product').drop('id')
x = df[x_columns].values 
dummies = pd.get_dummies(df['product'])
products = dummies.columns
y = dummies.values 

rkf = RepeatedKFold(n_splits=5, n_repeats=2, random_state=42)

oos_y = []
oos_pred = []
fold = 0

for train, test in rkf.split(x, df['product']):
    fold+=1
    print(f"Fold #{fold}")

    x_train = x[train]
    y_train = y[train]
    x_test  = x[test]
    y_test  = y[test]

    model = Sequential()
    model.add(Dense(50, input_dim=x.shape[1], activation='relu'))
    model.add(Dense(25, activation='relu'))
    model.add(Dense(y.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    model.fit(x_train, y_train, validation_data=(x_test, y_test), verbose=0, epochs=20)
    
    pred = model.predict(x_test)
    oos_y.append(y_test)

    pred = np.argmax(pred, axis=1)
    oos_pred.append(pred)

    y_compare = np.argmax(y_test, axis=1)
    score = metrics.accuracy_score(y_compare, pred)
    print(f"Fold score (accuracy) : {score}")

oos_y = np.concatenate(oos_y)
oos_pred = np.concatenate(oos_pred)
oos_y_compare = np.argmax(oos_y, axis=1)

score =  metrics.accuracy_score(oos_y_compare, oos_pred)
print(f"Final Score (accuracy): {score}")

oos_y = pd.DataFrame(oos_y)
oos_pred = pd.DataFrame(oos_pred)
oosDF = pd.concat([df, oos_y, oos_pred], axis=1)

# -*- coding: utf-8 -*-



import numpy as np
import pandas as pd

input_dir = ".\\"

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
#%matplotlib inline
color = sns.color_palette()

df = pd.read_csv('properties_2016.csv')
train_df = pd.read_csv("train_2016_v2.csv", parse_dates=["transactiondate"])
#train_df["transactiondate_dateformat"] = pd.to_datetime(train_df["transactiondate"])
features_name_dict = pd.read_excel("zillow_data_dictionary.xlsx")
df.dtypes

#df.to_html("temp.html")
print train_df.shape
print df.shape




plt.hist(df.finishedsquarefeet15.value_counts())

sid = df.finishedsquarefeet15.value_counts()
#explore properry data

print df.shape[0]

df2 = df[["finishedsquarefeet15", "lotsizesquarefeet", "bedroomcnt"]]

df2.corr()

plt.figure(figsize=(8,6))

sns.jointplot(df.finishedsquarefeet15, df.lotsizesquarefeet)
#sns.jointplot(x="finishedsquarefeet15", y="lotsizesquarefeet",data=df2, kind="reg")


##Convert all non numeric to numeric values
df = df.apply(pd.to_numeric, errors='coerce')
#df[~df.applymap(np.isreal).all(1)]
#np.argmin(df.applymap(np.isreal).all(1))


#explore architectural style

X = df.iloc[:,:].as_matrix()
X4reshaped = X[:,4].reshape(-1,1)
from sklearn.preprocessing import Imputer
imputer  = Imputer(missing_values="NaN", strategy="mean",axis=0)
#imputer  = imputer.fit(df.loc[df["bedroomcnt"]].values.reshape(-1,1)) 
imputer  = imputer.fit(X4reshaped) 
X4reshaped = imputer.fit_transform(X4reshaped)

plt.figure(figsize=(8,6))
plt.scatter(range(df.shape[0]), (df.finishedsquarefeet15.values))
plt.xlabel('index', fontsize=12)
plt.ylabel('Area', fontsize=12)
plt.show()

plt.figure(figsize=(8,6))
plt.scatter(range(df.shape[0]), (df.lotsizesquarefeet.values))
plt.xlabel('index', fontsize=12)
plt.ylabel('Area', fontsize=12)
plt.show()





# trying to use imputer
X = df.iloc[:,:].as_matrix()

X15reshaped = X[:,15].reshape(-1,1)
from sklearn.preprocessing import Imputer
imputer  = Imputer(missing_values="NaN", strategy="median",axis=0)
#imputer  = imputer.fit(df.loc[df["bedroomcnt"]].values.reshape(-1,1)) 
imputer  = imputer.fit(X15reshaped) 
X15reshaped = imputer.fit_transform(X15reshaped)
df['finishedsquarefeet15'] = X15reshaped

df2 = df['finishedsquarefeet15']
print df2.max()
df2 = df2.values.reshape(-1,1)

from sklearn.preprocessing import Imputer
imputer  = Imputer(missing_values="NaN", strategy="mean",axis=0)
imputer = imputer.fit(df2)
df2 = imputer.transform(df2)

print df2.max()
plt.plot(df2[:,0])




##Train-df eda

plt.figure(figsize=(8,6))
plt.scatter(range(train_df.shape[0]), np.sort(train_df.logerror.values))
plt.xlabel('index', fontsize=12)
plt.ylabel('logerror', fontsize=12)
plt.show()

ulimit = np.percentile(train_df.logerror.values, 99)
llimit = np.percentile(train_df.logerror.values, 1)
train_df['logerror'].ix[train_df['logerror']>ulimit] = ulimit
train_df['logerror'].ix[train_df['logerror']<llimit] = llimit
plt.figure(figsize=(12,8))
sns.distplot(train_df.logerror.values, bins=100)
plt.xlabel('logerror', fontsize=12)
plt.show()

train_df['transaction_month'] = train_df['transactiondate'].dt.month

cnt_srs = train_df['transaction_month'].value_counts()
plt.figure(figsize=(12,6))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color=color[3])
plt.xticks(rotation='vertical')
plt.xlabel('Month of transaction', fontsize=12)
plt.ylabel('Number of Occurrences', fontsize=12)
plt.show()






#explore the property data

cnt_srs = df['bedroomcnt'].value_counts()

plt.figure(figsize=(12,6))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color=color[2])
plt.xticks(rotation='vertical')
plt.xlabel('Bedroom counts', fontsize=12)
plt.ylabel('Number of Occurrences', fontsize=12)
plt.show()


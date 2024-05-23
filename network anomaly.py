#!/usr/bin/env python
# coding: utf-8

# ### Import Lıbrarıes

# In[18]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


# In[2]:


# read the data
data = pd.read_csv("../dataset/network_anomaly_detection/all_data (3).csv")


# In[3]:


data.head()


# In[4]:


data.tail()


# In[5]:


data.info()


# In[6]:


# describe and null control
data.describe()


# In[7]:


data.isna().sum()


# In[8]:


data.duplicated().sum()


# In[10]:


# lets check the data EDA
plt.figure(figsize=(12, 7))
sns.countplot(data=data, x="class")


# In[20]:


# data label transform
process_data = data.copy()


# In[24]:


encoder = LabelEncoder()
process_data["class"] = encoder.fit_transform(process_data["class"])


# In[28]:


process_data["class"].value_counts()


# In[32]:


# correlation
plt.figure(figsize=(20, 10))
sns.heatmap(process_data.corr(), annot=True)


# In[43]:


# box plots for high corellation 
process_data.corr()["class"].sort_values(ascending=False)
columns = ["tcpEstabResets", "ipInDiscards", "tcpOutRsts"]


# In[42]:


for i in columns:
    sns.boxplot(data=process_data, x="class", y=i)
    plt.show()


# In[44]:


process_data.shape


# In[92]:


scaler = MinMaxScaler()
sclaed_to_autoencoder = scaler.fit_transform(process_data.iloc[:, :-1])


# In[100]:


# principle component analysis with autoencoder
from keras.layers import Dense, Input
from keras.models import Sequential

# encoder
encoder = Sequential()
encoder.add(Input(shape=(34, )))
encoder.add(Dense(34, activation="relu"))
encoder.add(Dense(17, activation="relu"))
encoder.add(Dense(10, activation="relu"))


# In[101]:


decoder = Sequential()
encoder.add(Dense(17, activation="relu"))
decoder.add(Dense(34, activation="relu"))


# In[102]:


# autoencoder
autoencoder = Sequential([encoder, decoder])


# In[103]:


autoencoder.compile(loss="mse", optimizer="rmsprop", metrics=["mae"])


# In[104]:


autoencoder.fit(sclaed_to_autoencoder, sclaed_to_autoencoder, epochs=20)


# In[110]:


selected = np.argsort(autoencoder.predict(sclaed_to_autoencoder)[0])


# In[111]:


process_data.iloc[1, selected], process_data.corr()["class"].sort_values(ascending=False)


# In[140]:


# new data to train model
new_data = pd.concat([process_data.iloc[:, selected[:10]], process_data.iloc[:, -1]], axis=1)


# In[141]:


new_data.shape


# In[142]:


# machine learning section


# In[143]:


X = new_data.iloc[:, :-1].values
y = new_data.iloc[:, -1].values


# In[144]:


X.shape


# In[145]:


y.shape


# In[183]:


# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)


# In[184]:


X_train.shape, y_train.shape


# In[185]:


X_test.shape, y_test.shape


# In[186]:


# minmax scale
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)


# In[187]:


X_test = scaler.fit_transform(X_test)


# In[188]:


from sklearn.metrics import classification_report
# classification report function
def report(model):
    predicted = model.predict(X_test)
    print(classification_report(y_test, predicted))


# ### KNN

# In[189]:


knn = KNeighborsClassifier(n_jobs=4, n_neighbors=3)
knn.fit(X_train, y_train)


# In[190]:


report(knn)


# ### Random Forests

# In[191]:


# Random forest
rf = RandomForestClassifier(n_estimators=10, max_depth=4, n_jobs=4)
rf.fit(X_train, y_train)


# In[192]:


report(rf)


# ### Logistic regerssion

# In[193]:


lr = LogisticRegression(C=1.0)
lr.fit(X_train, y_train)


# In[194]:


report(lr)


# In[195]:


# # SVC
svc = SVC(C=1.0)
svc.fit(X_train, y_train)


# In[196]:


report(svc)


# In[ ]:





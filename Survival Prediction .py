#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


df = pd.read_csv('data/train.csv')
df_2 = pd.read_csv('data/test.csv')
df_3 = pd.read_csv('data/gender_submission.csv')


# In[ ]:


df_merge_test = pd.merge(df_2, df_3, on="PassengerId") 


# In[ ]:


df_merge_test.head()


# In[ ]:


df.isnull().any()


# In[ ]:


df.count()


# In[ ]:


df=df.drop(['Cabin', 'Ticket', 'Name'],axis=1)


# In[ ]:


df.head()


# In[ ]:


df['Age']=df['Age'].interpolate()


# In[ ]:


df['Age']=df['Age'].fillna(df.Age.median())


# In[ ]:


df.count()


# In[ ]:


df_clean = df.dropna()


# In[ ]:


df_clean.count()


# In[ ]:


df_clean['Sex'].value_counts()
def transform_sex(n):
  if n == 'female':
    return 1
  else:
    return 0

df_clean.head()

#get dummies label encoder 
df_clean['Sex_10'] = df['Sex'].map(transform_sex)


# In[ ]:


df_clean.head()


# In[ ]:


df_clean=df_clean.drop(['Sex'],axis=1)


# In[ ]:


df_clean.head()


# In[ ]:


column_names = ["PassengerId", "Pclass", "Age", "SibSp", "Parch", "Fare", "Embarked", "Sex_10", "Survived"]


# In[ ]:


df_clean = df_clean.reindex(columns=column_names)


# In[ ]:


df_clean.head()


# In[ ]:


# from sklearn.preprocessing import StandardScaler
# import os


# In[ ]:


X_train = df_clean.drop(["Survived", "Embarked", "PassengerId"], axis=1)
y_train = df_clean["Survived"]
print(X_train.shape, y_train.shape)


# In[ ]:


X_train.head()


# In[ ]:


#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)


# In[ ]:


from sklearn.linear_model import LogisticRegression
# classifier = LogisticRegression(max_iter = 1000)
classifier = LogisticRegression()
classifier


# In[ ]:


classifier.fit(X_train, y_train)


# In[ ]:


classifier.predict(X_train)


# In[ ]:


df_merge_test.isnull().any()


# In[ ]:


df_merge_test.count()


# In[ ]:


df_merge_test=df_merge_test.drop(['Cabin', 'Ticket', 'Name'],axis=1)


# In[ ]:


df_merge_test['Age']=df_merge_test['Age'].interpolate()


# In[ ]:


df_merge_test['Age']=df_merge_test['Age'].fillna(df.Age.median())


# In[ ]:


df_merge_test_clean = df_merge_test.dropna()
df_merge_test_clean.head()


# In[ ]:


df_merge_test_clean['Sex'].value_counts()


# In[ ]:


def transform_sex(n):
  if n == 'female':
    return 1
  else:
    return 0
df_merge_test_clean.head()

#get dummies label encoder 
df_merge_test_clean['Sex_10'] = df_merge_test_clean['Sex'].map(transform_sex)


# In[ ]:


df_merge_test_clean=df_merge_test_clean.drop(['Sex'],axis=1)


# In[ ]:


column_names = ["PassengerId", "Pclass", "Age", "SibSp", "Parch", "Fare", "Embarked", "Sex_10", "Survived"]


# In[ ]:


df_merge_test_clean = df_clean.reindex(columns=column_names)
df_merge_test_clean.head()


# In[ ]:


# from sklearn.preprocessing import StandardScaler
# import os


# In[ ]:


X_test = df_merge_test_clean.drop(["Survived", "Embarked", "PassengerId"], axis=1)
y_test = df_merge_test_clean["Survived"]
print(X_train.shape, y_train.shape)


# In[ ]:


from sklearn.linear_model import LogisticRegression
# classifier = LogisticRegression(max_iter = 1000)
classifier = LogisticRegression()
classifier


# In[ ]:


classifier.fit(X_test, y_test)


# In[ ]:


classifier.predict(X_test)


# In[ ]:


print(f"Training Data Score: {classifier.score(X_train, y_train)}")
print(f"Testing Data Score: {classifier.score(X_test, y_test)}")


# In[ ]:


predictions = classifier.predict(X_test)
print(f"First 10 Predictions:   {predictions[:10]}")
print(f"First 10 Actual labels: {y_test[:10].tolist()}")


# In[ ]:


pd.DataFrame({"Prediction": predictions, "Actual": y_test}).reset_index(drop=True)


# In[ ]:


# Save the model
import pickle
filename = 'survival_model_trained.pkl'
pickle.dump(classifier,open(filename,'wb'))


# In[ ]:


# df_clean.to_csv("data_cleaned.csv")


# In[ ]:





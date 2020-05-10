#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv(r'C:\Users\Niladri1996\Desktop\Python\train_1.csv')


# In[3]:


df.head()


# In[4]:


a=list(df[df['year']=='[]'].index)
print(len(a))


# In[5]:


df=df.drop(a)


# In[6]:


a=list(df[df['year']=='[]'].index)
print(len(a))


# In[7]:


X=df.drop(['Unnamed: 0', 'variety', 'length', 'dummy'],axis=1)
y=df['dummy']


# In[8]:


X.head()


# In[9]:


y


# In[19]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)


# In[20]:


#decision tree

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
clf = DecisionTreeClassifier(max_depth=5000)
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy:",accuracy_score(y_test, y_pred))


# In[25]:


#random forest

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 200)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print("Accuracy:",accuracy_score(y_test, y_pred))


# In[26]:


#Logistic Regression

from sklearn import datasets, linear_model 
reg = linear_model.LogisticRegression() 
reg.fit(X_train, y_train) 
y_pred = reg.predict(X_test) 
print("Accuracy:",  accuracy_score(y_test, y_pred)) 


# In[27]:


#Naive Bayes

from sklearn.naive_bayes import GaussianNB 
gnb = GaussianNB() 
gnb.fit(X_train, y_train) 
y_pred = gnb.predict(X_test)  
from sklearn import metrics 
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))


# In[28]:


#KNN with k=5

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[29]:


#KNN with k=8

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=8)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[30]:


#KNN with k=2

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[31]:


#KNN with k=1

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[32]:


#KNN with k=20

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=20)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[34]:


#So the best model corresponding to the data is the RANDOM FOREST

#so we'll apply it on the test data


# In[140]:


data=pd.read_csv(r'C:\Users\Niladri1996\Desktop\Python\test.csv')


# In[141]:


data.head()


# In[142]:


import re


# In[143]:


def revyr(a):
    return re.findall(r'\d+',a)


# In[144]:


def yy(a):
    if len(a)>=1:
        return a[0]
    else:
        return a


# In[145]:


def revstmnt(a):
    return re.findall(r"\((.*)\)",a)


# In[146]:


data['year']=data['review_title'].apply(revyr)
data['year']=data['year'].apply(yy)
data['statement']=data['review_title'].apply(revstmnt)
data['statement']=data['statement'].apply(yy)
data.head()


# In[147]:


col=data.columns
v=len(data.country)
for j in range(13):
    w=data.iloc[:,j].isnull().sum()
    print(col[j],"\t\t\t",w*100/v)  


# In[148]:


lits1=['country','year','statement','points','price','province','winery']
for col in lits1:
    data[col].replace('', np.nan, inplace=True)
    print(len(data.iloc[:,1]))
    data.dropna(subset=[col], inplace=True)
    print(len(data.iloc[:,1]))


# In[149]:


19267/20665


# In[150]:


data1=data


# In[59]:


data.drop(['user_name','review_title','review_description','region_2'],axis=1,inplace=True)
data.head()


# In[60]:


data.fillna(method='ffill',inplace=True)


# In[63]:


col=data.columns
y=len(data.country)
for j in range(9):
    x=data.iloc[:,j].isnull().sum()
    print(col[j],"\t\t\t",x*100/y)


# In[65]:


a=list(data[data['statement']=='[]'].index)
print(len(a))


# In[66]:


data.head()


# In[68]:


ab=list(data[data['statement']=='[]'].index)
print(len(ab))


# In[69]:


def length(a):
    return len(a)
data['length']=data['statement'].apply(length)
data.head()


# In[71]:


def nan(a):
    if len(a)>0:
        return a
    else:
        return 'NA'
data['statement']=data['statement'].apply(nan)
data['length']=data['statement'].apply(length)
data.head()


# In[72]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


# In[74]:


dt=data
dt.head()


# In[162]:


len(dt)


# In[80]:


dt.iloc[:,8]= le.fit_transform(dt.iloc[:,8])
dt.head()


# In[81]:


a=list(dt[dt['year']=='[]'].index)
print(len(a))


# In[83]:


test=dt.drop(['length'],axis=1)
test.head()


# In[ ]:





# In[ ]:





# In[109]:


m=list(test[test['year']=='[]'].index)
print(len(m))


# In[ ]:





# In[110]:


len(test.abc)


# In[111]:


616/19267


# In[151]:


data=data.drop(m)
test=test.drop(m)
data1=data1.drop(m)


# In[163]:





# In[159]:


len(data1.year)


# In[153]:


data1.head()


# In[120]:


m=list(test[test['year']=='[]'].index)
print(len(m))


# In[117]:


#random forest

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 20)
classifier.fit(X_train, y_train)
veriety_dummy = classifier.predict(test)
print(veriety_dummy)


# In[161]:


len(test)


# In[138]:


dict1={2:'jj'}
dict3={}
dict3.update(dict1)
print(dict3)
dict3[2]


# In[126]:


df.head()


# In[128]:


dict2={}
for j in range(len(df.year)):
    dict1={df['dummy'].iloc[j]:df['variety'].iloc[j]}
    dict2.update(dict1)
print(dict2)


# In[139]:


new_variety=[]
for j in range(len(veriety_dummy)):
    new_variety.append(dict2[veriety_dummy[j]])
print(new_variety)
print(len(new_variety))


# In[155]:


print(len(new_variety))


# In[156]:


len(data1)


# In[164]:


m=list(data1[data1['year']=='[]'].index)
print(len(m))


# In[167]:


def g(a):
    return len(str(a))
data1['a']=data1['year'].apply(g)


# In[169]:


m=list(data1[data1['a']==2].index)
print(len(m))


# In[170]:


data1=data1.drop(m)


# In[171]:


len(data1)


# In[173]:


data1['vareity new']=new_variety


# In[174]:


data1.head()


# In[175]:


data1.to_csv(r'C:\Users\Niladri1996\Desktop\Python\Updated_test_data.csv')
data1.head()


# In[ ]:





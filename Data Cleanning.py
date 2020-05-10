#!/usr/bin/env python
# coding: utf-8

# In[865]:


import pandas as pd
import re
wine1=pd.read_csv(r'C:\Users\Niladri1996\Desktop\Python\train.csv')


# In[866]:


wine1.columns


# In[867]:


wine1['review_title']


# In[868]:


def revyr(a):
    return re.findall(r'\d+',a)


# In[869]:


wine1['year']=wine1['review_title'].apply(revyr)


# In[870]:


def yy(a):
    if len(a)>=1:
        return a[0]
    else:
        return a


# In[871]:


wine1['year']=wine1['year'].apply(yy)


# In[872]:


wine1.head()


# In[873]:


def revstmnt(a):
    return re.findall(r"\((.*)\)",a)


# In[874]:


wine1['statement']=wine1['review_title'].apply(revstmnt)
wine1.head()


# In[875]:


wine1['statement']=wine1['statement'].apply(yy)
wine1.head()


# In[876]:


col=wine1.columns
print(col)


# In[877]:


v=len(wine1.country)
for j in range(14):
    w=wine1.iloc[:,j].isnull().sum()
    print(col[j],"\t\t\t",w*100/v)   


# In[878]:


lits=['country','year','statement','points','price','province','winery','variety']
print(len(lits))


# In[879]:


import numpy as np
for col in lits:
    wine1[col].replace('', np.nan, inplace=True)
    print(len(wine1.iloc[:,1]),len(wine1.iloc[:,7]))
    wine1.dropna(subset=[col], inplace=True)
    print(len(wine1.iloc[:,1]),len(wine1.iloc[:,7]))


# In[880]:


77057/82657*100


# In[881]:


wine1.head()


# In[882]:


y=len(wine1.country)
print(y)


# In[883]:


col=wine1.columns
print(col)


# In[884]:


for j in range(14):
    x=wine1.iloc[:,j].isnull().sum()
    print(col[j],"\t\t\t",x*100/y)


# In[885]:


wine1.drop(['user_name','review_title','review_description','region_2'],axis=1,inplace=True)
wine1.head()


# In[886]:


col=wine1.columns
print(col,"\n",len(wine1['country']),len(col))


# In[887]:


for j in range(10):
    x=wine1.iloc[:,j].isnull().sum()
    print(col[j],"\t\t\t",x*100/y)


# In[888]:


wine=wine1
wine.head()


# In[889]:


wine.fillna(method='ffill',inplace=True)


# In[890]:


col=wine.columns
y=len(wine.country)
print(col,len(col),y)


# In[891]:


for j in range(10):
    x=wine.iloc[:,j].isnull().sum()
    print(col[j],"\t\t\t",x*100/y)


# In[892]:


wine.head()


# In[893]:


print(wine.columns)


# In[894]:


wine3=pd.read_csv(r'C:\Users\Niladri1996\Desktop\Python\test.csv')


# In[895]:


wine3.columns


# In[896]:


wine3['review_title']


# In[897]:


wine3['year']=wine3['review_title'].apply(revyr)
#wine1[wine1['year']!=1]


# In[898]:


wine3['year']=wine3['year'].apply(yy)


# In[899]:


wine3.head()


# In[900]:


wine3['statement']=wine3['review_title'].apply(revstmnt)
wine3.head()


# In[901]:


wine3['statement']=wine3['statement'].apply(yy)
wine3.head()


# In[902]:


col=wine3.columns
print(col)


# In[903]:


v=len(wine3.country)
for j in range(13):
    w=wine3.iloc[:,j].isnull().sum()
    print(col[j],"\t\t\t",w*100/v)   


# In[904]:


lits1=['country','year','statement','points','price','province','winery']
print(len(lits1))


# In[905]:


for col in lits1:
    wine3[col].replace('', np.nan, inplace=True)
    print(len(wine3.iloc[:,1]),len(wine3.iloc[:,7]))
    wine3.dropna(subset=[col], inplace=True)
    print(len(wine3.iloc[:,1]),len(wine3.iloc[:,7]))


# In[906]:


print(77057/82657*100)
print(19267/20665*100)


# In[907]:


wine3.head()


# In[908]:


y=len(wine3.country)
print(y)


# In[909]:


col=wine3.columns
print(col)


# In[910]:


for j in range(13):
    x=wine3.iloc[:,j].isnull().sum()
    print(col[j],"\t\t\t",x*100/y)


# In[911]:


wine3.drop(['user_name','review_title','review_description','region_2'],axis=1,inplace=True)
wine3.head()


# In[912]:


col=wine3.columns
print(col,"\n",len(wine3['country']),len(col))


# In[913]:


for j in range(9):
    x=wine3.iloc[:,j].isnull().sum()
    print(col[j],"\t\t\t",x*100/y)


# In[914]:


Wine=wine3
Wine.head()


# In[915]:


Wine.fillna(method='ffill',inplace=True)


# In[916]:


col=Wine.columns
y=len(Wine.country)
print(col,len(col),y)


# In[917]:


for j in range(9):
    x=Wine.iloc[:,j].isnull().sum()
    print(col[j],"\t\t\t",x*100/y)


# In[918]:


Wine.head()


# In[919]:


print(Wine.columns)


# In[920]:


wine['variety'].value_counts()


# In[921]:


def length(a):
    return len(a)


# In[922]:


wine['length']=wine['statement'].apply(length)
wine.head()


# In[923]:


wine[wine['length']<1]


# In[862]:


def nan(a):
    if len(a)>0:
        return a
    else:
        return 'NA'


# In[924]:


wine['statement']=wine['statement'].apply(nan)
wine.head()


# In[926]:


wine[wine['length']<1]


# In[929]:


Wine['length']=Wine['statement'].apply(length)
Wine['statement']=Wine['statement'].apply(nan)
Wine[Wine['length']<1]


# In[575]:


#dummies = pd.get_dummies(wine['country'])
#dummies2 = pd.get_dummies(Wine['designation'])


# In[930]:


wine.head()


# In[931]:


Wine.head()


# In[958]:


from sklearn import metrics 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder


# In[933]:


le=LabelEncoder()


# In[962]:


def label(a):
    return le.fit_transform(a)


# In[964]:


wine['dummy']=wine['variety'].apply(length)


# In[965]:


wine.head()


# In[967]:


wine.iloc[:,11]= le.fit_transform(wine.iloc[:,7])
wine.head()


# In[946]:


Wine.iloc[:,6]= le.fit_transform(Wine.iloc[:,6])
Wine.head()


# In[969]:


wine['dummy'].value_counts()


# In[971]:


wine['variety'].value_counts()


# In[947]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[1010]:


feature_cols = ['country', 'designation', 'points', 'price', 'province', 'region_1', 'winery', 'year', 'statement']
X = wine[feature_cols] 
Y = wine[['dummy']]


# In[1011]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1) 


# In[1012]:


X_train.head()


# In[986]:


Y_train


# In[1006]:


wine['variety'].value_counts()


# In[1019]:


Wine.to_csv(r'C:\Users\Niladri1996\Desktop\Python\train_2.csv')
Wine.head()


# In[479]:


random forest

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


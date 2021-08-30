#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

TopikBerita = pd.read_csv('Judul Berita.csv')


# In[2]:


TopikBerita.head()


# In[3]:


TopikBerita.info()


# In[4]:


import re
import string

def text_clean_1(text):
    text = text.lower()
    text = re.sub( '\[.*?\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

cleaned1 = lambda x: text_clean_1(x)


# In[5]:


TopikBerita['Cleaned'] = pd.DataFrame(TopikBerita.Title.apply(cleaned1))
TopikBerita.head(10)


# In[6]:


def text_clean_2(text):
    text = re.sub( '[''""....]', '', text)
    text = re.sub('\n', '', text)
    return text

cleaned2 = lambda x: text_clean_2(x)


# In[7]:


TopikBerita['Cleaned_new'] = pd.DataFrame(TopikBerita['Cleaned'].apply(cleaned2))
TopikBerita.head(10)


# In[18]:


from sklearn.model_selection import train_test_split

X = TopikBerita.Cleaned_new
Y = TopikBerita.Class

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=0)


# In[19]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

tvec = TfidfVectorizer()
clf2 = RandomForestClassifier(n_estimators = 300, criterion = 'entropy', random_state = 0)

from sklearn.pipeline import Pipeline

model = Pipeline([('vectorizer',tvec), ('classifier',clf2)])
model.fit(X_train, Y_train)


# In[20]:


Y_Pred = model.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(Y_test, Y_Pred))
print(classification_report(Y_test, Y_Pred))

from sklearn.metrics import accuracy_score
p = accuracy_score(Y_test, Y_Pred)
print(p*100)


# **LOGISTIK REGRESSION**

# In[21]:


from sklearn.linear_model import LogisticRegression 
classifier = LogisticRegression(random_state=0) 

model = Pipeline([('vectorizer',tvec), ('classifier',classifier)])
model.fit(X_train, Y_train)

Y_Pred = model.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(Y_test, Y_Pred))
print(classification_report(Y_test, Y_Pred))

from sklearn.metrics import accuracy_score
p = accuracy_score(Y_test, Y_Pred)
print(p*100)


# **KNN**

# In[22]:


from sklearn.neighbors import KNeighborsClassifier

neigh = KNeighborsClassifier(n_neighbors=7)

model = Pipeline([('vectorizer',tvec), ('classifier',neigh)])
model.fit(X_train, Y_train)

Y_Pred = model.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(Y_test, Y_Pred))
print(classification_report(Y_test, Y_Pred))

from sklearn.metrics import accuracy_score
p = accuracy_score(Y_test, Y_Pred)
print(p*100)


# In[ ]:





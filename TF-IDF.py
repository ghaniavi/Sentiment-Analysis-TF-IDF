#!/usr/bin/env python
# coding: utf-8

# In[5]:


from sklearn.feature_extraction.text import TfidfVectorizer

myvocabulary = ['tim tam', 'jam', 'fresh milk', 'chocolates', 'biscuit pudding']
corpus = {1: "making chocolates biscuit pudding easy first get your favourite biscuit chocolates", 2: "tim tam drink new recipe that yummy and tasty more thicker than typical milkshake that uses normal chocolates", 3: "making chocolates drink different way using fresh milk egg"}


# In[7]:


tfidf = TfidfVectorizer(vocabulary = myvocabulary, stop_words = 'english', ngram_range=(1,2))
tfs = tfidf.fit_transform(corpus.values())

feature_names = tfidf.get_feature_names()
corpus_index = [n for n in corpus]
rows, cols = tfs.nonzero()
for row, col in zip(rows, cols):
    print((feature_names[col], corpus_index[row]), tfs[row, col])


# In[8]:


import pandas as pd
df = pd.DataFrame(tfs.T.todense(), index=feature_names, columns=corpus_index)
print(df)


# In[ ]:





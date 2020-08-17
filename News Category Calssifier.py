#!/usr/bin/env python
# coding: utf-8

# In[5]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (6,6)

from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints

from keras.layers import Dense, Input, LSTM, Bidirectional, Activation, Conv1D, GRU, TimeDistributed
from keras.layers import Dropout, Embedding, GlobalMaxPooling1D, MaxPooling1D, Add, Flatten, SpatialDropout1D
from keras.layers import GlobalAveragePooling1D, BatchNormalization, concatenate
from keras.layers import Reshape, merge, Concatenate, Lambda, Average
from keras.models import Sequential, Model, load_model
from keras.callbacks import ModelCheckpoint
from keras.initializers import Constant
from keras.layers.merge import add

from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.utils import np_utils

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

# Any results you write to the current directory are saved as output.


# # prepare data

# In[6]:


df = pd.read_json('News_Category_Dataset_v2.json', lines=True)
df.head()


# In[7]:


cates = df.groupby('category')
print("total categories: ", cates.ngroups)
print(cates.size())


# In[8]:


# THE WORLDPOST and WORLDPOST should be the same category, so merge them.
df.category = df.category.map(lambda x: "WORDPOST" if x == "THE WORLDPOST" else x)


# In[9]:


# using headlines and short_description as input X
df['text'] = df.headline + " " + df.short_description


# In[10]:


# tokenizing
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df.text)
X = tokenizer.texts_to_sequences(df.text)
df['words'] = X


# In[11]:


# delete some empty and short data
df['word_length'] = df.words.apply(lambda i: len(i))
df = df[df.word_length >= 5]

df.head()


# In[8]:


df.word_length.describe()


# In[9]:


# using 50 for padding length
maxlen = 50
X = list(sequence.pad_sequences(df.words, maxlen=maxlen))


# In[12]:


# category to id
categories = df.groupby('category').size().index.tolist()
category_int = {}
int_category = {}
for i, k in enumerate(categories):
    category_int.update({k:i})
    int_category.update({i:k})

df['c2id'] = df['category'].apply(lambda x: category_int[x])


# # glove embedding

# In[20]:


word_index = tokenizer.word_index

EMBEDDING_DIM = 100

embeddings_index = {}
f = open('glove.6B.100d.txt', 'rt', encoding='UTF8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.array(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s unique tokens.' % len(word_index))
print('Total %s word vectors.' % len(embeddings_index))


# In[21]:


embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(len(word_index)+1,
                            EMBEDDING_DIM, 
                            embeddings_initializer = Constant(embedding_matrix),
                            input_length = maxlen, 
                            trainable=False)


# # split dataset

# In[22]:


# prepared data

X = np.array(X)
Y = np_utils.to_categorical(list(df.c2id))

#and split to training set and validation set
seed = 29
x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=seed)


# # Bidirectional GRU + Conv

# In[23]:


# Bidrectional LSTM with convolution
# from https://www.kaggle.com/eashish/bidirectional-gru-with-convolution

inp = Input(shape=(maxlen,), dtype='int32')
x = embedding_layer(inp)
x = SpatialDropout1D(0.2)(x)
x = Bidirectional(GRU(128, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)
x = Conv1D(64, kernel_size=3)(x)
avg_pool = GlobalAveragePooling1D()(x)
max_pool = GlobalMaxPooling1D()(x)
x = concatenate([avg_pool, max_pool])
outp = Dense(len(int_category), activation="softmax")(x)

BiGRU = Model(inp, outp)
BiGRU.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

BiGRU.summary()


# In[24]:


# training
bigru_history = BiGRU.fit(x_train,
                         y_train,
                         batch_size=128,
                         epochs=20,
                         validation_data=(x_val, y_val))


# In[28]:


pip install newsapi-python


# In[30]:


from newsapi import NewsApiClient
# Init
newsapi = NewsApiClient(api_key='1442af938d214ef09a688b70293e9bea')

# /v2/top-headlines
top_headlines = newsapi.get_top_headlines(category='business',
                                          language='en',
                                          country='us')


# In[31]:


top_headlines


# In[73]:


df = pd.DataFrame(top_headlines['articles'])


# In[74]:


df.head(3)


# In[91]:


df['text'] = df.description + " " + df.title
drop_df = df.dropna(axis=0)
df = drop_df
df


# In[92]:


tokenizer = Tokenizer()
tokenizer.fit_on_texts(df.text)
X = tokenizer.texts_to_sequences(df.text)
df['words'] = X


# In[93]:


df


# In[94]:


df['word_length'] = df.words.apply(lambda i: len(i))
df = df[df.word_length >= 5]
df.head()


# In[95]:


df.word_length.describe()


# In[96]:


X = list(sequence.pad_sequences(df.words, maxlen=maxlen))
X = np.array(X)


# In[97]:


cate = BiGRU.predict(X)


# In[98]:


cate_list = []
for i in range(len(cate)):
    y = cate[i]
    p = np.argmax(y)
    cate_list.append(int_category[p])
    print(int_category[p])


# In[99]:


cate_list


# In[100]:


df['sub_category'] = cate_list


# In[101]:


view = df[['title','description','sub_category']]
view


# In[46]:


pip install dill


# In[2]:


import dill as pickle


# In[129]:


filename = 'NewsClf'


# In[130]:


with open(filename, 'wb') as file:
    pickle.dump(BiGRU, file)


# In[131]:


with open(filename, 'rb') as f:
    loaded_model = pickle.load(f)


# In[132]:


loaded_model.predict(X)


# In[13]:


with open('int_category', 'wb') as file:
    pickle.dump(int_category, file)


# In[ ]:





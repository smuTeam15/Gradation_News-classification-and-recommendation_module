from newsapi import NewsApiClient
import dill as pickle
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing import sequence

# max length of individual input dataset
maxlen = 50
# path of classifier file
clf_file = 'NewsClf'
# path of category index table file
index_file = 'int_category'

# intialize News API
newsapi = NewsApiClient(api_key='1442af938d214ef09a688b70293e9bea')

# crawlling news top-headlines(input dataset)
top_headlines = newsapi.get_top_headlines(category='business',
                                          language='en',
                                          country='us')

# make input dataset
df = pd.DataFrame(top_headlines['articles'])
df = df.dropna(axis=0)
df['text'] = df.description + " " + df.title

# tokenizing input dataset
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df.text)
X = tokenizer.texts_to_sequences(df.text)
df['words'] = X

df['word_length'] = df.words.apply(lambda i: len(i))
df = df[df.word_length >= 5]

# transform input data
X = list(sequence.pad_sequences(df.words, maxlen=maxlen))
X = np.array(X)

# read index table and classifier file
with open('/Users/kimhyunho/Desktop/Visual Code/code/'+index_file, 'rb') as f:
    int_category = pickle.load(f)
with open('/Users/kimhyunho/Desktop/Visual Code/code/'+clf_file, 'rb') as f:
    loaded_model = pickle.load(f)
# classifiying input dataset
cate = loaded_model.predict(X)

# extract classified data from category and append on dataframe
cate_list = []
for i in range(len(cate)):
    y = cate[i]
    p = np.argmax(y)
    cate_list.append(int_category[p])

df['sub_category'] = cate_list
print(df)

#!/usr/bin/env python
# coding: utf-8

# In[1]:


import neuralplot
import re
import codecs
import pandas as pd
import string
import numpy as np
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from gensim.models import Word2Vec
from text_processor import TextProcessor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from wordcloud_generator import WordCloudGenerator
from tensorflow.keras.models import save_model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import GlobalMaxPooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, Conv1D, MaxPooling1D, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils.vis_utils import plot_model
from sklearn.model_selection import cross_val_score
from keras.utils import to_categorical
from wordcloud import WordCloud
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv('spam.csv', header=0)


# In[3]:


df.columns


# In[4]:


df.drop_duplicates(inplace=True)
df.dropna(inplace=True)


# In[5]:


df


# In[6]:


df.shape


# In[7]:


df.describe()


# In[8]:


classes = df['label'].unique()
if __name__ == "__main__":
    print(classes)
    np.size(classes)


# In[9]:


classes = df['label'].unique()
if __name__ == "__main__":
    print(classes)
    np.size(classes)


# In[10]:


wcg = WordCloudGenerator(target='label', col='text', df=df)
wcg.set_classes(classes)
wcg.generate_wordclouds()


# In[11]:


tp = TextProcessor()
df = df.assign(Processed_Title=df["text"].apply(tp.process))
df.head()


# In[12]:


wcg = WordCloudGenerator(target='label', col='Processed_Title', df=df)
wcg.set_classes(classes)
wcg.generate_wordclouds()


# In[13]:


corpus = list(df["Processed_Title"])


# In[14]:


corpus


# In[15]:


np.random.seed(123)
num_classes = np.size(classes)

# Entraînement du modèle Word2Vec
model_w2v = Word2Vec(corpus, vector_size=100, min_count=7)

embedding_size = 100
max_len = 50
Y = df["label"]

# Convertir le texte en vecteurs utilisant les embeddings Word2Vec
X = []
for sentence in corpus:
    sentence_vec = []
    for word in sentence:
        if word in model_w2v.wv:
            sentence_vec.append(model_w2v.wv[word])
        else:
            sentence_vec.append(np.zeros(embedding_size))
    X.append(sentence_vec)
X = pad_sequences(X, max_len)
    
# Diviser le corpus en ensembles d'entraînement et de test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Combine Y_train and Y_test
Y_combined = np.concatenate((Y_train, Y_test))
Y_combined = Y_combined.astype(str)

# Fit a new LabelEncoder object on the combined set of Y_train and Y_test
le = LabelEncoder()
le.fit(Y_combined)

# Encode the labels in Y_train and Y_test as integers
Y_train_encoded = le.transform(Y_train)
Y_test_encoded = le.transform(Y_test)

# Convert the encoded labels in Y_train and Y_test to one-hot encoded categorical representation
Y_train_categorical = to_categorical(Y_train_encoded, num_classes)
Y_test_categorical = to_categorical(Y_test_encoded, num_classes)

# Créer le modèle CNN-RNN
num_filters = 48
filter_sizes = [3, 4, 5]
lstm_units = 50
sequence_input = Input(shape=(max_len, embedding_size), dtype='int32')
embedded_sequences = Embedding(len(model_w2v.wv.index_to_key)+1,
                               embedding_size,
                               input_length=max_len,
                               trainable=True)(sequence_input)

# Couches CNN
conv_blocks = []
for sz in filter_sizes:
    conv = Conv1D(filters=num_filters,
                  kernel_size=sz,
                  padding="valid",
                  activation="relu",
                  strides=1)(embedded_sequences)
    conv = Flatten()(conv) # dérouler (flatten) les sorties de la couche Conv1D
    conv = Reshape((-1, num_filters))(conv)
    conv = GlobalMaxPooling1D()(conv)
    conv_blocks.append(conv)
x = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
x = Reshape((-1, num_filters))(x)

# Couche RNN
x = Bidirectional(LSTM(lstm_units))(x)

# Couche Dense finale
x = Dense(num_classes, activation='softmax')(x)

# Compiler le modèle
model = Model(sequence_input, x)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Entraîner le modèle sur l'ensemble d'entraînement
model.fit(X_train, Y_train_categorical, batch_size=32, epochs=2, validation_split=0.1)

# Évaluer les performances du modèle sur l'ensemble de test
score = model.evaluate(X_test, Y_test_categorical, batch_size=12)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[16]:


model.summary()


# In[17]:


from PIL import ImageFont
import visualkeras
visualkeras.layered_view(model, legend=True)


# In[ ]:


save_model(model, 'Model.py')


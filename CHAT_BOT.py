#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle

# Importer X_train
with open('X_train.pkl', 'rb') as file:
    X_train = pickle.load(file)

# Importer X_test
with open('X_test.pkl', 'rb') as file:
    X_test = pickle.load(file)

# Importer Y_train
with open('Y_train.pkl', 'rb') as file:
    Y_train = pickle.load(file)

# Importer Y_test
with open('Y_test.pkl', 'rb') as file:
    Y_test = pickle.load(file)

# Importer Y_combined
with open('Y_combined.pkl', 'rb') as file:
    Y_combined = pickle.load(file)

# Importer Y_combined_encoded
with open('Y_combined_encoded.pkl', 'rb') as file:
    Y_combined_encoded = pickle.load(file)

# Importer le
with open('le.pkl', 'rb') as file:
    le = pickle.load(file)

# Importer Y_train_encoded
with open('Y_train_encoded.pkl', 'rb') as file:
    Y_train_encoded = pickle.load(file)

# Importer Y_test_encoded
with open('Y_test_encoded.pkl', 'rb') as file:
    Y_test_encoded = pickle.load(file)

# Importer Y_train_categorical
with open('Y_train_categorical.pkl', 'rb') as file:
    Y_train_categorical = pickle.load(file)

# Importer Y_test_categorical
with open('Y_test_categorical.pkl', 'rb') as file:
    Y_test_categorical = pickle.load(file)


# In[5]:


with open('corpus.pkl', 'rb') as file:
    corpus = pickle.load(file)


# In[6]:


len(corpus)


# In[7]:


corpus[10]


# In[8]:


import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
from gensim.models import Word2Vec
from text_processor import TextProcessor
from wordcloud_generator import WordCloudGenerator
from tensorflow.keras.preprocessing.sequence import pad_sequences
import colorama 
colorama.init()
from colorama import Fore, Style, Back
max_len = 50
embedding_size = 50


# In[9]:


with open('classes.pkl', 'rb') as file:
    classes = pickle.load(file)


# In[32]:


def chat(input_text):
    
    
    while True:
        
        with open('classes.pkl', 'rb') as file:
            classes = pickle.load(file)
        
        # load trained model
        model = load_model('New_Model.py')
        
        model_w2v = Word2Vec.load("model_w2v.bin")
        
        # Importer le
        with open('le.pkl', 'rb') as file:
            le = pickle.load(file)
            
        with open('corpus.pkl', 'rb') as file:
            corpus = pickle.load(file)
        
        print(Fore.LIGHTBLUE_EX + "User: " + Style.RESET_ALL, end="")
        
        
        if input_text.lower() == "quit":
            break

        # Create an instance of TextProcessor
        text_processor = TextProcessor()

        # Preprocess the new text
        preprocessed_text = text_processor.process(input_text)
        
        # Convert the preprocessed text into word embeddings
        new_text_vec = []
        for word in preprocessed_text:
            if word in model_w2v.wv:
                new_text_vec.append(model_w2v.wv[word])
            else:
                new_text_vec.append(np.zeros(embedding_size))
        new_text_vec = np.array([new_text_vec])

        # Pad the embedded sequence
        padded_new_text_vec = pad_sequences(new_text_vec, max_len)
        predictions = model.predict(padded_new_text_vec)
        predicted_class_index = np.argmax(predictions[0])
        predicted_class = le.inverse_transform([predicted_class_index])[0]
        predicted_class_str = str(predicted_class)
        return predicted_class_str
    

# In[33]:


print(Fore.YELLOW + "Start messaging with the bot (type quit to stop)!" + Style.RESET_ALL)



# In[15]:


model = load_model('New_Model.py')

model_w2v = Word2Vec.load("model_w2v.bin")

text_processor = TextProcessor()

input_text = "Déchets médicaux"

# Preprocess the new text
preprocessed_text = text_processor.process(input_text)
        
# Convert the preprocessed text into word embeddings
new_text_vec = []
for word in preprocessed_text:
    if word in model_w2v.wv:
        new_text_vec.append(model_w2v.wv[word])
    else:
        new_text_vec.append(np.zeros(embedding_size))
new_text_vec = np.array([new_text_vec])

# Pad the embedded sequence
padded_new_text_vec = pad_sequences(new_text_vec, max_len)
predictions = model.predict(padded_new_text_vec)
predicted_class_index = np.argmax(predictions[0])
predicted_class = le.inverse_transform([predicted_class_index])[0]
predicted_class


# In[22]:


predicted_class.dtype


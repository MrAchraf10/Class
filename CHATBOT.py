#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
from gensim.models import Word2Vec
from text_processor import TextProcessor
from wordcloud_generator import WordCloudGenerator
from tensorflow.keras.preprocessing.sequence import pad_sequences

max_len = 50
embedding_size = 50

def chat(input_text):

    while True:
        
        with open('classes.pkl', 'rb') as file:
            classes = pickle.load(file)
        
        # load trained model
        model = load_model('LSTM_Model.py')
        
        model_w2v = Word2Vec.load("model_w2v.bin")
        
        # Importer le
        with open('le.pkl', 'rb') as file:
            le = pickle.load(file)
            
        with open('corpus.pkl', 'rb') as file:
            corpus = pickle.load(file)
        
        if input_text.lower() == "quit":
            break

        # Create an instance of TextProcessor
        text_processor = TextProcessor()

        # Preprocess the new text
        preprocessed_text = text_processor.process(input_text)

        # Check if the input text is readable
        if any(char.isalpha() for char in input_text):

            # Check if the input text contains only one word
            if len(preprocessed_text) == 1:
                response = "Please add more details to your input."
            else:
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
        
        else:
            response = "Please enter a readable text."
        
        return response


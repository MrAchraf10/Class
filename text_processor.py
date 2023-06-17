#!/usr/bin/env python
# coding: utf-8

# In[2]:


import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer

class TextProcessor:
    def __init__(self, remove_stopwords=True, remove_punctuation=True):
        self.remove_stopwords = remove_stopwords
        self.remove_punctuation = remove_punctuation
        
        if self.remove_stopwords:
            self.stopwords_french = stopwords.words('french')
            
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = SnowballStemmer(language='french')
        
    def process(self, text):
        
        if not isinstance(text, str):   # Check if text is not already a string
            text = str(text)
        # remove stock market tickers like $GE
        text = re.sub(r'\$\w*', '', text)
        # remove hyperlinks    
        text = re.sub(r'https?://[^\s\n\r]+', '', text)
        # remove hashtags
        # only removing the hash # sign from the word
        text = re.sub(r'#', '', text)
        
        # convert to lowercase
        text = text.lower()
        
        # tokenize text
        text_tokens = word_tokenize(text)
        
        # lemmatize and stem text
        text_processed = []
        for word in text_tokens:
            word_lemmatized = self.lemmatizer.lemmatize(word)
            word_stemmed = self.stemmer.stem(word_lemmatized)
            text_processed.append(word_stemmed)
        
        # remove stopwords and punctuation
        text_clean = []
        for word in text_processed:
            if self.remove_stopwords and (word in self.stopwords_french):
                continue
                
            if self.remove_punctuation and (word in string.punctuation):
                continue
            
            text_clean.append(word)
            
        return text_clean
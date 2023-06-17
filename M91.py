#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
import codecs
import pandas as pd
import string
import numpy as np
import nltk
import random
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('omw-1.4')
from deep_translator import GoogleTranslator
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
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


df = pd.read_excel('SF par commande OIG.xlsx', header=0)


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


df.rename(columns={'Sous Famille': 'sous famille'}, inplace=True)


# In[9]:


df.drop("Famille", axis = 1, inplace = True)


# In[10]:


df


# In[11]:


from deep_translator.exceptions import NotValidPayload

def back_translate(text):
    try:
        # Translate the text to english
        translated_text = GoogleTranslator(source='auto', target='en').translate(text)
        # Translate the translated text back to the original language
        back_translated_text = GoogleTranslator(source='auto', target='french').translate(translated_text)
        return back_translated_text
    except NotValidPayload:
        # Return the original text if it is not a valid payload
        return text
augmented_rows1 = []
for index, row in df.iterrows():
    text = row['description commande']
    target = row['sous famille']
    augmented_text = back_translate(str(text))
    augmented_rows1.append({'description commande': augmented_text, 'sous famille': target})
augmented_df1 = pd.DataFrame(augmented_rows1)

frames = [df, augmented_df1]
merged_df = pd.concat(frames)


# In[12]:


def synonym_replacement(text):
    tokens = word_tokenize(text, language='french')  # Tokenize the text into individual words
    augmented_texts = []
    for token in tokens:
        synonyms = []
        for syn in wordnet.synsets(token, lang='fra'):
            for lemma in syn.lemmas(lang='fra'):
                synonyms.append(lemma.name())  # Collect synonyms for each word
        if synonyms:
            synonym = synonyms[0]  # Select the first synonym as replacement
            augmented_texts.append(text.replace(token, synonym))  # Replace the word with its synonym
    return augmented_texts
augmented_rows2 = []
for index, row in merged_df.iterrows():
    text = row['description commande']
    target = row['sous famille']
    augmented_texts = synonym_replacement(str(text))
    for augmented_text in augmented_texts:
        augmented_rows2.append({'description commande': augmented_text, 'sous famille': target})
augmented_df2 = pd.DataFrame(augmented_rows2)

frames = [merged_df, augmented_df2]
merged_df1 = pd.concat(frames)


# In[13]:


def random_swap(text, n=5):
    tokens = word_tokenize(text)  # Tokenize the text into individual words
    if len(tokens) < 2:
        return [text]
    augmented_texts = []
    for _ in range(n):
        augmented_tokens = tokens.copy()
        idx1, idx2 = random.sample(range(len(tokens)), 2)  # Select two random indices
        augmented_tokens[idx1], augmented_tokens[idx2] = augmented_tokens[idx2], augmented_tokens[idx1]  # Swap the words
        augmented_text = ' '.join(augmented_tokens)  # Recreate the augmented text
        augmented_texts.append(augmented_text)
    return augmented_texts
augmented_rows3 = []
for index, row in merged_df1.iterrows():
    text = row['description commande']
    target = row['sous famille']
    augmented_texts = random_swap(str(text))
    for augmented_text in augmented_texts:
        augmented_rows3.append({'description commande': augmented_text, 'sous famille': target})
augmented_df3 = pd.DataFrame(augmented_rows3)

frames = [merged_df1, augmented_df3]
merged_df = pd.concat(frames)


# In[14]:


def text_rotation(text):
    # Split the text into sentences or paragraphs
    sentences = str(text).split('. ')
    # Shuffle the order of the sentences or paragraphs
    random.shuffle(sentences)
    # Join the shuffled sentences or paragraphs back into text
    rotated_text = '. '.join(sentences)
    return rotated_text
augmented_rows4 = []
for index, row in merged_df1.iterrows():
    text = row['description commande']
    target = row['sous famille']
    augmented_text = text_rotation(text)
    augmented_rows4.append({'description commande': augmented_text, 'sous famille': target})
augmented_df4 = pd.DataFrame(augmented_rows4)

frames = [merged_df, augmented_df4]
merged_df = pd.concat(frames)


# In[15]:


def random_deletion(text, p):
    tokens = nltk.word_tokenize(text)
    remaining_tokens = [token for token in tokens if random.uniform(0, 1) > p]
    if len(remaining_tokens) == 0:
        return text
    else:
        return ' '.join(remaining_tokens)
p = 0.45
augmented_rows6 = []
for index, row in merged_df1.iterrows():
    text = row['description commande']
    deleted_text = random_deletion(str(text), p)
    augmented_rows6.append({'description commande': deleted_text, 'sous famille': row['sous famille']})
augmented_df6 = pd.DataFrame(augmented_rows6)

frames = [merged_df, augmented_df6]
merged_df = pd.concat(frames)


# In[16]:


merged_df


# In[17]:


a = merged_df.isnull()
duplicates = merged_df.duplicated()
if a.any().any():
    print("There are null values in the DataFrame.")
    merged_df1 = merged_df.dropna()
else:
    print("There are no null values in the DataFrame.")
if duplicates.any():
    print("There are duplicates in the DataFrame.")
else:
    print("There are no duplicates in the DataFrame.")


# In[18]:


merged_df.shape


# In[19]:


classes = merged_df['sous famille'].unique()
if __name__ == "__main__":
    print(classes)
    np.size(classes)


# In[20]:


merged_df.shape


# In[21]:


if __name__ == "__main__":
    wcg = WordCloudGenerator(target='sous famille', col='description commande', df=merged_df)
    wcg.set_classes(classes)
    wcg.generate_wordclouds()


# In[22]:


tp = TextProcessor()
merged_df = merged_df.assign(Processed_Title=merged_df["description commande"].apply(tp.process))


# In[23]:


merged_df.shape


# In[24]:


if __name__ == "__main__":
    wcg = WordCloudGenerator(target='sous famille', col='Processed_Title', df=merged_df)
    wcg.set_classes(classes)
    wcg.generate_wordclouds()


# In[25]:


classes = merged_df['sous famille'].unique()
if __name__ == "__main__":
    print(classes)
    np.size(classes)


# In[26]:


corpus = list(merged_df["Processed_Title"])


# In[27]:


corpus


# In[28]:


import tensorflow
import keras
print(tensorflow.__version__)
print(keras.__version__)


# In[29]:


np.random.seed(123)
num_classes = np.size(classes)

# Entraînement du modèle Word2Vec
model_w2v = Word2Vec(corpus, vector_size=50, min_count=2)

embedding_size = 50
max_len = 50
Y = merged_df["sous famille"]

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
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=42)

# Combine Y_train and Y_test
Y_combined = np.concatenate((Y_train, Y_test))
Y_combined = Y_combined.astype(str)

# Fit a new LabelEncoder object on the combined set of Y_train and Y_test
le = LabelEncoder()
le.fit(Y_combined)

Y_combined_encoded = le.transform(Y_combined)

# Encode the labels in Y_train and Y_test as integers
Y_train_encoded = Y_combined_encoded[:len(Y_train)]
Y_test_encoded = Y_combined_encoded[len(Y_train):]

# Convert the encoded labels in Y_train and Y_test to one-hot encoded categorical representation
Y_train_categorical = to_categorical(Y_train_encoded, num_classes)
Y_test_categorical = to_categorical(Y_test_encoded, num_classes)

# Custom Embedding layer that handles out-of-vocabulary (OOV) index
class CustomEmbedding(Embedding):
    def __init__(self, input_dim, output_dim, **kwargs):
        super().__init__(input_dim, output_dim, **kwargs)

    def compute_mask(self, inputs, mask=None):
        return tensorflow.math.logical_and(tensorflow.greater_equal(inputs, 0), tensorflow.less(inputs, self.input_dim))

    def call(self, inputs):
        inputs = tensorflow.where(tensorflow.less(inputs, 0), tensorflow.ones_like(inputs) * (self.input_dim - 1), inputs)
        return super().call(inputs)

# Créer le modèle CNN-RNN
num_filters = 128
filter_sizes = [4, 5, 6]
lstm_units = 50
sequence_input = Input(shape=(max_len, embedding_size), dtype='int32')
embedded_sequences = CustomEmbedding(len(model_w2v.wv.index_to_key)+1,
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

X_train.shape


# In[30]:


# Entraîner le modèle sur l'ensemble d'entraînement
model.fit(X_train, Y_train_categorical, batch_size=64, epochs=10, validation_split=0.1)


# In[31]:


# Évaluer les performances du modèle sur l'ensemble de test
score = model.evaluate(X_test, Y_test_categorical, batch_size=12)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[32]:


model.summary()


# In[33]:


from PIL import ImageFont
import visualkeras
visualkeras.layered_view(model, legend=True)


# In[34]:


save_model(model, 'Perfect_Model.py')


# In[35]:


# Create an instance of TextProcessor
text_processor = TextProcessor()

# Preprocess the new text
new_text = ""
preprocessed_text = text_processor.process(new_text)

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

# Make the prediction
predictions = model.predict(padded_new_text_vec)
predicted_class_index = np.argmax(predictions[0])
predicted_class = le.inverse_transform([predicted_class_index])[0]

print("Predicted class:", predicted_class)


# In[16]:


def add_noise(text, noise_level):
    noisy_text = ''
    for char in str(text):
        if random.random() < noise_level:
            noisy_char = random.choice(string.ascii_letters)
        else:
            noisy_char = char
        noisy_text += noisy_char
    return noisy_text
noise_level = 0.1
augmented_rows5 = []
for index, row in merged_df.iterrows():
    text = row['description commande']
    perturbed_text = add_noise(text, noise_level)
    augmented_rows5.append({'description commande': perturbed_text, 'sous famille': row['sous famille']})
augmented_df5 = pd.DataFrame(augmented_rows5)

frames = [merged_df, augmented_df5]
merged_df = pd.concat(frames)


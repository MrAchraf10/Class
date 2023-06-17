import numpy as np

def vectorize_input_text(input_text_preprocessed, model_w2v, embedding_size):
    input_text_vec = []
    for word in input_text_preprocessed:
        if word in model_w2v.wv:
            input_text_vec.append(model_w2v.wv[word])
        else:
            input_text_vec.append(np.zeros(embedding_size))
    input_text_vec = np.array(input_text_vec)
    return input_text_vec
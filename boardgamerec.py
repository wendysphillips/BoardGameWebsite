from flask import Flask
app = Flask(__name__)
app.config.from_object(__name__)

#!/usr/bin/python
import flask, os, string, re, gensim
import tarfile
from flask import Flask, render_template, flash, request, redirect, url_for
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, LSTM
import gensim
from nltk.tokenize import TreebankWordTokenizer
from gensim.models.keyedvectors import KeyedVectors
word_vectors = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True, limit=200000)

from keras.models import model_from_json
with open("lstm_model1.json", "r") as json_file:
    json_string = json_file.read()
model = model_from_json(json_string)

def tokenize_and_vectorize_original(sentence):
    tokenizer = TreebankWordTokenizer()
    vectorized_data = []
    expected = []
    for sample in dataset:
        tokens = tokenizer.tokenize(sample[1])
        sample_vecs = []
        for token in tokens:
            try:
                sample_vecs.append(word_vectors[token])
            except KeyError:
                pass  # No matching token in the Google w2v vocab
        vectorized_data.append(sample_vecs)
    return vectorized_data
    
def pad_trunc(data, maxlen):
    """ For a given dataset pad with zero vectors or truncate to maxlen """
    new_data = []
    # Create a vector of 0's the length of our word vectors
    zero_vector = []
    for _ in range(len(data[0][0])):
        zero_vector.append(0.0)
    for sample in data:
         if len(sample) > maxlen:
            temp = sample[:maxlen]
         elif len(sample) < maxlen:
            temp = sample
            additional_elems = maxlen - len(sample)
            for _ in range(additional_elems):
                temp.append(zero_vector)
         else:
            temp = sample
         new_data.append(temp)
    return new_data


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('opentext.html')


@app.route('/forward', methods=['POST','GET'])
def answerquestion(sent):   
    maxlen = 300    
    embedding_dims = 300    
    vec_list = pad_trunc(sent, maxlen)
    vec_list = np.reshape(vec_list, (len(vec_list), maxlen, embedding_dims))
    answer = []
    answer = model.predict_classes(vec_list)
    x =int(answer[0])
    if x == 0:
        the_result =  print("Recommended game: this one")
    else:
        the_result = print("Recommended game: that one")    
    return render_template('output.html, the_result=the_result')

if __name__ == '__main__':
	app.run(host='0.0.0.0',debug=False)
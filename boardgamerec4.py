from flask import Flask
app = Flask(__name__, static_url_path='/static')
#app.config.from_object(__name__)

#!/usr/bin/python
import flask, os, string, re, gensim
from flask import Flask, render_template, flash, request, redirect, url_for

#not sure if necessary
import itertools
# **

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import confusion_matrix

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.preprocessing import text, sequence
from keras import utils

# Special config parameters to get tensorflow to run
config = tf.ConfigProto(
    device_count={'GPU': 1},
    intra_op_parallelism_threads=1,
    allow_soft_placement=True
)

config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.6

session = tf.Session(config=config)

tf.compat.v1.keras.backend.set_session
# end special parameters

#Load model
from keras.models import model_from_json
with open("/Users/wendy/UpYourGame/static/keras_model101.json", "r") as json_file:
    json_string = json_file.read()
model = model_from_json(json_string)

# Load files
gamedetails = pd.read_csv("/Users/wendy/UpYourGame/static/gameinfoForKeras.csv")
text_labels = np.load("/Users/wendy/UpYourGame/static/text_labels.npy", allow_pickle=True)

global graph
graph = tf.get_default_graph()

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('opentext6.html')

@app.route('/opentext6', methods=['GET', 'POST'])
def opentext2():
    return render_template('opentext6.html')

@app.route('/output5')
def output5():
    images = os.listdir(os.path.join(Users/wendy/UpYourGame/static, "images"))
    return render_template('output5.html', images=images)
    
@app.route('/output5', methods=['POST'])
def answerquestion():   
	sentence = str(request.form['search'])
	tokenize = text.Tokenizer(num_words=300, char_level=False)
	tokenize.fit_on_texts(sentence)
	input = tokenize.texts_to_matrix(sentence)
	input2 = np.asarray(input)
	with graph.as_default():
	    preds = model.predict([input2])
	prediction_label = text_labels[np.argsort(preds)]
	Pred = prediction_label[0][95:100]
	ans = gamedetails['name'][gamedetails['id'].isin(Pred)]
	j = pd.Series.tolist(ans)
	games = pd.DataFrame()
	for game in j:
		z=game
		y = gamedetails[(gamedetails['name']==game)]['url'].values[0]
		x = gamedetails[(gamedetails['name']==game)]['id'].values[0]
		#recom3 = (recom3 + z + "\n" + y + "\n\n")
		games.loc[game, 'name'] = z
		games.loc[game, 'url'] = y
		games.loc[game, 'img'] = str(x+".jpg") 
	return render_template("output5.html", games=games)

			
if __name__ == '__main__':
    #this runs your app locally
    app.run(host='0.0.0.0', port=8080, debug=True)

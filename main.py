import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request
from nltk.tokenize import RegexpTokenizer
from tensorflow.contrib import learn
import sys
import pickle
import pandas as pd

def load(filename):
    file = open(filename, 'rb')
    data = pickle.load(file)
    file.close()
    return data

def p(sth):
   print(sth, file=sys.stdout) 

taxonomyList = load('mnist/data/taxonomyList_r.pkl')

tokenizer = RegexpTokenizer(r'\w+')
_vocab_processor = learn.preprocessing.VocabularyProcessor.restore('mnist/data/vocabulary.vocab')
p(len(_vocab_processor.vocabulary_))

save_model_path = 'mnist/data/trained_model'

graph = tf.Graph()

logits = tf.placeholder(tf.float32, [None, 17])

with graph.as_default():
    sess = tf.Session()
    with sess.as_default():
        sess.run(tf.global_variables_initializer())
        # Load model
        loader = tf.train.import_meta_graph(save_model_path + '.meta')
        loader.restore(sess, save_model_path)

        input_y = graph.get_operation_by_name("input_y").outputs[0]
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout").outputs[0]
        logits = graph.get_operation_by_name("logits").outputs[0]

def taxonomy(request, logits):
    text = request['text']
    text = " ".join(tokenizer.tokenize(text.lower()))
    p(text)
    x = np.array(list(_vocab_processor.transform([text])))
    feed_dict = {
      input_x: x,
      dropout_keep_prob: 1.0
    }
    logits = sess.run([logits], feed_dict)
    prediction = np.argmax(logits)

    result = {}
    result['prediction'] = taxonomyList[prediction]['text']
    weights = logits[0][0]
    fullResult = []
    for i in range(len(weights)):
        r = {'weight':str(weights[i]), 'taxonomy':taxonomyList[i]['text']}
        fullResult.append(r)
    result['fullResult'] = fullResult
    p(result)
    return result

# webapp
app = Flask(__name__)

@app.route('/api/taxonomy', methods=['POST'])
def taxonomyHandle():
    result = taxonomy(request.json, logits)
    return jsonify(result=result)

@app.route('/api/all_taxonomy', methods=['GET'])
def taxonomysHandle():
    return jsonify(result=taxonomyList)

@app.route('/')
def main():
    return render_template('index.html')


if __name__ == '__main__':
    app.run()

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request
from nltk.tokenize import RegexpTokenizer
from tensorflow.contrib import learn
import sys
import pickle

def load(filename):
    file = open(filename, 'rb')
    data = pickle.load(file)
    file.close()
    return data

def p(sth):
   print(sth, file=sys.stdout) 

taxonomyList = load('mnist/PocData/taxonomyList_r.pkl')

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

def classification(request, logits):
    text = request['text']
    text = " ".join(tokenizer.tokenize(text.lower()))
    p(text)
    x = np.array(list(_vocab_processor.transform([text])))
    feed_dict = {
      input_x: x,
      dropout_keep_prob: 1.0
    }
    logits = sess.run([logits], feed_dict)
    logits = np.array(logits) + 1
    p(logits)
    prediction = np.argmax(logits)
    p(prediction)
    return taxonomyList[prediction]['text']

# webapp
app = Flask(__name__)

@app.route('/api/classification', methods=['POST'])
def classificationHandle():
    result = classification(request.json, logits)
    return jsonify(result=result)


@app.route('/')
def main():
    return render_template('index.html')


if __name__ == '__main__':
    app.run()

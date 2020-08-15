import os
import tensorflow as tf
from base64 import b64encode
from flask import Flask, render_template, request


app = Flask(__name__)

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/check', methods=['POST'])
def check():
    file = request.files['inputFile']
    image_data=file.read()


    label_lines = [line.rstrip() for line in tf.io.gfile.GFile("model/face/retrained_labels.txt")]

    # Unpersists graph from file
    with tf.compat.v1.gfile.FastGFile("model/face/retrained_graph.pb", 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

    with tf.compat.v1.Session() as sess:
        # Feed the image_data as input to the graph and get first prediction
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        
        predictions = sess.run(softmax_tensor, \
                {'DecodeJpeg/contents:0': image_data})
        
        # Sort to show labels of first prediction in order of confidence
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
        
        for node_id in top_k:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            print('%s (score = %.5f)' % (human_string, score))
            all = list()
            for root, dirs, files in os.walk(('static/glasses/{}').format(human_string)):
                if files:
                    for i in files:
                        all.append(root+'/'+i)
            return render_template('result.html',
                                    face_type=human_string,
                                    dir=all)

app.run(debug=True)
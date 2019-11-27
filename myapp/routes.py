from myapp import app
import io
import os
import json, plotly
from flask import render_template, request
from wrangling_scripts.load_model import init_model
from PIL import Image
from util import decode_prob
from wrangling_scripts.wrangle_data import DataWrangler

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html', id="pred", preds='Hier kommen die Predictions hin')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """File selection and display results
    """

    if request.method == 'POST' and 'file[]' in request.files:
        inputs = []
        files = request.files.getlist('file[]')
        for file_obj in files:
            # Check if no files uploaded
            if file_obj.filename == '':
                if len(files) == 1:
                    return render_template('select_files.html')
                continue
            entry = {}
            entry.update({'filename': file_obj.filename})
            try:
                img_bytes = io.BytesIO(file_obj.stream.getvalue())
                entry.update({'data':
                              Image.open(
                                  img_bytes
                              )})
            except AttributeError:
                img_bytes = io.BytesIO(file_obj.stream.read())
                entry.update({'data':
                              Image.open(
                                  img_bytes
                              )})
            # keep image in base64 for later use
            img_b64 = base64.b64encode(img_bytes.getvalue()).decode()
            entry.update({'img': img_b64})

            inputs.append(entry)

        outputs = []

        with graph.as_default():
            for input_ in inputs:
                # convert to 4D tensor to feed into our model
                x = preprocess(input_['data'])
                # perform prediction
                out = model.predict(x)
                outputs.append(out)

        # decode output prob
        outputs = decode_prob(outputs)

        results = []

        for input_, probs in zip(inputs, outputs):
            results.append({'filename': input_['filename'],
                            'image': input_['img'],
                            'predict_probs': probs})

        return render_template('results.html', results=results)

    # if no files uploaded
    return render_template('select_files.html')

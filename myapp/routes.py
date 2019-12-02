from werkzeug.utils import secure_filename
from myapp import app
import io
import os
import json, plotly
from flask import Flask, render_template, request, url_for, flash, redirect, send_from_directory, Markup
from PIL import Image
from DogBreedPredictor import DogBreedPredictor_v2
import base64
from PIL import Image

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

dbp = DogBreedPredictor_v2()

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """File selection and display results
    """
    if request.method == 'POST':
        # double checking if a file has been uploaded
        if 'file[]' not in request.files:
            return render_template('index.html', preds="Error predicting breed. file not found.")
        file = request.files['file[]']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            prediction = dbp.getPredictions("." + url_for('uploaded_file', filename=filename))
            print(prediction)
            display_image = url_for('uploaded_file', filename=filename)
            print(display_image)
            full_filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            print(full_filename)

            display_image = Markup(f'<img src="{display_image}" alt="Image used for prediction" height="250">')

            return render_template('show_prediction.html', prediction=" ".join(prediction), predicted_image=display_image)
    return render_template('index.html')

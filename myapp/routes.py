from werkzeug.utils import secure_filename

from myapp import app
import io
import os
import json, plotly
from flask import Flask, render_template, request, url_for, flash, redirect, send_from_directory
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

# @app.route('/predict', methods=['GET', 'POST'])
# def predict():
#     """File selection and display results
#     """
#     if request.method == 'POST' and 'file[]' in request.files:
#         inputs = []
#         files = request.files.getlist('file[]')
#         for file_obj in files:
#             # Check if no files uploaded
#             if file_obj.filename == '':
#                 if len(files) == 1:
#                     return render_template('index.html', preds='Konnte noch keine Aussagen machen')
#                 continue
#             entry = {}
#             entry.update({'filename': file_obj.filename})
#             try:
#                 img_bytes = io.BytesIO(file_obj.stream.getvalue())
#                 entry.update({'data':
#                               Image.open(
#                                   img_bytes
#                               )})
#             except AttributeError:
#                 img_bytes = io.BytesIO(file_obj.stream.read())
#                 entry.update({'data':
#                               Image.open(
#                                   img_bytes
#                               )})
#             # keep image in base64 for later use
#             img_b64 = base64.b64encode(img_bytes.getvalue()).decode()
#             entry.update({'img': img_b64})
#
#             inputs.append(entry)
#
#         outputs = []
#
#         for input_ in inputs:
#             # perform prediction
#             out = dbp.getPredictions(input_['data'])
#             outputs.append(out)
#
#
#         return render_template('predict.html', preds=outputs[0])
#
#     # if no files uploaded
#     return render_template('index.html')


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
        print('POST method')
        # check if the post request has the file part
        print(request.files)
        if 'file[]' not in request.files:
            #flash('No file part')
            print('kein File vorhanden')
            return redirect(request.url)
        file = request.files['file[]']
        print(file)
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        print(file.filename)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print(url_for('uploaded_file', filename=filename))
            print('starting prediction')
            prediction = dbp.getPredictions("." + url_for('uploaded_file', filename=filename))
            print(prediction)
            print('rendering html')

            #return redirect(url_for('uploaded_file', filename=filename))
            return render_template('index.html', preds=url_for('uploaded_file', filename=filename))


    #
    #
    #
    #
    # if request.method == 'POST' and 'file[]' in request.files:
    #     print
    #     inputs = []
    #     outputs = []
    #     files = request.files.getlist('file[]')
    #     file = request.files.ge
    #     for file in files:
    #         # Check if no files uploaded
    #         if file.filename == '':
    #             if len(files) == 1:
    #                 return render_template('index.html', preds='Konnte noch keine Aussagen machen')
    #             continue
    #         entry = {}
    #         entry.update({'filename': file.filename})
    #
    #         filename = file.filename
    #         file.save(os.path.join("uploads", filename))
    #         image_url = url_for('uploaded_file', filename=filename)
    #
    #         out = dbp.getPredictions(image_url)
    #         outputs.append(out)
    #
    #
    #
    #
    #     return render_template('predict.html', preds=outputs[0])

    # if no files uploaded
    return render_template('index.html')

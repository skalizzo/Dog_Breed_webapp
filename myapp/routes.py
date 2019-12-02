from werkzeug.utils import secure_filename
from myapp import app
import os
from flask import Flask, render_template, request, url_for, flash, redirect, send_from_directory, Markup
from DogBreedPredictor import DogBreedPredictor_v2

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
dbp = DogBreedPredictor_v2()

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    print(f'uploaded file gestartet für {filename}')
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/static/uploads/<filename>')
def show_file2(filename):
    print('show file gestartet')
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/show/<filename>')
def show_file(filename):
    print('showing file')
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """File selection and display results
    """
    error_message = ""
    if request.method == 'POST':
        # double checking if a file has been uploaded
        if 'file[]' not in request.files:
            error_message="Error predicting breed. file not found."
        else:
            file = request.files['file[]']
            # if user does not select file, browser also
            # submit an empty part without filename
            if file.filename == '':
                error_message = 'No selected file'
            else:
                if file and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    print(f"saving to {os.path.join(app.config['UPLOAD_FOLDER'], filename)}")
                    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                    image_path = url_for('uploaded_file', filename=filename)
                    prediction = dbp.getPredictions("." + image_path)
                    #display_image = Markup(f"""<img src="{app.root_path + '/show/' + filename}" alt="Image used for prediction" height="250">""")
                    return render_template('show_prediction.html', prediction=" ".join(prediction))
    return render_template('index.html', error_message=error_message)

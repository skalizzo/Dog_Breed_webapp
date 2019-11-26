from myapp import app
import json, plotly
from flask import render_template
from wrangling_scripts.wrangle_data import DataWrangler

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html', id="pred", preds='Hier kommen die Predictions hin')

@app.route('/prediction')
def prediction():
    datawrangler = DataWrangler()
    print('wrangled data')
    preds = datawrangler.make_prediction("C://work_local//Python//DataScientistND_Project6_Capstone_Project//Dog_Breed//sample_imgs//Bernhardiner.jpg")

    return render_template('index.html', id="pred", preds=preds)
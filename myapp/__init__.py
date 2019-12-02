import os
from flask import Flask, flash, request, redirect, url_for, Blueprint
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = '.\\uploads'
#UPLOAD_FOLDER = '/static/uploads'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'super_secret_key'
app.config['SESSION_TYPE'] = 'filesystem'

from myapp import routes

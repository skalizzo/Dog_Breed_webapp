import os
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = '.\\uploads'

app = Flask(__name__,
            static_url_path='',
            static_folder='myapp/static',
            template_folder='myapp/templates',
            )
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

from myapp import routes

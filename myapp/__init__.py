from flask import Flask

UPLOAD_FOLDER = '.\\uploads'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'super_secret_key'
app.config['SESSION_TYPE'] = 'filesystem'

from myapp import routes

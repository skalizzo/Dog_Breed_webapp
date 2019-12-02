# Dog Breed Classification Webapp

This webapp uses the Flask framework to deploy a trained model based on ResNet50 
that is used to classify dog breeds in images. If you'd upload images of humans 
to the app it should recognise them as humans and give you the dog breed that 
resembles these humans most. 
 
This webapp is part of the Capstone project for the Udacity Data 
Scientist Nanodegree program. 

The corresponding Jupyter notebook where the model has been built can 
be found here: [https://github.com/skalizzo/Dog_Breed_Classification](https://github.com/skalizzo/Dog_Breed_Classification)

### Technologies used
<li>Python 3.7</li>
<li>Flask 1.1.1</li>
<li>Jinja2 2.10.3</li>
<li>Werkzeug 0.16.0</li>
<li>Keras 2.2.5</li>
<li>tensorflow 2.0.0 </li>
<li>OpenCV 4.1.2.30</li>
<li>numpy 1.16.1</li>



### Installation/Usage
1. Clone this repository
2. Install the required packages (see above)
3. Run with Python: $ python app.py
4. Open browser at http://localhost:5000, select an image on your computer 
and start the prediction using the "Predict Dog Breed" button

If you want to modify the UI you can modify the files in the templates directory.
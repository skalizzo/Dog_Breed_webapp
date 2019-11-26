import pandas as pd
import numpy as np
import cv2
import pickle
import plotly.graph_objs as go
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras_preprocessing import image


# Use this file to read in your data and prepare the plotly visualizations. The path to the data files are in
# `data/file_name.csv`


class DataWrangler():
    def __init__(self):
        print('loading model')
        self.model = self.loadModel()
        print('model loaded')
        # load list of dog names
        self.dog_names = pickle.load(open("dog_names.p", "rb"))

    def make_prediction(self, img_path):
        #pred = self.Resnet50_predict_breed(img_path)
        pred = self._getPredictions(img_path)
        print(" ".join(pred))
        return " ".join(pred)

    def _path_to_tensor(self, img_path):
        # loads RGB image as PIL.Image.Image type
        img = image.load_img(img_path, target_size=(224, 224))
        # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
        x = image.img_to_array(img)
        # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
        return np.expand_dims(x, axis=0)

    def _getPredictions(self, img):
        """
        checks if the image is predicted to be a dog, a human
        or something else;
        if it is a dog it tries to predict its breed
        """
        predictions = []
        breed = self.Resnet50_predict_breed(img)
        breed = str(breed).split('.')[-1]
        if self.face_detector(img):
            predictions.append("You're probably a human!")
            predictions.append(f"You look like a {breed}")
        elif self.dog_detector(img):
            predictions.append(f'Woof! You look like a {breed}')
        else:
            predictions.append("I am not sure what kind of species you are.")
        return predictions

    def loadModel(self):
        Resnet50_model = Sequential()
        Resnet50_model.add(GlobalAveragePooling2D(input_shape=(7, 7, 2048)))
        Resnet50_model.add(Dense(1000, activation='relu'))
        Resnet50_model.add(Dropout(0.3))
        Resnet50_model.add(Dense(133, activation='softmax'))
        Resnet50_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        Resnet50_model.load_weights('saved_models/weights.best.Resnet50_v2.hdf5')
        return Resnet50_model

    def extract_Resnet50(self, tensor):
        return ResNet50(weights='imagenet', include_top=False).predict(preprocess_input(tensor))

    def Resnet50_predict_breed(self, img_path):
        """
        uses our Resnet50 model to make a dog-breed-prediction for an image
        """
        # extract bottleneck features
        bottleneck_feature = self.extract_Resnet50(self._path_to_tensor(img_path))
        # obtain predicted vector
        predicted_vector = self.model.predict(bottleneck_feature)
        # return dog breed that is predicted by the model
        return self.dog_names[np.argmax(predicted_vector)]


    def face_detector(self, img_path):
        face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray)
        return len(faces) > 0

    def dog_detector(self, img_path):
        prediction = self.ResNet50_predict_labels(img_path)
        return ((prediction <= 268) & (prediction >= 151))

    def ResNet50_predict_labels(self, img_path):
        # define ResNet50 model
        ResNet50_model = ResNet50(weights='imagenet')
        # returns prediction vector for image located at img_path
        img = preprocess_input(self._path_to_tensor(img_path))
        return np.argmax(ResNet50_model.predict(img))


    def return_figures(self):
        """Creates four plotly visualizations

        Args:
            None

        Returns:
            list (dict): list containing the four plotly visualizations

        """

        # first chart plots arable land from 1990 to 2015 in top 10 economies
        # as a line chart

        graph_one = []
        graph_one.append(
          go.Scatter(
          x = [0, 1, 2, 3, 4, 5],
          y = [0, 2, 4, 6, 8, 10],
          mode = 'lines'
          )
        )

        layout_one = dict(title = 'Chart One',
                    xaxis = dict(title = 'x-axis label'),
                    yaxis = dict(title = 'y-axis label'),
                    )

    # second chart plots ararble land for 2015 as a bar chart
        graph_two = []

        graph_two.append(
          go.Bar(
          x = ['a', 'b', 'c', 'd', 'e'],
          y = [12, 9, 7, 5, 1],
          )
        )

        layout_two = dict(title = 'Chart Two',
                    xaxis = dict(title = 'x-axis label',),
                    yaxis = dict(title = 'y-axis label'),
                    )


    # third chart plots percent of population that is rural from 1990 to 2015
        graph_three = []
        graph_three.append(
          go.Scatter(
          x = [5, 4, 3, 2, 1, 0],
          y = [0, 2, 4, 6, 8, 10],
          mode = 'lines'
          )
        )

        layout_three = dict(title = 'Chart Three',
                    xaxis = dict(title = 'x-axis label'),
                    yaxis = dict(title = 'y-axis label')
                           )

    # fourth chart shows rural population vs arable land
        graph_four = []

        graph_four.append(
          go.Scatter(
          x = [20, 40, 60, 80],
          y = [10, 20, 30, 40],
          mode = 'markers'
          )
        )

        layout_four = dict(title = 'Chart Four',
                    xaxis = dict(title = 'x-axis label'),
                    yaxis = dict(title = 'y-axis label'),
                    )

        # append all charts to the figures list
        figures = []
        figures.append(dict(data=graph_one, layout=layout_one))
        figures.append(dict(data=graph_two, layout=layout_two))
        figures.append(dict(data=graph_three, layout=layout_three))
        figures.append(dict(data=graph_four, layout=layout_four))

        return figures


if __name__ == '__main__':
    dbp = DataWrangler()
    dbp.make_prediction("C://work_local//Python//DataScientistND_Project6_Capstone_Project//Dog_Breed//sample_imgs//Bernhardiner.jpg")
from keras.models import load_model
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.preprocessing import image
from keras import backend as K
import tensorflow as tf

import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt


class DogBreedPredictor_v2():
    """
    this class contains every function that is needed for our dog breed prediction
    """

    def __init__(self):
        # load list of dog names
        K.clear_session()
        tf.compat.v1.reset_default_graph()
        global model
        self.dog_names = pickle.load(open("saved_models/dog_names.p", "rb"))
        model = load_model('saved_models/Resnet50_model.h5')
        global graph
        graph = tf.get_default_graph()

    def getPredictions(self, img):
        """
        checks if the image is predicted to be a dog, a human
        or something else;
        if it is a dog it tries to predict its breed
        """
        print('getting predictions')
        predictions = []
        K.clear_session()
        with graph.as_default():
            breed, confidence = self.Resnet50_predict_breed(img)
            breed = str(breed).split('.')[-1]
            #K.clear_session()
            if self._face_detector(img):
                predictions.append("You're probably a human!")
                predictions.append(f"You look like a {breed}")
            elif self._dog_detector(img):
                predictions.append(f'Woof! You look like a {breed}. I am {confidence}% sure.')
            else:
                predictions.append("I am not sure what kind of species you are.")
        return predictions

    def Resnet50_predict_breed(self, img_path):
        """
        uses our Resnet50 model to make a dog-breed-prediction for an image
        """
        # extract bottleneck features
        print(f'extracting bottleneck feature for {img_path}')
        with graph.as_default():
            bottleneck_feature = self._extract_Resnet50(self._path_to_tensor(img_path))
            # obtain predicted vector

            predicted_vector = model.predict(bottleneck_feature, batch_size=1, verbose=0)
            confidence = round(np.max(predicted_vector) * 100, 2)
            pred_label = self.dog_names[np.argmax(predicted_vector)]
            # return dog breed that is predicted by the model
            return pred_label, confidence

    def _path_to_tensor(self, img_path):
        """
        converts a given image to a 4D tensor
        """
        # loads RGB image as PIL.Image.Image type
        print(img_path)
        img = image.load_img(img_path, target_size=(224, 224))
        # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
        x = image.img_to_array(img)
        # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
        return np.expand_dims(x, axis=0)

    def _extract_Resnet50(self, tensor):
        print(tensor)
        return ResNet50(weights='imagenet', include_top=False).predict(preprocess_input(tensor))

    def _face_detector(self, img_path):
        """
        returns "True" if face is detected in image stored at img_path
        """
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
        faces = face_cascade.detectMultiScale(gray)
        return len(faces) > 0

    def _dog_detector(self, img_path):
        """
        returns "True" if a dog is detected in the image stored at img_path
        """
        with graph.as_default():
            img = preprocess_input(self._path_to_tensor(img_path))
            prediction = np.argmax(ResNet50(weights='imagenet').predict(img))
            return ((prediction <= 268) & (prediction >= 151))


if __name__ == '__main__':
    dbp = DogBreedPredictor_v2()
    preds = dbp.getPredictions("C://work_local//Python//DataScientistND_Project6_Capstone_Project//Dog_Breed//sample_imgs//Bernhardiner.jpg")
    print(preds)
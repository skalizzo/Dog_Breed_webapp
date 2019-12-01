from tensorflow.keras.models import load_model, model_from_json
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
#from keras import backend as K
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.keras.backend import clear_session, set_session
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Dropout, Flatten, Dense
from tensorflow.keras.models import Sequential
print('comp mode on')
tf.disable_v2_behavior()

# import keras.backend.tensorflow_backend as tb
# tb._SYMBOLIC_SCOPE.value = True

import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt

class Model_cnn():
    def __init__(self):
        Resnet50_model = Sequential()
        Resnet50_model.add(GlobalAveragePooling2D(input_shape=(1, 1, 2048)))
        Resnet50_model.add(Dense(1000, activation='relu'))
        Resnet50_model.add(Dropout(0.3))
        Resnet50_model.add(Dense(133, activation='softmax'))
        Resnet50_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        Resnet50_model.load_weights('saved_models/weights.best.Resnet50_v2.hdf5')

print('session start')

sess = tf.Session()

graph = tf.get_default_graph()

with graph.as_default():
    set_session(sess)
    model = Model_cnn()
    model = load_model('saved_models/Resnet50_model.h5', compile=True)
#model = model_from_json('saved_models/model.Resnet50.json')

#init = tf.global_variables_initializer()
#init = tf.initialize_all_variables()
#init = tf.variables_initializer()

#sess.run(init)
#model.load_weights('saved_models/weights.best.Resnet50_v2.hdf5')
print('loading class')
class DogBreedPredictor_v2():
    """
    this class contains every function that is needed for our dog breed prediction
    """

    def __init__(self):
        self.dog_names = pickle.load(open("saved_models/dog_names.p", "rb"))

    def getPredictions(self, img):
        """
        checks if the image is predicted to be a dog, a human
        or something else;
        if it is a dog it tries to predict its breed
        """
        predictions = []
        global graph
        global model
        global sess
        with graph.as_default():
            set_session(sess)
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
        global graph
        global model
        global sess
        with graph.as_default():
            set_session(sess)
            bottleneck_feature = self._extract_Resnet50(self._path_to_tensor(img_path))
            # obtain predicted vector
            print(bottleneck_feature.shape)  # returns (1, 2048)
            bottleneck_feature = np.expand_dims(bottleneck_feature, axis=0)
            bottleneck_feature = np.expand_dims(bottleneck_feature, axis=0)

            #predicted_vector = model.predict(bottleneck_feature, batch_size=1, verbose=0)

            predicted_vector = model.predict(bottleneck_feature, verbose=0)
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
        global graph
        global model
        global sess
        with graph.as_default():
            set_session(sess)
            prediction = ResNet50(weights='imagenet', include_top=False, pooling="avg", input_shape=(224, 224, 3)).predict(preprocess_input(tensor))
            return prediction

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
        global graph
        global model
        global sess
        with graph.as_default():
            set_session(sess)
            img = preprocess_input(self._path_to_tensor(img_path))
            prediction = np.argmax(ResNet50(weights='imagenet').predict(img))
            return ((prediction <= 268) & (prediction >= 151))

class Model_cnn():
    def __init__(self):
        Resnet50_model = Sequential()
        Resnet50_model.add(GlobalAveragePooling2D(input_shape=(1, 1, 2048)))
        Resnet50_model.add(Dense(1000, activation='relu'))
        Resnet50_model.add(Dropout(0.3))
        Resnet50_model.add(Dense(133, activation='softmax'))

        Resnet50_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

        Resnet50_model.load_weights('saved_models/weights.best.Resnet50_v2.hdf5')

if __name__ == '__main__':
    dbp = DogBreedPredictor_v2()
    preds = dbp.getPredictions("uploads/Labrador_retriever_06455.jpg")
    print(preds)
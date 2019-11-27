from typing import Tuple
from keras.models import model_from_json
import tensorflow as tf
from keras.preprocessing import image
import numpy as np


IMAGE_DIM = (224, 224)

def init_model(model="./saved_models/model.Resnet50.json", weights="./saved_models/weights.best.Resnet50_v2.hdf5") -> Tuple:
    """Initialize the keras model
    :param model, str: path to model.json
    :param weights, str: path to weights for model
    :return keras model and graph
    """
    print('loading pickled model')
    with open(model, 'r') as json_file:
        loaded_model_json = json_file.read()

    loaded_model = model_from_json(loaded_model_json)
    # load woeights into new model
    print('loading weights for model')
    loaded_model.load_weights(weights)

    # compile and evaluate loaded model
    print('compiling model')
    loaded_model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])
    graph = tf.compat.v1.get_default_graph()


    return loaded_model, graph



if __name__ == '__main__':
    mod, graphobj = init_model()
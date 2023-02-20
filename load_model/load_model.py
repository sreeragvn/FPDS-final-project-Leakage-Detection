from utility import *
from layers_class import *
from keras.utils import CustomObjectScope

import os
print(os.getcwd())
cwd = os.getcwd()

from os import listdir
from os.path import isfile, join
path = cwd+"/models"
models = [f for f in listdir(path) if isfile(join(path, f))]
print(models)

for model in models:
       with CustomObjectScope({'Hidden_layer':Hidden_layer, 'Output_layer':Output_layer}):
              load_model = tf.keras.models.load_model('./models/NN_1000_Aug.h5')
# load_model.summary()
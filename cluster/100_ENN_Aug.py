import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn import preprocessing
# import tensorflow_datasets as tfds

from sklearn.metrics import mean_squared_error
from tensorflow import keras
from keras import layers
# from sklearn import preprocessing
# Make numpy values easier to read.
# np.set_printoptions(precision=3, suppress=True)

# physical_devices = tf.config.list_physical_devices("gpu")
print("# GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

tf.keras.backend.set_floatx('float64')
tf.compat.v1.enable_eager_execution()

# np.random.seed(101)
# tf.random.set_seed(101)

import warnings
warnings.filterwarnings('ignore')

# loading the data of data

leakage_train_100 = pd.read_csv("leakage_dataset_train_100.csv")
leakage_train_1000 = pd.read_csv("leakage_dataset_train_1000.csv")
leakage_val_1000 = pd.read_csv("leakage_dataset_validation_1000.csv")

def load_data(train_data):

    train_ds = train_data
    val_ds = leakage_val_1000

    train_ds = train_ds.sample(frac=1)
    val_ds = val_ds.sample(frac=1)

    X_train = train_ds.iloc[:,2:].to_numpy()
    Y_train = train_ds.iloc[:,:2]

    X_validation = val_ds.iloc[:,2:].to_numpy()
    Y_validation = val_ds.iloc[:,:2]

    Y_train = Y_train.to_numpy()
    Y_validation = Y_validation.to_numpy()
    return X_train, Y_train, X_validation, Y_validation

X_train, Y_train, X_validation, Y_validation = load_data(leakage_train_100)


#test dataset
X_test = np.array([[0.25, 0.25, 0.25,0.25], 
                    [0.35, 0.15, 0.25,0.25], 
                    [0.25, 0.25, 0.15,0.35],
                    [0.15, 0.35, 0.25,0.25],
                    [0.25, 0.25, 0.35,0.15],
                    [0.4, 0.1, 0.25,0.25],
                    [0.1, 0.4, 0.25,0.25],
                    [0.42, 0.08, 0.25,0.25],
                    [0.25, 0.25, 0.4,0.1],
                    [0.25, 0.25, 0.1,0.4]])
                    
# scX = preprocessing.StandardScaler()
# stY = preprocessing.StandardScaler()

# def data_scaling(X_train, X_validation, Y_train, Y_validation, scX, stY):
#     X_train = scX.fit_transform(X_train)
#     X_validation = scX.transform(X_validation)

#     Y_train = stY.fit_transform(Y_train)
#     Y_validation = stY.transform(Y_validation)

#     return X_train, X_validation, Y_train, Y_validation

# X_train, X_validation, Y_train, Y_validation = data_scaling(X_train, X_validation, Y_train, Y_validation, scX, stY)

X_train = tf.convert_to_tensor(X_train)
X_validation = tf.convert_to_tensor(X_validation)
Y_train = tf.convert_to_tensor(Y_train)
Y_validation = tf.convert_to_tensor(Y_validation)
X_test = tf.convert_to_tensor(X_test)

# num_rows, num_cols = X_train.shape

batch_size = 32
epochs = 1000
# steps_per_epoch = sum(train_occurences) / batch_size
starter_learning_rate = 1e-1
end_learning_rate = 1e-8
decay_steps = epochs * 3
scheduler = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate= starter_learning_rate,
    decay_steps= decay_steps,
    end_learning_rate= end_learning_rate,
    power=1)
kernel_regularizer=tf.keras.regularizers.L1L2(0.01)
callbacks=[keras.callbacks.EarlyStopping(monitor='val_mean_squared_error',patience=20)]
initializer=tf.keras.initializers.HeUniform

def learning_curves(history)   :
    sns.set_style('darkgrid', {'axes.facecolor': '.9'})
    sns.set_context('notebook')

    # your code
    ### Learning curves
    history_frame = pd.DataFrame(history.history)
    history_frame.plot(figsize=(8, 5))
    plt.show()

def prediction_accuracy(predictions, Y_validation): 
    predictions = predictions.transpose()
    Y_validation = tf.transpose(Y_validation)
    y1 = predictions[0]
    y2 = predictions[1]
    y1_validation = Y_validation[0]
    y2_validation = Y_validation[1]
    fig, axs = plt.subplots(2)
    # print(y1_validation.shape, y1.shape)
    # print(y2_validation.shape, y2.shape)
    # fig.suptitle('')
    axs[0].scatter(y1_validation, y1)
    axs[0].set_title('y1')
    axs[1].scatter(y2_validation, y2)
    axs[1].set_title('y2')
    for ax in axs.flat:
        ax.set(xlabel='true value', ylabel='predicted value')
    for ax in axs.flat:
        ax.label_outer()

    print("rmse of y1: ", mean_squared_error(y1_validation, y1, squared=False))
    print("rmse of y2: ", mean_squared_error(y2_validation, y2, squared=False))

# def dataloading(data):
#     data = data.repeat()
#     data = data.shuffle(buffer_size=1024, seed=0)
#     data = data.batch(batch_size=batch_size)
#     data = data.prefetch(buffer_size=1)
#     return data

# data Augmentation
# Requires cleaning up

def rotation_matrix(angle):
    theta = np.radians(angle)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    return R

def Augmentation_clock(x,y):

    x = x.copy()
    y = y.copy()
    # print(y)
    y_aug = np.transpose(np.matmul(rotation_matrix(-90), np.transpose(y)))
    # print(y_aug)

    temp = x.copy()
    x0 = temp[:,0]
    x1 = temp[:,1]
    x2 = temp[:,2]
    x3 = temp[:,3]

    x[:,0] = x3
    x[:,1] = x0
    x[:,2] = x1
    x[:,3] = x2
 
    return x,y_aug

def Augmentation_flip(x,y):
    x = x.copy()
    y = y.copy()
    x = np.flip(x, axis=1)
    y[:,1] = -1 * y[:,1]
    return x,y

def Augmentation_anticlock(x,y):

    x = x.copy()
    y = y.copy()
    y_aug = np.transpose(np.matmul(rotation_matrix(90), np.transpose(y)))

    temp = x.copy()
    x0 = temp[:,0]
    x1 = temp[:,1]
    x2 = temp[:,2]
    x3 = temp[:,3]

    x[:,0] = x1
    x[:,1] = x2
    x[:,2] = x3
    x[:,3] = x0
 
    return x,y_aug

def data_augmentation(x,y):
    x = x.numpy()
    y = y.numpy()
    x_aug1,y_aug1 = Augmentation_clock(x, y)
    x_aug2,y_aug2 = Augmentation_clock(x_aug1,y_aug1)
    x_aug3,y_aug3 = Augmentation_clock(x_aug2,y_aug2)
    x_aug4,y_aug4 = Augmentation_flip(x_aug3,y_aug3)
    x_aug5,y_aug5 = Augmentation_clock(x_aug4,y_aug4)
    x_aug6,y_aug6 = Augmentation_clock(x_aug5,y_aug5)
    x_aug7,y_aug7 = Augmentation_clock(x_aug6,y_aug6)
    X_train_Aug = np.concatenate((x, x_aug1, x_aug2, x_aug3, x_aug4, x_aug5, x_aug6, x_aug7))
    Y_train_Aug = np.concatenate((y, y_aug1, y_aug2, y_aug3, y_aug4, y_aug5, y_aug6, y_aug7))

    X_train_Aug = tf.convert_to_tensor(X_train_Aug)
    Y_train_Aug = tf.convert_to_tensor(Y_train_Aug)

    return X_train_Aug, Y_train_Aug

X_train_Aug, Y_train_Aug = data_augmentation(X_train, Y_train)
# X_validation_Aug, Y_validation_Aug = data_augmentation(X_validation, Y_validation)

# Equivariant NN - on Aug data

class Hidden_layer(layers.Layer):
    def __init__(self, units, kernel_regularizer, initializer):
        super(Hidden_layer, self).__init__()
        self.units = units
        self.kernel_regularizer = kernel_regularizer
        self.initializer = initializer

    def build(self, input_shape):
        self.a = self.add_weight(shape=(1,), initializer=initializer,
                                 trainable=True, regularizer=self.kernel_regularizer)
        self.b = self.add_weight(shape=(1,), initializer=initializer,
                                 trainable=True, regularizer=self.kernel_regularizer)
        self.c = self.add_weight(shape=(1,), initializer=initializer,
                                 trainable=True, regularizer=self.kernel_regularizer)
        self.a_matrix = tf.constant([[1,0,0,0], [0,1,0,0], [0,0,1,0],[0,0,0,1]], dtype=tf.float64)
        self.b_matrix = tf.constant([[0,1,0,1], [1,0,1,0], [0,1,0,1],[1,0,1,0]], dtype=tf.float64)
        self.c_matrix = tf.constant([[0,0,1,0], [0,0,0,1], [1,0,0,0],[0,1,0,0]], dtype=tf.float64)
    
    def call(self, inputs):
        self.W = tf.multiply(self.a, self.a_matrix) + tf.multiply(self.b, self.b_matrix) + tf.multiply(self.c, self.c_matrix)
        x = tf.matmul(inputs, self.W)
        # tf.print(self.W)
        return x

class Output_layer(layers.Layer):
    def __init__(self, units, kernel_regularizer, initializer):
        super(Output_layer, self).__init__()
        self.units = units
        self.kernel_regularizer = kernel_regularizer
        self.initializer = initializer

    def build(self, input_shape):
        self.d = self.add_weight(shape=(1,), initializer=initializer,
                                 trainable=True, regularizer=self.kernel_regularizer)
        self.d_matrix = tf.constant([[1,-1], [-1,-1], [-1,1],[1,1]], dtype=tf.float64)

    def call(self, inputs):
        self.W = tf.multiply(self.d, self.d_matrix)
        x = tf.matmul(inputs, self.W)
        # tf.print(tf.transpose(self.W))
        return x

class MyReLU(layers.Layer):
    def __init__(self):
        super(MyReLU, self).__init__()

    def call(self, x):
        return tf.math.maximum(x, 0)

model = keras.models.Sequential()
model.add(Hidden_layer(4, kernel_regularizer=kernel_regularizer, initializer=initializer))
model.add(MyReLU())
model.add(Output_layer(2, kernel_regularizer=kernel_regularizer, initializer=initializer))

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=scheduler),
              loss='mean_squared_error',
              metrics = [tf.keras.metrics.MeanSquaredError()]
)

history = model.fit(X_train_Aug, Y_train_Aug, epochs=epochs, batch_size= batch_size, verbose=2, validation_data=(X_validation, Y_validation),
                    callbacks=callbacks,
                    # shuffle=True
                    )

predictions = model.predict(X_validation)
prediction_accuracy(predictions, Y_validation)
learning_curves(history)

evaluate_train = model.evaluate(X_train_Aug, Y_train_Aug, batch_size=batch_size)
evaluate_validation = model.evaluate(X_validation,Y_validation, batch_size=batch_size)

y_pred = model.predict(X_test)
print(y_pred)
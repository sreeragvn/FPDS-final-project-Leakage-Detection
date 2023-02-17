import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from keras import layers

print("# GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
tf.keras.backend.set_floatx('float64')
tf.compat.v1.enable_eager_execution()

from sklearn.metrics import mean_squared_error


def data_plot(data):
    fig, axs = plt.subplots(5,1)
    axs = axs.flatten()
    for i in range(len(axs)-1):
        ax = axs[i]
        ax.plot(data[:,i])

    ax = axs[-1]
    ax.plot(tf.reduce_sum(data, axis=1))

    print("Mean of inputs: {}".format(tf.reduce_mean(data, axis=0)))
# np.random.seed(101)
# tf.random.set_seed(101)

# def data_exploration():
#     leakage_train_100 = pd.read_csv("leakage_dataset_train_100.csv")
#     leakage_train_1000 = pd.read_csv("leakage_dataset_train_1000.csv")
#     leakage_val_1000 = pd.read_csv("leakage_dataset_validation_1000.csv")
#     print(leakage_train_100.columns)
#     print(leakage_train_1000.columns)
#     print(leakage_val_1000.columns)

#     def data_description(datset):
#         print(datset['y1'].describe())
#         print(datset['y2'].describe())
#         print(datset['mfc1'].describe())
#         print(datset['mfc2'].describe())
#         print(datset['mfc3'].describe())
#         print(datset['mfc4'].describe())

#     data_description(leakage_train_100)
#     data_description(leakage_train_1000)
#     data_description(leakage_val_1000)

#     sns.set()
#     cols = ['y1', 'y2', 'mfc1', 'mfc2', 'mfc3', 'mfc4']
#     sns.pairplot(leakage_train_1000[cols], size = 2.5)
#     plt.show()

#     # missing data
#     def missingdata(df_train):
#         total = df_train.isnull().sum().sort_values(ascending=False)
#         percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
#         missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
#         print(missing_data.head(20))

#     missingdata(leakage_val_1000)

#     # histogram and normal probability plot
#     from scipy.stats import norm
#     from scipy import stats

#     sns.distplot(leakage_val_1000['y1'], fit=norm)
#     fig = plt.figure()
#     res = stats.probplot(leakage_val_1000['y1'], plot=plt)
#     return 0

def load_data(a):

    leakage_train_100 = pd.read_csv("leakage_dataset_train_100.csv")
    leakage_train_1000 = pd.read_csv("leakage_dataset_train_1000.csv")
    leakage_val_1000 = pd.read_csv("leakage_dataset_validation_1000.csv")

    if a == 100:
        print("100 rowed data loaded")
        train_ds = leakage_train_100
    elif a == 1000:
        print("1000 rowed data loaded")
        train_ds = leakage_train_1000
    val_ds = leakage_val_1000

    train_ds = train_ds.sample(frac=1)
    val_ds = val_ds.sample(frac=1)

    X_train = train_ds.iloc[:,2:].to_numpy()
    Y_train = train_ds.iloc[:,:2]

    X_validation = val_ds.iloc[:,2:].to_numpy()
    Y_validation = val_ds.iloc[:,:2]

    Y_train = Y_train.to_numpy()
    Y_validation = Y_validation.to_numpy()

    X_test = np.array([[0.25, 0.25, 0.25,0.25], 
                    [0.35, 0.25, 0.15,0.25], 
                    [0.25, 0.15, 0.25,0.35],
                    [0.15, 0.25, 0.35,0.25],
                    [0.25, 0.35, 0.25,0.15],
                    [0.4, 0.25, 0.1,0.25],
                    [0.1, 0.25, 0.4,0.25],
                    [0.42, 0.25, 0.08,0.25],
                    [0.1, 0.25, 0.4,0.25],
                    [0.25, 0.1, 0.25,0.4]])

    X_train = tf.convert_to_tensor(X_train)
    X_validation = tf.convert_to_tensor(X_validation)
    Y_train = tf.convert_to_tensor(Y_train)
    Y_validation = tf.convert_to_tensor(Y_validation)
    X_test = tf.convert_to_tensor(X_test)

    return X_train, Y_train, X_validation, Y_validation, X_test

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

def model_eval(model, history, X_validation, Y_validation, X_train, Y_train, batch_size, X_test):
    model.summary()
    predictions = model.predict(X_validation)
    prediction_accuracy(predictions, Y_validation)
    learning_curves(history)
    evaluate_train = model.evaluate(X_train, Y_train, batch_size=batch_size)

    # evaluate_train = model.evaluate(X_train, Y_train, batch_size=batch_size)
    evaluate_validation = model.evaluate(X_validation,Y_validation, batch_size=batch_size)

    y_pred = model.predict(X_test)
    print(y_pred)



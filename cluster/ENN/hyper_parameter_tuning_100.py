# things to do/doubts/discussions
# model sometimes give constant loss over epochs and gives poor results on prediction
# is the loss value supposed to reduce continously
# should we do augmentation of validation data ?
# All values are already between -1 to +1. should we do scaling on top of this ?
# if we are adding scaling of data, how to ensure that when it is being tested would give out rescaled data


# how to save subclassing api
# hyperparameter tuning
# how to save the best model among all the epochs
# should we shuffle repeat, prefetch etc ?
# incase of early stopping - what metric is to be monitored - val loss or val mse ?

#observations
#load model works without regularizer and initializer being passed  as arguments during forward pass

from utility import *

#load the data
data_size = 100
X_train, Y_train, X_validation, Y_validation, X_test = load_data(data_size)

# generating augmented data
X_train_Aug, Y_train_Aug = data_augmentation(X_train, Y_train)

class Hidden_layer(layers.Layer):
    def __init__(self, units, **kwargs):
        super(Hidden_layer, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.a = self.add_weight(name = 'a',shape=(1,), initializer=tf.keras.initializers.HeUniform(seed = 22),
                                 trainable=True, regularizer=tf.keras.regularizers.L1L2(0.01))
        self.b = self.add_weight(name = 'b',shape=(1,), initializer=tf.keras.initializers.HeUniform(seed = 22),
                                 trainable=True, regularizer=tf.keras.regularizers.L1L2(0.01))
        self.c = self.add_weight(name = 'c',shape=(1,), initializer=tf.keras.initializers.HeUniform(seed = 22),
                                 trainable=True, regularizer=tf.keras.regularizers.L1L2(0.01))
        self.a_matrix = tf.constant([[1,0,0,0], [0,1,0,0], [0,0,1,0],[0,0,0,1]], dtype=tf.float64)
        self.b_matrix = tf.constant([[0,1,0,1], [1,0,1,0], [0,1,0,1],[1,0,1,0]], dtype=tf.float64)
        self.c_matrix = tf.constant([[0,0,1,0], [0,0,0,1], [1,0,0,0],[0,1,0,0]], dtype=tf.float64)
    
    def call(self, inputs):
        self.W = tf.multiply(self.a, self.a_matrix) + tf.multiply(self.b, self.b_matrix) + tf.multiply(self.c, self.c_matrix)
        x = tf.keras.activations.relu(tf.matmul(inputs, self.W))
        # tf.print(self.W)
        return x
    def get_config(self):
        config = super(Hidden_layer, self).get_config()
        config.update({"units": self.units})
        # config.update({"initializer": initializer})
        # config.update({"kernel_regularizer": kernel_regularizer})
        return config
        # return {"units": self.units, "kernel_regularizer": kernel_regularizer, "initializer": initializer}
    # @classmethod
    # def from_config(cls, config):
    #     return cls(**config)
        
class Output_layer(layers.Layer):
    def __init__(self,units, **kwargs):
        super(Output_layer, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.d = self.add_weight(name = 'd',shape=(1,), initializer=tf.keras.initializers.HeUniform(seed = 22),
                                 trainable=True, regularizer=tf.keras.regularizers.L1L2(0.01))
        self.d_matrix = tf.constant([[1,-1], [-1,-1], [-1,1],[1,1]], dtype=tf.float64)

    def call(self, inputs):
        self.W = tf.multiply(self.d, self.d_matrix)
        x = tf.matmul(inputs, self.W)
        # tf.print(tf.transpose(self.W))
        return tf.keras.activations.tanh(x)
    def get_config(self):
        config = super(Output_layer, self).get_config()
        config.update({"units": self.units})
        # config.update({"initializer": initializer})
        # config.update({"kernel_regularizer": kernel_regularizer})
        return config
        # return {"units": self.units, "kernel_regularizer": kernel_regularizer, "initializer": initializer}
    # @classmethod
    # def from_config(cls, config):
    #     return cls(**config)

# class MyReLU(layers.Layer):
#     def __init__(self):
#         super(MyReLU, self).__init__()

#     def call(self, x):
#         return tf.math.maximum(x, 0)


results = pd.DataFrame(columns=['learning_rate','loss','optimizer_name', 'width', 'depth', 'batch_size', 'train_loss', 'val_loss', 'train_mse', 'val_mse'])
# defining the parameters
epochs = 1000
verbose=2
learning_rates = [1e-2,1e-3,1e-4]
optimizers = [keras.optimizers.Adam,keras.optimizers.SGD]
losses = [tf.keras.losses.MeanSquaredError,tf.keras.losses.MeanAbsoluteError]
batch_sizes = [32, 64]
widths = [4, 32, 64]
depths = [0, 1, 2, 3]

# learning_rates = [1e-2]
# optimizers = [keras.optimizers.SGD]
# losses = [tf.keras.losses.MeanAbsoluteError]
# batch_sizes = [32]
# widths = [4]
# depths = [1]

# make cross product

for optimizer in optimizers:
    optimizer_name = optimizer().get_config()['name']
    for loss in losses:
        loss_name = loss().get_config()['name']
        for learning_rate in learning_rates:
            for width in widths:
                for depth in depths:
                    for batch_size in batch_sizes:
                        for _ in range(3):
                            model = keras.models.Sequential()
                            for _ in range(depth):
                                model.add(Hidden_layer(width))
                            model.add(Output_layer(2))
                            model.compile(optimizer=optimizer(learning_rate),
											loss = loss(),
											metrics = tf.keras.metrics.MeanSquaredError()
										)
                            history = model.fit(X_train, Y_train, 
												epochs=epochs, 
												batch_size= batch_size, 
												verbose=verbose,
												validation_data=(X_validation, Y_validation),
												# callbacks=callbacks,
												# shuffle=True
												)
                            train_loss, train_mse = model.evaluate(X_train, Y_train, batch_size=batch_size)
                            val_loss, val_mse = model.evaluate(X_validation,Y_validation, batch_size=batch_size)
                            results_tmp = np.array([learning_rate,loss_name,optimizer_name, width, depth, batch_size, train_loss, val_loss, train_mse, val_mse]).reshape(1, -1)
                            print("results = ", results_tmp)
							# either use tensor board or save the training curve in each loop. fuction for plotting training curve is available in utility
                            results = results.append(pd.DataFrame(data=results_tmp, columns=results.columns), ignore_index=True)
results.to_csv('results_ENN_100.csv')
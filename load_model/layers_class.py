from utility import *
seed = 22

class Hidden_layer(layers.Layer):
    def __init__(self,units, **kwargs):
        super(Hidden_layer, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.W = self.add_weight(name = 'w',shape=(input_shape[-1],self.units), initializer=tf.keras.initializers.HeUniform(seed = seed),
                                 trainable=True, regularizer=tf.keras.regularizers.L1L2(0.01))

    def call(self, inputs):
        x = tf.keras.activations.relu(tf.matmul(inputs, self.W))
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
    def __init__(self, units, **kwargs):
        super(Output_layer, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.W = self.add_weight(name = 'w',shape=(input_shape[-1],self.units), initializer=tf.keras.initializers.HeUniform(seed = seed),
                                 trainable=True, regularizer=tf.keras.regularizers.L1L2(0.01))

    def call(self, inputs):
        x = tf.matmul(inputs, self.W)
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

class EqHidden_layer(layers.Layer):
    def __init__(self, units, **kwargs):
        super(EqHidden_layer, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.a = self.add_weight(name = 'a',shape=(1,), initializer=tf.keras.initializers.HeUniform(seed = seed),
                                 trainable=True, regularizer=tf.keras.regularizers.L1L2(0.01))
        self.b = self.add_weight(name = 'b',shape=(1,), initializer=tf.keras.initializers.HeUniform(seed = seed),
                                 trainable=True, regularizer=tf.keras.regularizers.L1L2(0.01))
        self.c = self.add_weight(name = 'c',shape=(1,), initializer=tf.keras.initializers.HeUniform(seed = seed),
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
        config = super(EqHidden_layer, self).get_config()
        config.update({"units": self.units})
        # config.update({"initializer": initializer})
        # config.update({"kernel_regularizer": kernel_regularizer})
        return config
    # @classmethod
    # def from_config(cls, config):
    #     return cls(**config)

class EqOutput_layer(layers.Layer):
    def __init__(self,units, **kwargs):
        super(EqOutput_layer, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.d = self.add_weight(name = 'd',shape=(1,), initializer=tf.keras.initializers.HeUniform(seed = seed),
                                 trainable=True, regularizer=tf.keras.regularizers.L1L2(0.01))
        self.d_matrix = tf.constant([[1,-1], [-1,-1], [-1,1],[1,1]], dtype=tf.float64)

    def call(self, inputs):
        self.W = tf.multiply(self.d, self.d_matrix)
        x = tf.matmul(inputs, self.W)
        # tf.print(tf.transpose(self.W))
        return tf.keras.activations.tanh(x)
    def get_config(self):
        config = super(EqOutput_layer, self).get_config()
        config.update({"units": self.units})
        # config.update({"initializer": initializer})
        # config.update({"kernel_regularizer": kernel_regularizer})
        return config

    # @classmethod
    # def from_config(cls, config):
    #     return cls(**config)
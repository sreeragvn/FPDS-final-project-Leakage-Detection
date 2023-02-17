from utility import *
#load the data
data_size = 100
X_train, Y_train, X_validation, Y_validation, X_test = load_data(data_size)

# defining the parameters
# loss
# Mean Squared Error
# Root Mean Squared Error
# Mean Absolute Error
batch_size = 32
epochs = 100
# steps_per_epoch = sum(train_occurences) / batch_size
starter_learning_rate = 1e-1
end_learning_rate = 1e-8
decay_steps = epochs * 3
loss = tf.keras.losses.MeanAbsoluteError()
metrics = tf.keras.metrics.MeanSquaredError()
scheduler = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate= starter_learning_rate,
    decay_steps= decay_steps,
    end_learning_rate= end_learning_rate,
    power=1)
scheduler = 0.01
optimizer=tf.keras.optimizers.Adam(learning_rate=scheduler)
kernel_regularizer=tf.keras.regularizers.L1L2(0.01)
callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_mean_squared_error',patience=100)]
initializer=tf.keras.initializers.HeUniform()

verbose=2

# generating augmented data
X_train_Aug, Y_train_Aug = data_augmentation(X_train, Y_train)

# NN Model
class NNmodel():
    def model_structure():
        model = tf.keras.models.Sequential()
        model.add(Hidden_layer(4, kernel_regularizer=kernel_regularizer, initializer=initializer))
        model.add(Output_layer(2, kernel_regularizer=kernel_regularizer, initializer=initializer))
        model.compile(optimizer=optimizer,
                    loss = loss,
                    metrics = metrics,
        )
        return model

# Model 1 - NN fitting with Normal data
model = NNmodel.model_structure()
history = model.fit(X_train, Y_train, 
                    epochs=epochs, 
                    batch_size= batch_size, 
                    verbose=verbose, 
                    validation_data=(X_validation, Y_validation),
                    # callbacks=callbacks,
                    # shuffle=True
                    )  
model_eval(model, history, X_validation, Y_validation, X_train, Y_train, batch_size, X_test)

batch_size = 32
epochs = 1000
# steps_per_epoch = sum(train_occurences) / batch_size
starter_learning_rate = 1e-1
decay_steps = epochs * 3
scheduler = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate= starter_learning_rate,
    decay_steps= decay_steps,
    end_learning_rate= end_learning_rate,
    power=1)
kernel_regularizer=tf.keras.regularizers.L1L2(0.01)
callbacks=[keras.callbacks.EarlyStopping(monitor='val_mean_squared_error',patience=20)]
initializer=tf.keras.initializers.HeUniform

results = pd.DataFrame(columns=['start_learning_rate', 'width', 'depth', 'l2_weight', 'train_loss', 'val_loss', 'train_acc', 'val_acc'])

starter_learning_rate = [1e-1]
batch_sizes = [32, 64]
widths = [4]
depths = [1, 2, 4]
# regularizer_strength = [0.01]
# kernel_regularizer=[tf.keras.regularizers.L1L2(0.01)]
l2_weights = [0.01, 0.001, 1e-4]
# make cross product

for starter_learning_rate in starter_learning_rate:
    for width in widths:
        for depth in depths:
            for l2_weight in l2_weights:
                for batch_size in batch_sizes:
                    model = keras.models.Sequential()
                    model.add(Hidden_layer(4, kernel_regularizer=keras.regularizers.l2(l2_weight)))
                    for _ in range(depth):
                        model.add(Hidden_layer(4, kernel_regularizer=keras.regularizers.l2(l2_weight)))
                    model.add(Output_layer(2, kernel_regularizer=keras.regularizers.l2(l2_weight)))
                    scheduler = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate= starter_learning_rate,
                                decay_steps= decay_steps,
                                end_learning_rate= end_learning_rate,
                                power=1)
                    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=scheduler),
                                    loss='mean_squared_error',
                                    metrics = [tf.keras.metrics.MeanSquaredError()]
                                    )
                    history = model.fit(X_train, Y_train, epochs=epochs, batch_size= batch_size, verbose=2, validation_data=(X_validation, Y_validation),
                                        callbacks=callbacks,
                                        # shuffle=True
                                        )
                    train_loss, train_acc = model.evaluate(X_train, Y_train, batch_size=batch_size)
                    val_loss, val_acc = model.evaluate(X_validation,Y_validation, batch_size=batch_size)
                    results_tmp = np.array([starter_learning_rate, width, depth, l2_weight, train_loss, val_loss, train_acc, val_acc]).reshape(1, -1)
                    results = results.append(pd.DataFrame(data=results_tmp, columns=results.columns), ignore_index=True)
results.to_csv('results.csv')
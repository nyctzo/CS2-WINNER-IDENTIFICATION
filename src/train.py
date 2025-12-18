from tensorflow import keras
model = keras.models.Sequential()
model.add(keras.layers.Input(shape = (20,)))
model.add(keras.layers.Dense(200, activation = "relu" ))
model.add(keras.layers.Dense(100, activation = "relu" ))
model.add(keras.layers.Dense(100, activation = "relu" ))
model.add(keras.layers.Dense(1, activation = "sigmoid"))

model.compile( loss = "binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])

early_stopping_cb = keras.callbacks.EarlyStopping(patience = 5)
X_train_scaled_train , x_valid , y_train_train , y_valid = train_test_split(X_trained_scaled, y_train, test_size = 0.15)
model.fit(X_train_scaled_train , y_train_train, epochs = 10, callbacks =[early_stopping_cb], validation_data = (x_valid, y_valid))

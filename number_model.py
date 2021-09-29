#Imports

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

num_classes = 10 #How many possible outputs there are
input_shape = (28, 28, 1) #shape of the input image

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data() #loads the MNIST dataset from keras

#normalizes the values to be 0-1 rather than 0-255
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#creates the model to be trained with several layers.
model = keras.Sequential([
    keras.Input(shape=input_shape),
    layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
    layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation="softmax"),
])

#initializes batch size and how many epochs to train over
batch_size = 128
epochs = 15

#compiles the model
model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

#trains the model
model.fit(x_train,
          y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=.1)

#creates metric to output what final loss and accuracy values are
(loss, accuracy) = model.evaluate(x_test, y_test, verbose=0)
print("Loss: ", loss)
print("Accuracy: ", accuracy)

#saves the model to disk
model.save("C:/Users/andre/Desktop/Test/MNIST/model2")

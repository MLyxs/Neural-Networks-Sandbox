import numpy as np

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.core import Activation
from keras.utils import np_utils

batch_size = 128
nb_classes = 10
nb_epochs = 1

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_train /= 255
X_test /= 255

print(X_train.shape[0], "train samples")
print(X_test.shape[0], "test samples")

y_train = np_utils.to_categorical(y_train, nb_classes)
y_test =  np_utils.to_categorical(y_test, nb_classes)

print(y_train[0]) 
print(y_train[1])
print(y_train[2])
print(y_train[3])

model = Sequential()

model.add(Dense(output_dim = 200, input_dim = 784))
model.add(Activation("sigmoid"))
model.add(Dense(output_dim=nb_classes))
model.add(Activation("softmax"))


model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics=['accuracy'])

model.fit(X_train, y_train, batch_size = batch_size, nb_epoch = nb_epochs, verbose = 2, validation_data = (X_test, y_test))
model.save('trained_model_for_jscript.h5')

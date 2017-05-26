
from keras.models import Sequential
from keras.layers import Flatten, Dense, Convolution2D, Dropout


def nvidia_model_with_dropout(input_shape):
    model = Sequential()
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation="elu", input_shape=input_shape))
    model.add(Dropout(0.7))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="elu"))
    model.add(Dropout(0.7))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="elu"))
    model.add(Dropout(0.7))
    model.add(Convolution2D(64, 3, 3, activation="elu"))
    model.add(Dropout(0.7))
    model.add(Convolution2D(64, 3, 3, activation="elu"))
    model.add(Flatten())
    model.add(Dense(100, activation="elu"))
    model.add(Dropout(0.7))
    model.add(Dense(50, activation="elu"))
    model.add(Dense(10, activation="elu"))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')
    return model


def simplified_nvidia_model_with_dropout(input_shape):
    model = Sequential()
    model.add(Convolution2D(24, 5, 5, subsample=(3, 3), activation="elu", input_shape=input_shape))
    model.add(Dropout(0.7))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="elu"))
    model.add(Dropout(0.7))
    model.add(Convolution2D(48, 3, 3, subsample=(2, 2), activation="elu"))
    model.add(Dropout(0.7))
    model.add(Convolution2D(64, 3, 3, subsample=(2, 2), activation="elu"))
    model.add(Flatten())
    model.add(Dense(100, activation="elu"))
    model.add(Dense(50, activation="elu"))
    model.add(Dense(10, activation="elu"))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')
    return model

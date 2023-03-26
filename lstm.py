import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras as keras
DATASET_PATH = "data.json"


def load_data(dataset_path):
    """
    :param dataset_path: a string representing the file path to the dataset
    :return: a tuple containing the inputs and targets as numpy arrays
    """
    with open(dataset_path, "r") as fp:
        data = json.load(fp)

    # convert a list into numpy arrays
    inputs = np.array(data["mfcc"])
    targets = np.array(data["labels"])
    return inputs, targets


def prepare_datasets(test_size, validation_size):
    """
    :param test_size: a float representing the proportion of the dataset to be used as test set
    :param validation_size: a float representing the proportion of the dataset to be used as validation set
    :return: a tuple containing the train, validation, and test sets as numpy arrays, and their corresponding targets
    """

    # load data
    X, y = load_data(DATASET_PATH)

    # create train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    # create train/validation split
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)

    # 3d array -> (130,13,1)
    # X_train = X_train[..., np.newaxis] #4d array -> (num_samples, 130, 13, 1)
    # X_validation = X_validation[..., np.newaxis]
    # X_test = X_test[..., np.newaxis]

    return X_train, X_validation, X_test, y_train, y_validation, y_test


def build_model(input_shape):
    """
    :param input_shape: a tuple representing the shape of the input data
    :return: a sequential model
    """

    # create RNN-LSTM model
    model = tf.keras.Sequential()

    # 2 LSTM layers
    model.add(tf.keras.layers.LSTM(64, input_shape=input_shape, return_sequences=True))
    model.add(tf.keras.layers.LSTM(64))

    # Dense layer
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.3))

    # output layer
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    return model

def predict(model, X, y):
    """
    :param model: a Keras model used for making the prediction
    :param X: a numpy array representing the input data
    :param y: an integer representing the expected index
    :return: None
    """
    X = X[np.newaxis, ...]

    # prediction = [ [0.1, 0.2, ...] ]
    prediction = model.predict(X) # X -> (1, 130, 13, 1)

    # extract index with max value
    predicted_index = np.argmax(prediction, axis=1) # [4]
    print(f'Expected index: {y}, Predicted index: {predicted_index}')


if __name__ == "__main__":
    # create train, validation and test sets
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(0.25, 0.2)

    # build the cnn net
    input_shape = (X_train.shape[1], X_train.shape[2]) # 130, 13
    model = build_model(input_shape)

    # compile the network
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    # train the cnn
    model.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size=32, epochs=30)

    # evaluate the CNN on the test set
    test_error, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    print(f'Accuracy on test set is: {test_accuracy}')


    # make prediction on sample
    X = X_test[102]
    y = y_test[102]

    predict(model, X, y)

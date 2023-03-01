import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import The_Free_Spoken_Digit_dataset as ds

# load TIDIGITS dataset, extract features, and preprocess data

x_data, y_data = ds.load_data("free-spoken-digit-dataset/recordings")
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)
x_train = np.array(x_train)
x_test = np.array(x_test)

# handle division by zero warning
me, std = np.mean(x_train), np.std(x_train)
std = std if std != 0 else 1e-6
x_train = (x_train - mean) / std
x_test = (x_test - mean) / std

# convert y_train and y_test to numpy arrays and set their data types to integers
y_train = np.array(y_train).astype(int)
y_test = np.array(y_test).astype(int)

# define the model
num_inputs = len(x_train[0])
num_hidden = [100, 200]
num_outputs = len(set(y_train))

model = tf.keras.Sequential([
    tf.keras.layers.Dense(num_hidden[0], activation='sigmoid', input_dim=num_inputs),
    tf.keras.layers.Dense(num_hidden[1], activation='sigmoid'),
    tf.keras.layers.Dense(num_outputs, activation='softmax')
])

# compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# train the model
epochs = 50
history = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test))

# test the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

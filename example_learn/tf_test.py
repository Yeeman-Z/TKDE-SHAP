import os
import pickle as pk
import numpy as np
# gpus= tf.config.list_physical_devices('GPU') 
# tf.config.experimental.set_memory_growth(gpus[0], True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)



data_path = "../new_fed_shapley_scheme/datasets/emnist/client_5_same/"
file_client_trainX = open(data_path+'client_trainX.pk', 'rb')
file_client_trainY = open(data_path+'client_trainY.pk', 'rb')
client_trainX = pk.load(file_client_trainX)
client_trainY = pk.load(file_client_trainY)

data_path = "../new_fed_shapley_scheme/datasets/emnist/client_5_same/"
file_testX = open(data_path+'testX.pk', 'rb')
file_testY = open(data_path+'testY.pk', 'rb')
x_test = pk.load(file_testX)
y_test = pk.load(file_testY)
# client_trainX = axis=0
x_train = np.concatenate(client_trainX, axis=0)
y_train = np.concatenate(client_trainY, axis=0)
# mnist = tf.keras.datasets.mnist

# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# x_train, x_test = x_train / 255.0, x_test / 255.0

# model = tf.keras.models.Sequential([
#   tf.keras.layers.Flatten(input_shape=(28, 28)),
#   # tf.keras.Input(shape=(28, 28)),
#   # tf.keras.layers.Conv2D(2, 5, strides=2, activation="relu"),
#   tf.keras.layers.Dense(128, activation='relu'),
#   tf.keras.layers.Dense(128, activation='relu'),
#   tf.keras.layers.Dense(128, activation='relu'),
#   tf.keras.layers.Dropout(0.2),
#   tf.keras.layers.Dense(10)
# ])
# x_train, y_train = tf.reshape(x_train, [-1,28,28,1]), tf.reshape(y_train, [-1,1])
# x_test, y_test =  tf.reshape(x_test, [-1,28,28,1]),   tf.reshape(y_test, [-1,1])

print(x_train.shape, y_train.shape)

# model = tf.keras.models.Sequential([
#         tf.keras.layers.Reshape((28,28,1), input_shape=(28, 28)),
#         # tf.keras.Input(shape=(28,28,1), name='Input'),
#         # tf.reshape([28,28,1]),
#         tf.keras.layers.Conv2D(32, (5,5), activation='relu', padding='same', name='Conv2D-3x3'),  #(28, 28, 32)
#         tf.keras.layers.MaxPooling2D((2,2), name='Pool2D-2x2'),   # (14,14,32)
#         tf.keras.layers.Conv2D(64, (2,2),padding='same', activation='relu'), #(14,14,64)
#         tf.keras.layers.MaxPooling2D((2,2)), #[7, 7, 64]
#         # tf.reshape([7*7*64]), 
#         tf.keras.layers.Conv2D(64, (3, 3), activation='relu'), #(5,5,64)
#         tf.keras.layers.Flatten(), #5*5*64
#         # tf.keras.layers.Dense(5*5*64, activation='relu'),
#         tf.keras.layers.Dense(64, activation='relu'),
#         # tf.keras.layers.Dropout(rate=0.2),
#         tf.keras.layers.Dense(10,)
# ])

model = tf.keras.models.Sequential([
            tf.keras.layers.Reshape((28,28,1), input_shape=(28, 28)),
            tf.keras.layers.Conv2D(16, (3,3), activation='relu', padding='same', name='Conv2D-3x3'),  #(28, 28, 32)
            tf.keras.layers.MaxPooling2D((2,2), name='Pool2D-2x2'),   # (14,14,32)
            tf.keras.layers.Conv2D(32, (2,2),padding='same', activation='relu'), #(14,14,64)
            tf.keras.layers.MaxPooling2D((2,2)), #[7, 7, 64]
            # tf.keras.layers.Conv2D(64, (3, 3), activation='relu'), #(5,5,64)
            tf.keras.layers.Flatten(), #5*5*64
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(10,)
        ])


model.summary()

# predictions = model(x_train[:1]).numpy()

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=32000, epochs=20)
model.evaluate(x_test,  y_test, verbose=2)

import tensorflow as tf
import numpy as np


CLINT_NUM = 5
BASIC_PORT = 50020
STOP_PORT =  50081
FED_ROUND = 10
DATA_SHAPE = ((28,28), 10)
LOCAL_EPOCH = 5
LOCAL_BATCH = 120


def nparray_to_rpcio(nparray):
    byte_array_data = [x.tobytes() for x in nparray]
    byte_array_type = [str(x.dtype) for x in nparray]
    byte_array_shape = [str(x.shape) for x in nparray]
    return byte_array_data, byte_array_type, byte_array_shape 

def rpcio_to_nparray(byte_data, byte_type, byte_shape):
    # request.
    return [np.frombuffer(data, dtype=np.dtype(rtype)).reshape(eval(shape)) 
            for data,rtype,shape in zip(byte_data, byte_type, byte_shape)]


class linear_model():

    def __init__(self, _input, _output):

        print("Now we are creating Linear Model with input={}, output={}".format(_input,_output))
        self.input  = _input
        # self.output = _output
        self.model = tf.keras.models.Sequential([
            # tf.keras.Input(shape=_input),
            tf.keras.layers.Flatten(input_shape=_input),
            # tf.keras.layers.Dense(128, activation='relu'),
            # tf.keras.layers.Dense(128, activation='relu'),
            # tf.keras.layers.Dense(_output),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10)
        ])
        # self.model.summary()
        self.model_compile()

    
    def model_compile(self):
        self.model.compile(optimizer="adam", 
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])
    
    def model_fit(self, _datasets, _local_epoches, _batchsize):
        self.model.fit(_datasets[0], _datasets[1], batch_size=_batchsize, epochs=_local_epoches)
    

    def model_load_weights(self, _weights):
        self.model.set_weights(_weights)


    def model_get_weights(self):
        return self.model.get_weights()
    
    def model_get_eval(self, _test_data, _test_label, verbose=2):
        return self.model.evaluate(_test_data, _test_label, verbose=2)


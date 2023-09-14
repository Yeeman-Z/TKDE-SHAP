import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np


CLINT_NUM = 5
BASIC_PORT = 50020
STOP_PORT =  50081
FED_ROUND = 100
DATA_SHAPE = ((28,28), 10)
LOCAL_EPOCH = 5
LOCAL_BATCH = 64

LOCAL_MINI_FLAG = False
LOCAL_MINI_BATCH_SIZE = 4000
LOCAL_MINI_EPOCH = 1


def nparray_to_rpcio(nparray):
    byte_array_data = [x.tobytes() for x in nparray]
    byte_array_type = [str(x.dtype) for x in nparray]
    byte_array_shape = [str(x.shape) for x in nparray]
    return byte_array_data, byte_array_type, byte_array_shape 

def rpcio_to_nparray(byte_data, byte_type, byte_shape):
    # request.
    return [np.frombuffer(data, dtype=np.dtype(rtype)).reshape(eval(shape)) 
            for data,rtype,shape in zip(byte_data, byte_type, byte_shape)]


class basic_model():

    def __init__(self, _input, _output, _type):

        self.local_mini_flag = LOCAL_MINI_FLAG
        self.local_mini_batch_size = LOCAL_MINI_BATCH_SIZE
        self.local_mini_epoch = LOCAL_MINI_EPOCH
        print("Now we are creating {} with input={}, output={}".format(_type, _input,_output))
        print("Local_Mini_Batch is {}, with Mini_Size is {}".format(self.local_mini_flag, self.local_mini_batch_size))
        # pass
        # self.input  = _input
        # # self.output = _output


    
    def model_compile(self):
        self.model.compile(optimizer="adam", 
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])
    
    def model_fit(self, _datasets, _local_epoches, _batchsize):
        if not self.local_mini_flag:
            self.model.fit(_datasets[0], _datasets[1], batch_size=_batchsize, epochs=_local_epoches)
        else:
            ids = np.random.choice(np.array(range(len(_datasets[1]))),size=self.local_mini_batch_size, replace=False)
            mini_datasets = (_datasets[0][ids], _datasets[1][ids])
            # print("ids:", ids, '\n', "labels:", mini_datasets[1])
            self.model.fit(mini_datasets[0], mini_datasets[1], batch_size=self.local_mini_batch_size, epochs=self.local_mini_epoch)


    

    def model_load_weights(self, _weights):
        self.model.set_weights(_weights)


    def model_get_weights(self):
        return self.model.get_weights()
    
    def model_get_eval(self, _test_data, _test_label, verbose=2):
        return self.model.evaluate(_test_data, _test_label, verbose=2)


class linear_model(basic_model):
    
    def  __init__(self, _input, _output):
        self.input = _input
        self.output = _output
        self.model_type = "Linear Model"
        super(linear_model, self).__init__(self.input, self.output, self.model_type)

        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=_input),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10)
        ])
        # self.model.summary()
        self.model_compile() 


class cnn_model(basic_model):
    
    def  __init__(self, _input, _output):
        self.input = _input
        self.output = _output
        self.model_type = "CNN Model"
        super(cnn_model, self).__init__(self.input, self.output, self.model_type)

        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Reshape((28,28,1), input_shape=(28, 28)),
            tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same', name='Conv2D-3x3'),  
            tf.keras.layers.MaxPooling2D((2,2), name='Pool2D-2x2'),  
            tf.keras.layers.Conv2D(64, (2,2),padding='same', activation='relu'), 
            tf.keras.layers.MaxPooling2D((2,2)), 
            tf.keras.layers.Flatten(), 
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(10,)
        ])
        # self.model = tf.keras.models.Sequential([
        #     tf.keras.layers.Reshape((28,28,1), input_shape=(28, 28)),
        #     tf.keras.layers.Conv2D(32, (8,8), activation='relu', padding='same', name='Conv2D-3x3'),  #(28, 28, 32)
        #     tf.keras.layers.MaxPooling2D((8,8), name='Pool2D-2x2'),   # (14,14,32)
        #     # tf.keras.layers.Conv2D(64, (2,2),padding='same', activation='relu'), #(14,14,64)
        #     # tf.keras.layers.MaxPooling2D((2,2)), #[7, 7, 64]
        #     tf.keras.layers.Flatten(), #5*5*64
        #     tf.keras.layers.Dense(16, activation='relu'),
        #     tf.keras.layers.Dense(10,)
        # ])

        # self.model.summary()
        self.model_compile() 




FED_MODEL = cnn_model

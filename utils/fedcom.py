from myconst.dataset_const import *
import collections
import random
import numpy as np
import os
import nest_asyncio
nest_asyncio.apply()
import tensorflow as tf
import tensorflow_federated as tff
tff.backends.reference.set_reference_context()

CLIENT_NUM = MNIST_CLIENT_NUM
DATSETSHAPE = MNISTSHAPE
DATSETLABEL = MNISTLABEL
if DATASET == "AGE":
    CLIENT_NUM = AGE_CLIENT_NUM
    DATSETSHAPE = AGESHAPE
    DATSETLABEL = AGELABEL


random.seed(42)
np.random.seed(42) 

############===============TensorFlow-Federated-BEGIN==========###################
BATCH_SPEC = collections.OrderedDict(
    x=tf.TensorSpec(shape=[None, DATSETSHAPE], dtype=tf.float32),
    y=tf.TensorSpec(shape=[None,], dtype=tf.int32)
)
BATCH_TYPE = tff.to_type(BATCH_SPEC)

MODEL_SPEC = collections.OrderedDict(
    weights=tf.TensorSpec(shape=[DATSETSHAPE, DATSETLABEL], dtype=tf.float32),
    bias=tf.TensorSpec(shape=[DATSETLABEL], dtype=tf.float32)
)
MODEL_TYPE = tff.to_type(MODEL_SPEC)

LOCAL_DATA_TYPE = tff.SequenceType(BATCH_TYPE)
SERVER_FLOAT_TYPE = tff.type_at_server(tf.float32)
SERVER_MODEL_TYPE = tff.type_at_server(MODEL_TYPE)
CLIENT_DATA_TYPE = tff.type_at_clients(LOCAL_DATA_TYPE)

# forward_pass: the loss function of TensorFlow
@tf.function
def forward_pass(model, batch):
    predicted_y = tf.nn.softmax(
        tf.matmul(batch['x'], model['weights']) + model['bias']
    )
    return -tf.reduce_mean(
        tf.reduce_sum(tf.one_hot(batch['y'], DATSETLABEL)*tf.math.log(predicted_y), axis=[1]))

# batch_loss: the loss function of TensorFlow-Federated
@tff.tf_computation(MODEL_TYPE, BATCH_TYPE)
def batch_loss(model, batch):
    return forward_pass(model, batch)


# batch_train: train the model based on
@tff.tf_computation(MODEL_TYPE, BATCH_TYPE, tf.float32)
def batch_train(initModel, batch, learning_rate):
    model_vars = collections.OrderedDict([
        (name, tf.Variable(name=name, initial_value=value))
        for name, value in initModel.items()
    ])

    optimizer = tf.keras.optimizers.SGD(learning_rate)

    @tf.function
    def _train_on_batch(model_vars, batch):
        with tf.GradientTape() as tape:
            loss = forward_pass(model_vars, batch)
        grads = tape.gradient(loss, model_vars)
        optimizer.apply_gradients(
            zip(tf.nest.flatten(grads), tf.nest.flatten(model_vars)))
        return model_vars

    return _train_on_batch(model_vars, batch)


# train the local model
@tff.federated_computation(MODEL_TYPE, tf.float32, LOCAL_DATA_TYPE)
def local_train(initial_model, learning_rate, all_batches):

  # Mapping function to apply to each batch.
    @tff.federated_computation(MODEL_TYPE, BATCH_TYPE)
    def batch_fn(model, batch):
        return batch_train(model, batch, learning_rate)
    return tff.sequence_reduce(all_batches, initial_model, batch_fn)


@tff.federated_computation(MODEL_TYPE, LOCAL_DATA_TYPE)
def local_eval(model, all_batches):
  return tff.sequence_sum(
      tff.sequence_map(
          tff.federated_computation(lambda b: batch_loss(model, b), BATCH_TYPE), all_batches))


@tff.federated_computation(SERVER_MODEL_TYPE, SERVER_FLOAT_TYPE,
                           CLIENT_DATA_TYPE)
def federated_train(model, learning_rate, data):
  return tff.federated_map(local_train,
        [tff.federated_broadcast(model),
          tff.federated_broadcast(learning_rate), data])

@tff.federated_computation(SERVER_MODEL_TYPE, CLIENT_DATA_TYPE)
def federated_eval(model, data):
  return tff.federated_mean(
      tff.federated_map(local_eval, [tff.federated_broadcast(model), data]))
############===============TensorFlow-Federated-END==========###################
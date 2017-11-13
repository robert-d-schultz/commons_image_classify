import numpy as np
import tensorflow as tf
import os

tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    input_layer = tf.reshape(features, [-1, 28, 28, 1])

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d( inputs = input_layer
                            , filters = 32
                            , kernel_size = [5, 5]
                            , padding = "same"
                            , activation = tf.nn.relu
                            )

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs = conv1, pool_size = [2, 2], strides = 2)

    # Convolutional Layer #2
    conv2 = tf.layers.conv2d( inputs = pool1
                            , filters = 64
                            , kernel_size = [5, 5]
                            , padding = "same"
                            , activation = tf.nn.relu
                            )

    # Pooling Layer #2
    pool2 = tf.layers.max_pooling2d(inputs = conv2, pool_size = [2, 2], strides = 2)

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs = pool2_flat, units = 1024, activation = tf.nn.relu)

    # Add dropout operation; 0.6 probability that element will be kept
    dropout = tf.layers.dropout( inputs = dense
                               , rate = 0.4
                               , training = mode == tf.estimator.ModeKeys.TRAIN
                               )

    # Logits layer
    logits = tf.layers.dense(inputs = dropout, units = 2)

    # Generate predictions (for PREDICT and EVAL mode)
    predictions = { "classes": tf.argmax(input = logits, axis = 1)
                  , "probabilities": tf.nn.softmax(logits, name = "softmax_tensor")
                  }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode = mode, predictions = predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    onehot_labels = tf.one_hot(indices = tf.cast(labels, tf.int32), depth = 2)
    loss = tf.losses.softmax_cross_entropy(onehot_labels = onehot_labels, logits = logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.001)
        train_op = optimizer.minimize(loss = loss, global_step = tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode = mode, loss = loss, train_op = train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = { "accuracy": tf.metrics.accuracy(labels = labels, predictions = predictions["classes"])
                      , "confusion": eval_confusion_matrix(labels = labels, predictions = predictions["classes"])
                      , "precision": tf.metrics.precision(labels = labels, predictions = predictions["classes"])
                      }
    return tf.estimator.EstimatorSpec( mode = mode
                                     , loss = loss
                                     , eval_metric_ops = eval_metric_ops
                                     )

def eval_confusion_matrix(labels, predictions):
    with tf.variable_scope("eval_confusion_matrix"):
        con_matrix = tf.confusion_matrix(labels = labels, predictions = predictions, num_classes = 2)
        con_matrix_sum = tf.Variable(tf.zeros(shape = (2,2), dtype = tf.int32)
                                    , trainable = False
                                    , name = "confusion_matrix_result"
                                    , collections = [tf.GraphKeys.LOCAL_VARIABLES]
                                    )
        update_op = tf.assign_add(con_matrix_sum, con_matrix)
        return tf.convert_to_tensor(con_matrix_sum), update_op

def my_input_fn():
    above_list = []
    for f in os.listdir('./above_data_train/'):
        if f.endswith(".png"):
            above_list.append(os.path.join('./above_data_train/', f))

    below_list = []
    for g in os.listdir('./below_data_train/'):
        if g.endswith(".png"):
            below_list.append(os.path.join('./below_data_train/', g))

    filename_list = above_list + below_list
    label_list = [1]*len(above_list) + [0]*len(below_list)

    filenames = tf.convert_to_tensor(filename_list, dtype = tf.string)
    labels = tf.convert_to_tensor(label_list, dtype = tf.int32)
    filenames_queue, labels_queue = tf.train.slice_input_producer([filenames, labels], shuffle = True)

    images_queue = tf.read_file(filenames_queue)
    images_queue = tf.image.decode_png(images_queue, channels = 1)
    images_queue = tf.image.resize_images(images_queue, [28, 28])

    return tf.train.batch([images_queue, labels_queue], batch_size = 100, num_threads = 4)

def my_eval_input_fn():
    above_list2 = []
    for f in os.listdir('./above_data_eval/'):
        if f.endswith(".png"):
            above_list2.append(os.path.join('./above_data_eval/', f))

    below_list2 = []
    for g in os.listdir('./below_data_eval/'):
        if g.endswith(".png"):
            below_list2.append(os.path.join('./below_data_eval/', g))

    filename_list2 = above_list2 + below_list2
    label_list2 = [1]*len(above_list2) + [0]*len(below_list2)

    filenames2 = tf.convert_to_tensor(filename_list2, dtype = tf.string)
    labels2 = tf.convert_to_tensor(label_list2, dtype = tf.int32)
    filenames_queue2, labels_queue2 = tf.train.slice_input_producer([filenames2, labels2], shuffle = True)

    images_queue2 = tf.read_file(filenames_queue2)
    images_queue2 = tf.image.decode_png(images_queue2, channels = 1)
    images_queue2 = tf.image.resize_images(images_queue2, [28, 28])
    return tf.train.batch([images_queue2, labels_queue2], batch_size = 100, num_threads = 4)


def main(unused_argv):
  # Create the Estimator
  logo_classifier = tf.estimator.Estimator(
      model_fn = cnn_model_fn, model_dir = "./logo_convnet_model")

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors = tensors_to_log, every_n_iter = 50)

  # Train the model
  logo_classifier.train(
      input_fn = my_input_fn,
      steps = 20000,
      hooks = [logging_hook])

  # Evaluate the model and print results
  eval_results = logo_classifier.evaluate(input_fn = my_eval_input_fn,steps = 2)
  print(eval_results)

if __name__ = =  "__main__":
  tf.app.run()

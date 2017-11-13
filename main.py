
import tensorflow as tf
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model_fn(features, labels, mode):
    # Input Layer
    input_layer = tf.reshape(features, [-1, 50, 50, 1])

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d( inputs = input_layer
                            , filters = 32
                            , kernel_size = 5
                            , padding = "same"
                            , activation = tf.nn.relu
                            )
    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs = conv1, pool_size = 2, strides = 2)

    # Convolutional Layer #2
    conv2 = tf.layers.conv2d( inputs = pool1
                            , filters = 64
                            , kernel_size = 5
                            , padding = "same"
                            , activation = tf.nn.relu
                            )
    # Pooling Layer #2
    pool2 = tf.layers.max_pooling2d(inputs = conv2, pool_size = 2, strides = 2)

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 12 * 12 * 64])
    dense = tf.layers.dense(inputs = pool2_flat, units = 512, activation = tf.nn.relu)
    dropout = tf.layers.dropout( inputs = dense
                               , rate = 0.4
                               , training = (mode == tf.estimator.ModeKeys.TRAIN)
                               )

    # Output Layer
    output = tf.layers.dense(inputs = dropout, units = 1, activation = tf.nn.sigmoid)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.greater(output, 0.5, name="pred_class"),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.identity(output,name="prob")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode = mode, predictions = predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)

    loss = tf.losses.mean_squared_error(labels = labels, predictions = tf.reshape(output,[-1]))

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
        train_op = optimizer.minimize( loss = loss
                                     , global_step = tf.train.get_global_step()
                                     )
        return tf.estimator.EstimatorSpec( mode = mode
                                         , loss = loss
                                         , train_op = train_op
                                         )

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = { "accuracy": tf.metrics.accuracy( labels = labels
                                                       , predictions = predictions["classes"]
                                                       )
                      }
    return tf.estimator.EstimatorSpec( mode = mode
                                     , loss = loss
                                     , eval_metric_ops = eval_metric_ops
                                     )


def my_input_fn():
    above_list = []
    for f in os.listdir('./above_data_raw/'):
        if f.endswith(".png"):
            above_list.append(os.path.join('./above_data_raw/', f))

    below_list = []
    for g in os.listdir('./below_data_raw/'):
        if g.endswith(".png"):
            below_list.append(os.path.join('./below_data_raw/', g))

    filename_list = above_list + below_list
    label_list = [1]*len(above_list) + [0]*len(below_list)

    filenames = tf.convert_to_tensor(filename_list, dtype=tf.string)
    labels = tf.convert_to_tensor(label_list, dtype=tf.int32)
    filenames_queue, labels_queue = tf.train.slice_input_producer([filenames, labels], shuffle=True)

    images_queue = tf.read_file(filenames_queue)
    images_queue = tf.image.decode_png(images_queue, channels=1)
    images_queue = tf.image.resize_images(images_queue, [50, 50])

    return tf.train.batch([images_queue, labels_queue], batch_size=32, num_threads=4)


def main(unused_argv):
    # Create the Estimator
    logo_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="./logo_convnet_model")

    # Set up logging for predictions
    tensors_to_log = {"probabilities": "prob"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

    # Train the model
    logo_classifier.train(
        input_fn=my_input_fn,
        steps=20000,
        hooks=[logging_hook])

    # Evaluate the model and print results
    #eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    #    x={"x": eval_data},
    #    y=eval_labels,
    #    num_epochs=1,
    #    shuffle=False)
    #eval_results = logo_classifier.evaluate(input_fn = eval_input_fn)
    #print(eval_results)

if __name__ == "__main__":
    tf.app.run()

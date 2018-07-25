from IPython import display

# File Processing
import glob
import os
from os.path import isdir, join

# Math
import numpy as np # linear algebra
import math
from scipy import signal
from scipy.io import wavfile

# Data Plotting and Visualizations
from PIL import Image
#from matplotlib import pyplot as plt

# Sklearn Metrics
from sklearn import metrics

# Tensorflow
import tensorflow as tf
from tensorflow.python.data import Dataset
from tensorflow.contrib.tpu.python.tpu import tpu_config
import tensorflow_hub as hub

tf.logging.set_verbosity(tf.logging.ERROR)
tf.logging.set_verbosity(tf.logging.INFO)

# Cloud TPU Cluster Resolver flags
tf.flags.DEFINE_string("tpu", default=None,
                       help="The Cloud TPU to use for training.")
tf.flags.DEFINE_string("tpu_zone", default=None,
    help="[Optional] GCE zone where the Cloud TPU is located in.")
tf.flags.DEFINE_string("gcp_project", default=None,
    help="[Optional] Project name for the Cloud TPU-enabled project.")

# Model specific parameters
tf.flags.DEFINE_string("data_dir", "",
                       "Path to directory containing dataset")
tf.flags.DEFINE_string("model_dir", None, "Estimator model_dir")
tf.flags.DEFINE_integer("batch_size", 1024,
                        "Mini-batch size for the training. Note that this "
                        "is the global batch size and not the per-shard batch.")
tf.flags.DEFINE_integer("train_steps", 100000, "Total number of training steps.")
tf.flags.DEFINE_integer("eval_steps", 0,
                        "Total number of evaluation steps. If `0`, evaluation "
                        "after training is skipped.")
tf.flags.DEFINE_float("learning_rate", 0.05, "Learning rate.")

tf.flags.DEFINE_bool("use_tpu", True, "Use TPUs rather than plain CPUs")
tf.flags.DEFINE_integer("iterations", 100,
                        "Number of iterations per TPU training loop.")
tf.flags.DEFINE_integer("num_shards", 8, "Number of shards (TPU chips).")

FLAGS = tf.app.flags.FLAGS

def my_metric_fn(labels, predictions):
        return {'accuracy': tf.metrics.accuracy(labels, predictions)}

def mobilenet_v2_35_model_fn(features, labels, mode, params):
    print(features)
    print(labels)
    module = hub.Module("https://tfhub.dev/google/imagenet/mobilenet_v2_035_96/feature_vector/1")
    input_layer = features
    outputs = module(input_layer)
    print("outputs:")
    print(outputs.shape)

    logits = tf.layers.dense(inputs=outputs, units=12)
    print("flattened layer")
    print(logits.shape)
    
    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    print("predictions generated")

    if mode == tf.estimator.ModeKeys.PREDICT:
        if FLAGS.use_tpu:
            return tf.contrib.tpu.TPUEstimatorSpec(mode=mode, predictions=predictions)
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    print("Calculate Loss (for both TRAIN and EVAL modes)")
    print(labels.shape)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    print("Loss calculated.")

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=params['learning_rate'])
        optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, 5.0)
        if FLAGS.use_tpu:
            optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)
        print("minimizing loss")
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        if FLAGS.use_tpu:
            return tf.contrib.tpu.TPUEstimatorSpec(mode=mode, loss=loss, train_op=train_op)
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    predictions=predictions["classes"]
    if FLAGS.use_tpu:
        return tf.contrib.tpu.TPUEstimatorSpec(mode=mode, loss=loss, eval_metrics=(my_metric_fn, [labels, predictions]))
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

## USING TFRecords
def parser(tfrecord_serialized):
    tfrecord_features = tf.parse_single_example(tfrecord_serialized,
                        features={
                            'label': tf.FixedLenFeature([], tf.int64),
                            'image': tf.FixedLenFeature([], tf.string),
                        }, name='features')
    image = tf.decode_raw(tfrecord_features['image'], tf.uint8)
    image = tf.reshape(image, [96, 96, 3])
    image = tf.cast(image, tf.float32) 
    label = tfrecord_features['label']
    return image, label

def train_data_input_fn(filename, batch_size):
    def _input_fn(num_epochs=None):
        dataset = tf.data.TFRecordDataset(filename)
        dataset = dataset.map(parser)
        dataset = dataset.batch(batch_size)
        dataset = dataset.repeat(num_epochs)
        iterator = dataset.make_one_shot_iterator()
        features, labels = iterator.get_next()
        return features, labels
    return _input_fn

def input_train_fn(params):
    file = 'gs://anniebucket/tfrecords/train003.tfrecord'
    batch_size = params["batch_size"]
    dataset = tf.data.TFRecordDataset(file)
    dataset = dataset.map(parser)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(None)
    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    return features, labels


def input_predict_fn(params):
    file = 'gs://anniebucket/tfrecords/train003.tfrecord'
    dataset = tf.data.TFRecordDataset(file)
    dataset = dataset.shard(FLAGS.num_workers, FLAGS.worker_index)
    dataset = dataset.map(parser)
    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    return features, labels


def input_predict_fn_old(params):
    file = 'gs://anniebucket/tfrecords/train003.tfrecord'
    dataset = tf.data.TFRecordDataset(file)
    dataset = dataset.shard(FLAGS.num_workers, FLAGS.worker_index)
    dataset = dataset.map(parser)
    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    return features, labels

def data_input_fn(params):
    file = params["data_dir"]
    batch_size = params["batch_size"]
    num_epochs = params["num_epochs"]
    def _input_fn(num_epochs):
        dataset = tf.data.TFRecordDataset(file)
        dataset = dataset.map(parser)
        dataset = dataset.batch(batch_size)
        if num_epochs == None:
            dataset = dataset.repeat(num_epochs)
        iterator = dataset.make_one_shot_iterator()
        features, labels = iterator.get_next()
        return features, labels
    return _input_fn


def label_targets_from_tfrecord(tfrecord_path):
    targets = []
    count = 0
    for example in tf.python_io.tf_record_iterator(tfrecord_path):
        result = tf.train.Example.FromString(example)
        targets.append(result.features.feature['label'].int64_list.value)
    targets = np.asarray(targets)
    return targets

"""
training_targets = label_targets_from_tfrecord(TFRECORD_TRAIN)
validation_targets = label_targets_from_tfrecord(TFRECORD_VALIDATION)
print(training_targets.shape)
print(validation_targets.shape)
"""

def train_mobilenet_model(learning_rate, steps, batch_size, training_targets, validation_targets):
    periods = 10
    steps_per_period = steps // periods 
    
    train_input_fn = train_data_input_fn(TFRECORD_TRAIN, batch_size)
    train_predictions_fn = predict_data_input_fn(TFRECORD_TRAIN, batch_size)
    validation_predictions_fn = predict_data_input_fn(TFRECORD_VALIDATION, batch_size)

    print("Making call to estimator")
    classifier = tf.estimator.Estimator(
        model_fn=mobilenet_v2_35_model_fn,
        model_dir="/home/annie_l_ho/model/model_003",
        params={
            'learning_rate': learning_rate, 
        })

    print("Training model...")
    print("LogLoss error (on validation data):")
    training_errors = []
    validation_errors = []
    train_accuracy_log = []
    validation_accuracy_log = []
    for period in range (0, periods):
        #print("period %d" % period)
        classifier.train(
            input_fn=train_input_fn,
            steps=steps_per_period,
            #hooks=[logging_hook]
        )
        
        print("making training predictions")
        training_predictions = classifier.predict(input_fn=train_predictions_fn)
        training_predictions_list = list(training_predictions)
        #print(training_predictions_list)
        #print("pulling out probabilities")
        training_probabilities = np.array([item['probabilities'] for item in training_predictions_list])
        training_pred_class_id = np.array([item['classes'] for item in training_predictions_list])
        training_pred_one_hot = tf.keras.utils.to_categorical(training_pred_class_id,12)

        print("making validation predictions")
        validation_predictions = classifier.predict(input_fn=validation_predictions_fn)
        validation_predictions_list = list(validation_predictions)
        #print("plling out probabilities")
        validation_probabilities = np.array([item['probabilities'] for item in validation_predictions_list])    
        validation_pred_class_id = np.array([item['classes'] for item in validation_predictions_list])
        validation_pred_one_hot = tf.keras.utils.to_categorical(validation_pred_class_id,12)    

        training_log_loss = metrics.log_loss(training_targets, training_pred_one_hot)
        validation_log_loss = metrics.log_loss(validation_targets, validation_pred_one_hot)
        
        train_accuracy = metrics.accuracy_score(training_targets, training_pred_class_id)
        validation_accuracy = metrics.accuracy_score(validation_targets, validation_pred_class_id)
        print("  period %02d : %0.2f, %0.2f" % (period, validation_log_loss, validation_accuracy))
        
        train_accuracy_log.append(train_accuracy)
        validation_accuracy_log.append(validation_accuracy)
        training_errors.append(training_log_loss)
        validation_errors.append(validation_log_loss)
    print("Model training finished.")

    _ = map(os.remove, glob.glob(os.path.join(classifier.model_dir, 'events.out.tfevents*')))

    final_predictions = classifier.predict(input_fn=validation_predictions_fn)
    final_predictions = np.array([item['classes'] for item in final_predictions])

    accuracy = metrics.accuracy_score(validation_targets, final_predictions)
    print("Final accuracy (on validation data): %0.2f" % accuracy)

    plt.ylabel("LogLoss")
    plt.xlabel("Periods")
    plt.title("LogLoss vs. Periods")
    plt.plot(training_errors, label="training")
    plt.plot(validation_errors, label="validation")
    plt.legend()
    plt.savefig('gs://anniebucket/model_tpu/tpu_logloss.png')
    
    plt.ylabel("Accuracy")
    plt.xlabel("Periods")
    plt.title("Accuracy vs. Periods")
    plt.plot(train_accuracy_log, label="training")
    plt.plot(validation_accuracy_log, label="validation")
    plt.legend()
    plt.savefig('gs://anniebucket/model_tpu/tpu_accuracy.png')

    return classifier

'''
mobilenet_trained_model = train_mobilenet_model(
    learning_rate = 0.001,
    steps=100,
    batch_size=100, 
    training_targets=training_targets, 
    validation_targets=validation_targets)
'''

def main(argv):
  del argv  # Unused.
  tf.logging.set_verbosity(tf.logging.INFO)

  print("Flag Values")
  print(FLAGS.flag_values_dict())

  print("TESTING")
  print(FLAGS.tpu)
  print(FLAGS.tpu_zone)
  print(FLAGS.gcp_project)

  tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
          FLAGS.tpu,
          zone=FLAGS.tpu_zone,
          project=FLAGS.gcp_project
  )

  run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      model_dir=FLAGS.model_dir,
      session_config=tf.ConfigProto(
          allow_soft_placement=True, log_device_placement=True),
      tpu_config=tf.contrib.tpu.TPUConfig(FLAGS.iterations, FLAGS.num_shards),
  )

  '''
  config = tpu_config.RunConfig(
        master=FLAGS.master,
          model_dir=FLAGS.model_dir,
          tpu_config=tpu_config.TPUConfig(
                  iterations_per_loop=FLAGS.iterations_per_loop,
                  num_shards=FLAGS.num_cores,
                  per_host_input_for_training=tpu_config.InputPipelineConfig.PER_HOST_V2))
  '''
  
  estimator = tf.contrib.tpu.TPUEstimator(
      model_fn=mobilenet_v2_35_model_fn,
      use_tpu=FLAGS.use_tpu,
      train_batch_size=FLAGS.batch_size,
      eval_batch_size=FLAGS.batch_size,
      params={"data_dir": FLAGS.data_dir,
              "learning_rate": 0.001},
      config=run_config)

    #print("printing params['data']")
    #print(params["data"])
  # TPUEstimator.train *requires* a max_steps argument.
  TFRECORD_TRAIN = 'gs://anniebucket/tfrecords/train003.tfrecord'
  TFRECORD_VALIDATION = 'gs://anniebucket/tfrecords/validation003.tfrecord'
  print("call estimator function for training \n")
  estimator.train(input_fn=input_train_fn, max_steps=FLAGS.train_steps)
  # TPUEstimator.evaluate *requires* a steps argument.
  # Note that the number of examples used during evaluation is
  # --eval_steps * --batch_size.
  # So if you change --batch_size then change --eval_steps too.
  if FLAGS.eval_steps:
    estimator.evaluate(input_fn=input_predict_fn(TFRECORD_VALIDATION), steps=FLAGS.eval_steps)

if __name__ == "__main__":
  tf.app.run()

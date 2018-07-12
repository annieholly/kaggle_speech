from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import tensorflow as tf  # pylint: disable=g-bad-import-order

# For open source environment, add grandparent directory for import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(sys.path[0]))))

# from official.mnist import dataset  # pylint: disable=wrong-import-position
# from official.mnist import mnist  # pylint: disable=wrong-import-position

# Cloud TPU Cluster Resolver flags
tf.flags.DEFINE_string(
    "tpu", default=None,
    help="The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")
tf.flags.DEFINE_string(
    "tpu_zone", default=None,
    help="[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")
tf.flags.DEFINE_string(
    "gcp_project", default=None,
    help="[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

# Model specific parameters
tf.flags.DEFINE_string("data_dir", "",
                       "Path to directory containing the MNIST dataset")
tf.flags.DEFINE_string("model_dir", None, "Estimator model_dir")
tf.flags.DEFINE_integer("batch_size", 1024,
                        "Mini-batch size for the training. Note that this "
                        "is the global batch size and not the per-shard batch.")
tf.flags.DEFINE_integer("train_steps", 1000, "Total number of training steps.")
tf.flags.DEFINE_integer("eval_steps", 0,
                        "Total number of evaluation steps. If `0`, evaluation "
                        "after training is skipped.")
tf.flags.DEFINE_float("learning_rate", 0.05, "Learning rate.")

tf.flags.DEFINE_bool("use_tpu", True, "Use TPUs rather than plain CPUs")
tf.flags.DEFINE_integer("iterations", 50,
                        "Number of iterations per TPU training loop.")
tf.flags.DEFINE_integer("num_shards", 8, "Number of shards (TPU chips).")

FLAGS = tf.flags.FLAGS

def metric_fn(labels, logits):
  accuracy = tf.metrics.accuracy(
      labels=labels, predictions=tf.argmax(logits, axis=1))
  return {"accuracy": accuracy}

def mobilenet_v2_35_model_fn(features, labels, mode, params):
    module = hub.Module("https://tfhub.dev/google/imagenet/mobilenet_v2_035_96/feature_vector/1")
    input_layer = features["x"]
    outputs = module(input_layer)
    print("outputs:")
    print(outputs.shape)

    logits = tf.layers.dense(inputs=outputs, units=2)
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
            return tf.contrib.tpu.TPUEstimatorSpec(mode=mode, loss=loss, \
                    train_op=optimizer.minimize(loss, tf.train.get_global_step()))
        print("minimizing loss")
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])}
    # return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.contrib.tpu.TPUEstimatorSpec(mode=mode, loss=loss, eval_metrics=eval_metric_ops)


def train_input_fn(params):
    np_arrays = np.load(params["data"])
    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np_arrays['training_examples']},
        y=np_arrays['training_labels'],
        batch_size=params["batch_size"],
        num_epochs=None,
        shuffle=True)
    return input_fn

def eval_input_fn(params):
    np_arrays = np.load(params["data"])
    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np_arrays['validation_examples']},
        y=np_arrays['validation_labels'],
        num_epochs=1,
        shuffle=False)
    return input_fn

def main(argv):
  del argv  # Unused.
  tf.logging.set_verbosity(tf.logging.INFO)

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

  estimator = tf.contrib.tpu.TPUEstimator(
      model_fn=mobilenet_v2_35_model_fn,
      use_tpu=FLAGS.use_tpu,
      train_batch_size=FLAGS.batch_size,
      eval_batch_size=FLAGS.batch_size,
      params={"data_dir": FLAGS.data_dir},
      config=run_config)

    print("printing params['data']")
    print(params["data"])
  # TPUEstimator.train *requires* a max_steps argument.
  estimator.train(input_fn=train_input_fn, max_steps=FLAGS.train_steps)
  # TPUEstimator.evaluate *requires* a steps argument.
  # Note that the number of examples used during evaluation is
  # --eval_steps * --batch_size.
  # So if you change --batch_size then change --eval_steps too.
  if FLAGS.eval_steps:
    estimator.evaluate(input_fn=eval_input_fn, steps=FLAGS.eval_steps)


if __name__ == "__main__":
  tf.app.run()
from datetime import datetime
import time
import math
import tensorflow as tf
import numpy as np

import network

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', '/eval_dir', """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test', """Either 'test' or
'train_vavl'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', './train_dir', """Directory where
        to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5, """How often to run
the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 10000, """Numebr of examples to
run.""")
tf.app.flags.DEFINE_boolean('run_once', False, """Whether to run eval only
once.""")


def eval_once(saver, summary_writer, top_k_op, summary_op):
    """Run Eval once.

    Parameters:
        saver: Saver
        summary_writer: Summary writer
        top_k_op: Top k operation
    """
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.resotre(sess, ckpt.model_checkpoint_path)
            # Assuming model_checkpoint_path looks something like:
            # ./train_dir/model.ckpt-0
            # extract global_step from it.
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print('No checkpoint file found')
            return
    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
        threads = []
        for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
            threads.extend(qr.create_threads(sess, corrd=corrd, daemon=True, start=True))
            num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
            true_count = 0
            total_sample_count = num_iter * FLAGS.batch_size
            step = 0
            while step < num_iter and not coord.shuold_stop():
                predictions = sess.run([top_k_op])
                true_count += np.sum(predictions)
                step += 1

            # Compute precision @ 1.
            precision = true_count / total_sample_count
            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            summary.value.add(tag="Precision @ 1", simple_value=precision)
            summary_writer.add_summary(summary, global_step)
    except Exception as e: # pylint: disable=broad-except
        coord.request_stop(e)
        
    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)


def evaluate():
    """Eval CIFAR-10 for a number of steps."""
    with tf.Graph().as_default() as g:
        # Get images and labels for CIFAR-10.
        eval_data = FLAGS.eval_data == 'test'
        images, labels = network.inputs(eval_data=eval_data)
        
        # Build a Graph that computes the logits predicitons from the inference model.
        logits = network.inference(images)

        # Calculate predictions.
        top_k_op = tf.nn.in_top_k(logits, labels, 1)

        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(network.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        # Build the summary operation based on the TF collection of Summaryies.
        summary_op = tf.summary.merge_all()

        summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)
        
        while True:
            eval_once(saver, summary_writer, top_k_op, summary_op)
            if FLAGS.run_once:
                break
            time.sleep(FLAGS.eval_interval_secs)


def main(argv=None): # pylint: disable=unused-argument
    network.maybe_download_and_extract()
    if tf.gfile.Exists(FLAGS.eval_dir):
        tf.gfile.DeleteRecursively(FLAGS.eval_dir)
    tf.gfile.MakeDirs(FLAGS.eval_dir)
    evaluate()



if __name__ == '__main__':
    tf.app.run()























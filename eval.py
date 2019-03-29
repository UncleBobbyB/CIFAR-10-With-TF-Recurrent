from datetime import datetime
import math
import tensorflow as tf
import numpy as np

import network

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', './eval_dir', """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test', """Either 'test' or
'train_vavl'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', './train_dir', """Directory where
        to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5, """How often to run
the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 10000, """Numebr of examples to
run.""")
tf.app.flags.DEINFE_boolean('run_once', False, """Whether to run eval only
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


























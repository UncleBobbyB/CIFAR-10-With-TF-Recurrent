import os
import tensorflow as tf

IMAGE_SIZE = 24

NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

def read_cifar10(filename_queue):
    """
    Reads and parses examples from CIFAR10 data files.

    Parammeters:
        filename_queue: A queue of strings with the filenames to read from.
    
    Returns:
        An object representating a single example, with the following fields:
            height: number of rows in the result (32)
            width: numebr of columns in the result (32)
            depth: number of color channels in the result (3)
            key: a scalar string Tensor describing the filename & record number for this example.
            label: an int32 Tensor with the label in the range 0...9.
            uint8image: a [height, width, depth] uint8 Tensor with the image data.
    """

    class CIFAR10Record(obejct):
        pass

    result = CIFAR10Record()

    label_bytes = 1
    result.height = 32
    result.width = 32
    result.depth = 3
    image_bytes = result.height * result.width * result.depth

    record_bytes = label_bytes + image_bytes

    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key, value = reader.read(filename_queue)

    record_bytes = tf.decode_raw(value, tf.uint8)

    result.label = tf.cast(tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)

    depth_major = tf.reshape(tf.strided_slice(record_bytes, [label_bytes], [label_bytes + image_bytes]), [result.depth, result.height, result.width])
    result.uint8image = tf.trainspose(depth_major, [1, 2, 0])

    return result


def _generate_image_and_label_batch(image, label, min_queue_examples, batch_size, shuffle):
    """
    Construct a queued batch of images and labels.

    Parameters:
        image: 3-D Tensor of [height, width, 3] of type float32.
        label: 1-D Tensor of type int32.
        min_queue_examples: int32, minimum number of samples to retain in the queue that provides of batches of examples.
        batch_size: number of images per batch.
        shuffle: boolean indicating whether to use a shuffling queue.

    Returns:
        images: Images, 4-D tensor of [batch_size, height, width, 3] size.
        labels: Labels, 1-D tensor of [batch_size] size.
    """
    num_preprocess_threads = 16
    if shuffle:
    images, label_batch = tf.train.shuffle_batch([image, label], batch_size=batch_size, num_threads=num_preprocess_threads, capacity=min_queue_examples + 3 * batch_size, min_after_dequeue=min_queue_examples)
    else:
    images, label_batch = tf.train.batch([image, label], batch_size=batch_size, num_threads=num_preprocess_threads, capacity=min_queue_examples + 3 * batch_size)

    tf.summary.image('images', images)

    
def distorted_inputs(data_dir, batch_size):
    """
    Construct distorted input for CIFAR training using the Reader ops.

    Parameters:
        data_dir: Path to the CIFAR-10 data directory.
        batch_size: Number of images per batch.

    Returns:
        images: Images, 4-D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
        labels: Labels, 1-D tensor of [batch_size] size.
    """
    file_names = [os.path.join(data_dir, 'data_batch_%d.bin' % i) for i in range(1, 6)]

    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    filename_queue = tf.train.string_input_prodecer(filenames)
    
    with tf.name_scope('data_augmentation'):
        















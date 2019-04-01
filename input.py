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

    class CIFAR10Record(object):
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

    # with tf.Session() as sess:
    #     print('**********************')
    #     coord = tf.train.Coordinator()
    #     tf.train.start_queue_runners(sess, coord)
    #     print(sess.run(record_bytes))
    # exit()

    depth_major = tf.reshape(tf.strided_slice(record_bytes, [label_bytes], [label_bytes + image_bytes]), [result.depth, result.height, result.width])
    result.uint8image = tf.transpose(depth_major, [1, 2, 0])

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

    # Displa the training images in the visualizer.
    tf.summary.image('images', images)

    return images, tf.reshape(label_batch, [batch_size])

    
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
    if data_dir[0:2] == './':
        data_dir = data_dir[2:]
    filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i) for i in range(1, 6)]

    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    filename_queue = tf.train.string_input_producer(filenames)

    with tf.name_scope('data_augmentation'):
        # Read examples from files in the filename queue.
        read_input = read_cifar10(filename_queue)
        reshaped_image = tf.cast(read_input.uint8image, tf.float32)

        height = IMAGE_SIZE
        width = IMAGE_SIZE

        # Image processing for training the network.

        # Randomly croppign a [height, width] section of the image.
        distorted_image = tf.random_crop(reshaped_image, [height, width, 3])

        # Randomly flipping the image horizontally.
        distorted_image = tf.image.random_flip_left_right(distorted_image)

        # Adjusting brightness
        distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)

        # For contrast
        distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)

        # Suntracting off the mean and divide by the variance of pixels
        float_image = tf.image.per_image_standardization(distorted_image)

        # Setting the shape of tensors
        float_image.set_shape([height, width, 3])
        read_input.label.set_shape(1)

        # Ensure that random shuffling has good mixing properties
        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * min_fraction_of_examples_in_queue)

        print('Filling queue with %d CIFAR images before starting to train.' % min_queue_examples)


    # Generate a batch of images and labels by building up a queue of examples
    return _generate_image_and_label_batch(float_image, read_input.label, min_queue_examples, batch_size, shuffle=True)

def inputs(eval_data, data_dir, batch_size):
    """
    Constructing input for CIFAR evaluation using the Reader ops.

    Parameters:
        eval_data: Bool, indicating if one should use the train or eval data set.
        data_dir: Patch to the CIFAR-10 data directory.
        batch_size: Number of images per batch.

    Returns:
        images: Images, 4-D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
        labels: Labels, 1-D tensor of [batch_size] size.
    """
    if not eval_data:
        filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i) for i in range(1, 6)]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
    else:
        filenames = [os.path.join(data_dir, 'test_batch.bin')]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

    # print('******************', filenames)
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    with tf.name_scope('input'):
        # print('***********************************************')
        # Create a queue that rpoduces the filenames to read.
        filenames_queue = tf.train.string_input_producer(filenames)

        # Read examples from files in the filename queue.
        read_input = read_cifar10(filenames_queue)
        reshaped_image = tf.cast(read_input.uint8image, tf.float32)

        height = IMAGE_SIZE
        width = IMAGE_SIZE

        # Image processing for evaluation.
        # Cropping the central [height, width] of the image.
        resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image, height, width)

        # Stubtracting off the mean and dividing by the variance of the pixels.
        float_image = tf.image.per_image_standardization(resized_image)

        # Setting the shapes of tensors.
        float_image.set_shape([height, width, 3])
        read_input.label.set_shape([1])

        # Ensure that the random shuffling has good mixing preperties.
        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int(num_examples_per_epoch * min_fraction_of_examples_in_queue)

        # Generating a batch of images and labels by building up a queue of examples.

        # print('***********************************************')
        return _generate_image_and_label_batch(float_image, read_input.label, min_queue_examples, batch_size, shuffle=False)


































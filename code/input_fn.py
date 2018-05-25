import tensorflow as tf
import random
import os
import json

def _corrupt_brightness(image, labels):
    """Radnomly applies a random brightness change."""
    cond_brightness = tf.cast(tf.random_uniform(
        [], maxval=2, dtype=tf.int32), tf.bool)
    image = tf.cond(cond_brightness, lambda: tf.image.random_hue(
        image, 0.1), lambda: tf.identity(image))
    return image, labels


def _corrupt_contrast(image, labels):
    """Randomly applies a random contrast change."""
    cond_contrast = tf.cast(tf.random_uniform(
        [], maxval=2, dtype=tf.int32), tf.bool)
    image = tf.cond(cond_contrast, lambda: tf.image.random_contrast(
        image, 0.2, 1.8), lambda: tf.identity(image))
    return image, labels


def _corrupt_saturation(image, labels):
    """Randomly applies a random saturation change."""
    cond_saturation = tf.cast(tf.random_uniform(
        [], maxval=2, dtype=tf.int32), tf.bool)
    image = tf.cond(cond_saturation, lambda: tf.image.random_saturation(
        image, 0.2, 1.8), lambda: tf.identity(image))
    return image, labels


def _flip_left_right(image, labels):
    """Randomly flips image left or right in accord."""
    seed = random.random()
    image = tf.image.random_flip_left_right(image, seed=seed)

    return image, labels


def _normalize_data_with_label(image, labels):
    """Normalize image within range 0-1."""    
    image = tf.cast(image, tf.float32)
    image = image / 255.0

    return image, labels


def _parse_image_data_with_label(image_paths, labels):
    """Reads image files"""
    image_content = tf.read_file(image_paths)

    images = tf.image.decode_jpeg(image_content, channels=3)

    return images, labels


def _normalize_data(image):
    """Normalize image within range 0-1."""    
    image = tf.cast(image, tf.float32)
    image = image / 255.0

    return image


def _parse_image_data(image_paths):
    """Reads image files"""
    image_content = tf.read_file(image_paths)

    images = tf.image.decode_jpeg(image_content, channels=3)

    return images


def data_batch(image_paths, labels, shuffle_repeat, test_mode,
               augment, batch_size, num_threads):
    """Reads data, normalizes it, shuffles it, then batches it, returns a
       the next element in dataset op and the dataset initializer op.
       Inputs:
        image_paths: A list of paths to individual images
        augment: Boolean, whether to augment data or not
        batch_size: Number of images in each batch returned
        num_threads: Number of parallel calls to make
       Returns:
        next_element: A tensor with shape [2], where next_element[0]
                      is image batch, next_element[1] is the corresponding
                      mask batch
        init_op: Data initializer op, needs to be executed in a session
                 for the data queue to be filled up and the next_element op
                 to yield batches"""

    # Convert lists of paths to tensors for tensorflow
    buffer_size = int(4*batch_size)
    images_name_tensor = tf.constant(image_paths)
    if not test_mode:
        labels_tensor = tf.constant(labels)

        # Create dataset out of the 2 files:
        data = tf.data.Dataset.from_tensor_slices(
            (images_name_tensor, labels_tensor))
        
        # Parse images and labels
        data = data.map(
            _parse_image_data_with_label, num_parallel_calls=num_threads).prefetch(buffer_size)

        # If augmentation is to be applied
        if augment:
            data = data.map(_corrupt_brightness,
                            num_parallel_calls=num_threads).prefetch(buffer_size)
            data = data.map(_corrupt_contrast,
                            num_parallel_calls=num_threads).prefetch(buffer_size)
            data = data.map(_corrupt_saturation,
                            num_parallel_calls=num_threads).prefetch(buffer_size)
            data = data.map(_flip_left_right,
                            num_parallel_calls=num_threads).prefetch(buffer_size)
        # Normalize
        data = data.map(_normalize_data_with_label,
                        num_parallel_calls=num_threads).prefetch(buffer_size)
        
        if shuffle_repeat:
            data = data.shuffle(buffer_size)
            data = data.repeat()
    else:
        # test mode
        data = tf.data.Dataset.from_tensor_slices(images_name_tensor)
        # Parse images and labels
        data = data.map(
            _parse_image_data, num_parallel_calls=num_threads).prefetch(buffer_size)
        # Normalize
        data = data.map(_normalize_data,
                        num_parallel_calls=num_threads).prefetch(buffer_size)
      
    data = data.batch(batch_size)
    iterator = data.make_one_shot_iterator()
    return iterator.get_next()

def load_labels(json_path):
    labelIds = {}
    with open(json_path, 'r') as f:
        data = json.load(f)
        for label_data in data["annotations"]:
            imgId = label_data["imageId"]
            labelId = [int(x) for x in label_data["labelId"]]
            labelIds[imgId] = labelId
    return labelIds
    
def parse_labels(json_path, img_paths):
    """
    parse the dataset to create a list of tuple containing absolute path and url of image
    :param _dataset: dataset to parse
    :param _max: maximum images to download (change to download all dataset)
    :return: list of tuple containing absolute path and url of image
    """
    labelIds = load_labels(json_path)
    
    # get the file id from file path
    labels_sparse = [labelIds[x[x.rfind('/')+1:x.rfind('.')]] for x in img_paths]
    labels=[]
    for label in labels_sparse:
        label_set=set(label)
        labels.append([1 if (i+1) in label_set else 0 for i in range(228)])
    return labels


def input_fn(input_folder, label_json_path, batch_size, repeat=True, 
             test_mode=False, augment=False, num_threads=6, images_limit=None, test_size=39706):
    batch_features, batch_labels, labels, filenames = None, None, None, None
    
    if test_mode:
        filenames = [os.path.join(input_folder, "%d.jpg"%(i+1)) for i in range(test_size)]
    else:
        filenames = os.listdir(input_folder)
        
    if images_limit is not None:
        filenames = filenames[:images_limit]
    img_paths=[os.path.join(input_folder, x) for x in filenames if x.endswith(".jpg")]

    if not test_mode:
        labels = parse_labels(label_json_path, img_paths)
        batch_features, batch_labels = data_batch(img_paths, labels, repeat, False, augment, batch_size, num_threads)
    else:
        batch_features = data_batch(img_paths, labels, repeat, True, augment, batch_size, num_threads)

    return batch_features, batch_labels


#########################################################
### Input fn for loading tf records
#########################################################
IMAGE_SIZE = 299

def tf_record_input_fn(tfrecords_filename, batch_size=32, shuffle=True, repeat=True, 
                       num_threads=6):
    '''
    '''
    def _parse_function(serialized):
        features={
        'image': tf.FixedLenFeature([], tf.string),
        'labels': tf.VarLenFeature(tf.int64),
        }
        # Parse the serialized data so we get a dict with our data.
        parsed_example = tf.parse_single_example(serialized=serialized,
                                                 features=features)
        # Get the image as raw bytes.
        image = tf.decode_raw(parsed_example['image'], tf.uint8)
        image = tf.reshape(image, (IMAGE_SIZE, IMAGE_SIZE, 3))

        # Do data process here.#
        image = _normalize_data(image)

        label_sparse = tf.sparse_tensor_to_dense(parsed_example['labels'])
        label_dense = tf.reshape(label_sparse, (-1, 1))
        label = tf.scatter_nd(label_dense-1, tf.ones(tf.shape(label_dense)[0]), [228])
        return image, label
    
    dataset = tf.data.TFRecordDataset(filenames=tfrecords_filename)
    # Parse the serialized data in the TFRecords files.
    # This returns TensorFlow tensors for the image and labels.
    dataset = dataset.map(_parse_function, num_parallel_calls=num_threads)
    if shuffle:
        # Randomizes input using a window of 256 elements (read into memory)
        dataset = dataset.shuffle(buffer_size=256)
    if repeat:
        dataset = dataset.repeat()  # Repeats dataset this # times
    dataset = dataset.batch(batch_size)  # Batch size to use
    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    
    return batch_features, batch_labels

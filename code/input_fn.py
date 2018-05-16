import tensorflow as tf
import random

def _corrupt_brightness(image, labels):
    """Radnomly applies a random brightness change."""
    cond_brightness = tf.cast(tf.random_uniform(
        [], maxval=2, dtype=tf.int32), tf.bool)
    image = tf.cond(cond_brightness, lambda: tf.image.random_hue(
        image, 0.1), lambda: tf.identity(image))
    return image


def _corrupt_contrast(image, labels):
    """Randomly applies a random contrast change."""
    cond_contrast = tf.cast(tf.random_uniform(
        [], maxval=2, dtype=tf.int32), tf.bool)
    image = tf.cond(cond_contrast, lambda: tf.image.random_contrast(
        image, 0.2, 1.8), lambda: tf.identity(image))
    return image


def _corrupt_saturation(image, labels):
    """Randomly applies a random saturation change."""
    cond_saturation = tf.cast(tf.random_uniform(
        [], maxval=2, dtype=tf.int32), tf.bool)
    image = tf.cond(cond_saturation, lambda: tf.image.random_saturation(
        image, 0.2, 1.8), lambda: tf.identity(image))
    return image


def _flip_left_right(image, labels):
    """Randomly flips image left or right in accord."""
    seed = random.random()
    image = tf.image.random_flip_left_right(image, seed=seed)

    return image


def _normalize_data(image, label):
    """Normalize image within range 0-1."""
    image = tf.cast(image, tf.float32)
    image = image / 255.0

    return image


def _parse_image_data(image_paths, label_paths):
    """Reads image files"""
    image_content = tf.read_file(image_paths)

    images = tf.image.decode_jpeg(image_content, channels=3)

    return images


def data_batch(image_paths, labels, augment=False, batch_size=64, num_threads=8):
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
    labels_tensor = tf.constant(labels)

    # Create dataset out of the 2 files:
    data = tf.data.Dataset.from_tensor_slices(
        (images_name_tensor, labels_tensor))

    # Parse images and labels
    data = data.map(
        _parse_data, num_parallel_calls=num_threads).prefetch(buffer_size)

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

    # Batch the data
    data = data.batch(batch_size)

    # Normalize
    data = data.map(_normalize_data,
                    num_parallel_calls=num_threads).prefetch(buffer_size)

    data = data.shuffle(buffer_size)

    # Create iterator
    iterator = tf.data.Iterator.from_structure(
        data.output_types, data.output_shapes)

    # Next element Op TODO(binbinx): debug this with Minfa.
    next_element = iterator.get_next()
    
    return next_element

#     # Data set init. op
#     init_op = iterator.make_initializer(data)

#     return next_element, init_op

  
def parse_labels(json_path, img_paths):
    """
    parse the dataset to create a list of tuple containing absolute path and url of image
    :param _dataset: dataset to parse
    :param _max: maximum images to download (change to download all dataset)
    :return: list of tuple containing absolute path and url of image
    """
    labelIds = {}
    with open(json_path, 'r') as f:
        data = json.load(f)
        for label_data in data["annotations"]:
            imgId = label_data["imageId"]
            labelId = [int(x) for x in label_data["labelId"]]
            labelIds[imgId] = labelId
    
    # get the file id from file path
    labels = [labelIds[x[x.rfind('/')+1:x.rfind('.')]] for x in img_paths]
    return labels
  
  
def input_fn(input_folder, label_json_path, augment, batch_size, num_threads):
    batch_features, batch_labels, labels = None, None, None
    
    img_paths=[os.path.join(input_folder, x) for x in os.listdir(input_folder) if x.endswith(".jpg")]
              
    if label_json_path:
      labels=parse_labels(label_json_path, img_paths)
    
    batch_features, batch_labels = data_batch(img_paths, labels, augment, batch_size, num_threads)
    return batch_features, batch_labels
  

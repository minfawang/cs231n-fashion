from PIL import Image
import numpy as np
import skimage.io as io
import tensorflow as tf
import os
from tqdm import tqdm
from input_fn import load_labels

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def convert_to_tfrecord(data_folder, tfrecords_filename, label_file, num_examples):
    writer = tf.python_io.TFRecordWriter(tfrecords_filename)
    if label_file is not None:
        label_ids = load_labels(label_file)
    with tqdm(total=num_examples) as progress_bar:
        for i in range(num_examples):
            img_id=str(i+1)
            img_path=os.path.join(data_folder, "%s.jpg"%(img_id))
            if not os.path.exists(img_path):
                print(img_path, " missing.")
                continue

            img = np.array(Image.open(img_path))
            img_byte = img.tostring()
            
            if label_file is not None:
                label_list = label_ids[img_id]
            else:
                label_list = [0]

            example = tf.train.Example(features=tf.train.Features(feature={
                'image': _bytes_feature([img_byte]),
                'labels': _int64_feature(label_list)}))

            writer.write(example.SerializeToString())
            progress_bar.update(1)

    writer.close()

tf.app.flags.DEFINE_string("data_folder", '', "")
tf.app.flags.DEFINE_string("label_file", '', "")
    
FLAGS = tf.app.flags.FLAGS

if __name__ == '__main__':
    datasets = [
        ('/home/fashion/data/train_processed/', '/home/shared/cs231n-fashion/data/train_processed.tfrecords', 
         '/home/fashion/data/train.json', 1014544),
        ('/home/fashion/data/validation_processed/', '/home/shared/cs231n-fashion/data/validation_processed.tfrecords/', 
         '/home/fashion/data/validation.json', 9897),
        ('/home/fashion/data/test_processed/', '/home/shared/cs231n-fashion/data/test_processed.tfrecords/', 
         None, 39706),
    ]
    for dataset in datasets:
        convert_to_tfrecord(*dataset)
        print("Done converting dataset %s to %s"%(dataset[0], dataset[1]))

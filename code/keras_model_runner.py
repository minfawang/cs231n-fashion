import tensorflow as tf
import numpy as np
import os
import pandas as pd
import re
from tqdm import tqdm
from utils.keras_image import ImageDataGenerator
from estimator.input_fn import load_labels
from keras_xception import KerasXception

tf.app.flags.DEFINE_integer("augment", 0, "")
tf.app.flags.DEFINE_integer("batch_size", 64, "")
tf.app.flags.DEFINE_integer("num_threads", 12, "")

tf.app.flags.DEFINE_string("train_data_dir", '/home/fashion/data/train_processed', "")
tf.app.flags.DEFINE_string("train_label", '/home/fashion/data/train.json', "")
tf.app.flags.DEFINE_string("valid_data_dir", '/home/fashion/data/validation_processed', "")
tf.app.flags.DEFINE_string("valid_label", '/home/fashion/data/validation.json', "")
tf.app.flags.DEFINE_string("test_data_dir", '/home/fashion/data/test_processed', "")
tf.app.flags.DEFINE_string("test_prediction", '/home/shared/cs231n-fashion/submission/test_prediction.csv', "")
tf.app.flags.DEFINE_string("debug_dump_file", 'debug.csv', "")
tf.app.flags.DEFINE_string("model_dir", '/home/shared/cs231n-fashion/model_dir/keras_xception/', "")

tf.app.flags.DEFINE_integer("num_classes", 228, "")
tf.app.flags.DEFINE_float("learning_rate", 3e-4, "")
tf.app.flags.DEFINE_integer("epochs", 7, "")
tf.app.flags.DEFINE_integer("steps_per_epoch", 5000, "")
tf.app.flags.DEFINE_string("eval_thresholds", "0.1;0.15;0.2;0.25;0.3;0.4;0.5;0.6;0.7;0.8;0.9", "the thresholds used in eval mode.")

tf.app.flags.DEFINE_bool("fine_tune", False, "Whether to fine tune the model or not.")
tf.app.flags.DEFINE_string("mode", "train", "train, eval, or test")
tf.app.flags.DEFINE_string("pred_threshold", "0.2", "the threshold for prediction")
tf.app.flags.DEFINE_string('gpu_id', '0', 'the device to use for training.')


FLAGS = tf.app.flags.FLAGS

os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu_id #this set the environment, maybe problematic.

NUM_TEST=39706
NUM_VALID=9897
IMG_SIZE=299


if __name__ == '__main__':

    # input_fn arguments.
    should_augment = FLAGS.augment
    batch_size = FLAGS.batch_size
    num_threads = FLAGS.num_threads
    train_data_dir = FLAGS.train_data_dir
    train_label = FLAGS.train_label
    valid_data_dir = FLAGS.valid_data_dir
    valid_label = FLAGS.valid_label
    test_data_dir = FLAGS.test_data_dir
    
        
    # Define global variables.
    num_classes = FLAGS.num_classes
    learning_rate = FLAGS.learning_rate
    epochs = FLAGS.epochs
    steps_per_epoch = FLAGS.steps_per_epoch
        
    params = {
        'model_dir': FLAGS.model_dir,
        'fine_tune': FLAGS.fine_tune,
        'num_classes': FLAGS.num_classes,
    }
    
    
    # Get model
    model = KerasXception(params)
    
    ##########################
    ##Prepare data generator##
    ##########################
    def get_train_generator():
        train_label_map = load_labels(json_path=train_label)

        # Read Data and Augment it: Make sure to select augmentations that are appropriate to your images.
        # To save augmentations un-comment save lines and add to your flow parameters.
        train_datagen = ImageDataGenerator(rescale=1. / 255)
        #                                    rotation_range=transformation_ratio,
        #                                    shear_range=transformation_ratio,
        #                                    zoom_range=transformation_ratio,
        #                                    cval=transformation_ratio,
        #                                    horizontal_flip=True,
        #                                    vertical_flip=True)
        # os.makedirs(os.path.join(os.path.abspath(train_data_dir), '../preview'), exist_ok=True)
        train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                            classes=range(num_classes),
                                                            target_size=(IMG_SIZE, IMG_SIZE),
                                                            batch_size=batch_size,
                                                            class_mode='multilabel',
                                                            multilabel_classes=train_label_map)
        # save_to_dir=os.path.join(os.path.abspath(train_data_dir), '../preview')
        # save_prefix='aug',
        # save_format='jpeg')
        # use the above 3 commented lines if you want to save and look at how the data augmentations look like
        return train_generator
    
    def get_validation_generator():
        valid_label_map = load_labels(json_path=valid_label)
        validation_datagen = ImageDataGenerator(rescale=1. / 255)
        validation_generator = validation_datagen.flow_from_directory(valid_data_dir,
                                                                      classes=range(num_classes),
                                                                      target_size=(IMG_SIZE, IMG_SIZE),
                                                                      batch_size=batch_size,
                                                                      shuffle=False,
                                                                      class_mode='multilabel',
                                                                      multilabel_classes=valid_label_map)
        return validation_generator
    
    def get_test_generator():
        test_datagen = ImageDataGenerator(rescale=1./255)
        test_generator = test_datagen.flow_from_directory(test_data_dir,
                                                          classes=range(num_classes),
                                                          target_size=(IMG_SIZE, IMG_SIZE),
                                                          batch_size=batch_size,
                                                          shuffle=False,
                                                          class_mode=None)
        return test_generator
    
    ###############################
    ##Prepare data generator DONE##
    ###############################
    
    if FLAGS.mode.lower() == "train":
        print("Training mode..")
        model.train(get_train_generator(),
                    max_queue_size=256,
                    epochs=epochs,
                    workers=12,
                    steps_per_epoch=steps_per_epoch,
                    validation_data=get_validation_generator())
        
    elif FLAGS.mode.lower() == "eval":
        print("Eval mode..")
        model.eval(get_validation_generator())
        
    elif FLAGS.mode.lower() == "test":
        print("Saving test data to: ", FLAGS.test_prediction)
        f=open(FLAGS.test_prediction, "w")
        f.write("image_id,label_id\n")
        img_id=1
        # deal with unified threshold or per class thresholding
        thresholds=[]
        if re.match("^\d+?\.\d+?$", FLAGS.pred_threshold) is None:
            print("Use per class thresholding.")
            thresholds=pd.read_csv(FLAGS.pred_threshold)['thresholds'].values
        else:
            th=float(FLAGS.pred_threshold)
            thresholds=[th for i in range(228)]
        
        test_pred=model.predict(get_test_generator())
        with tqdm(total=NUM_TEST) as progress_bar:
            for pred in test_pred:
                labels=" ".join([str(i+1) for i in range(len(pred)) if pred[i] >= thresholds[i]])
                f.write("%d,%s\n"%(img_id, labels))
                img_id += 1
                progress_bar.update(1)
        print("Processed %d examples. Good Luck! :)"%(img_id))
        f.close()
        
    elif FLAGS.mode.lower() == "debug":
        print("Debugging model, output class prediction probablities to %s."%(FLAGS.debug_dump_file))
        f=open(FLAGS.debug_dump_file, "w")
        f.write("image_id,label_prob\n")
        
        # TODO: Please set the corresponding input_fn for your data set!!
        valid_pred=model.predict(get_validation_generator())
        img_id=1
        
        with tqdm(total=NUM_VALID) as progress_bar:
            for pred in valid_pred:
                labels=" ".join(["%.2f"%(p) for p in pred])
                f.write("%d,%s\n"%(img_id, labels))
                img_id += 1
                progress_bar.update(1)
        print("Processed %d examples. Happy Debugging! :)"%(img_id))
        f.close()

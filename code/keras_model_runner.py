import tensorflow as tf
import numpy as np
import os
import pandas as pd
import re
from tqdm import tqdm
from utils.keras_image import ImageDataGenerator
from estimator.input_fn import load_labels
from xception import KerasXception
from wide_and_deep import WideDeep
from xception_rnn import KerasXceptionRNN
# from xception_v2 import KerasXception
# from densenet169 import KerasDenseNet

tf.app.flags.DEFINE_integer("augment", 0, "")
tf.app.flags.DEFINE_integer("batch_size", 64, "")
tf.app.flags.DEFINE_integer("num_threads", 8, "")

tf.app.flags.DEFINE_string("train_data_dir", '/home/fashion/data/train_processed', "")
tf.app.flags.DEFINE_string("train_label", '/home/fashion/data/train.json', "")
tf.app.flags.DEFINE_string("valid_data_dir", '/home/fashion/data/validation_processed', "")
tf.app.flags.DEFINE_string("valid_label", '/home/fashion/data/validation.json', "")
tf.app.flags.DEFINE_string("test_data_dir", '/home/fashion/data/test_processed', "")
tf.app.flags.DEFINE_string("test_prediction", '/home/shared/cs231n-fashion/submission/test_prediction.csv', "")
tf.app.flags.DEFINE_string("debug_dump_file", 'debug.csv', "")
tf.app.flags.DEFINE_string("model_dir", '/home/shared/cs231n-fashion/model_dir/keras_xception/', "")
tf.app.flags.DEFINE_string("wide_model_dir", '/home/shared/cs231n-fashion/model_dir/wide_keras_xception/', "")
tf.app.flags.DEFINE_string("deep_model_dir", '/home/shared/cs231n-fashion/model_dir/keras_xception/', "")

tf.app.flags.DEFINE_integer("num_classes", 228, "")
tf.app.flags.DEFINE_float("learning_rate", 3e-4, "")
tf.app.flags.DEFINE_integer("epochs", 36, "")
tf.app.flags.DEFINE_integer("steps_per_epoch", 1000, "")
tf.app.flags.DEFINE_integer("initial_epoch", 0, "")
tf.app.flags.DEFINE_float("reg", 1e-5, "")
tf.app.flags.DEFINE_float("drop_out_rate", 0.2, "")

tf.app.flags.DEFINE_bool("fine_tune", False, "Whether to fine tune the model or not.")
tf.app.flags.DEFINE_string("mode", "train", "train, eval, or test")
tf.app.flags.DEFINE_string("pred_threshold", "0.2", "the threshold for prediction")
tf.app.flags.DEFINE_string('gpu_id', '0', 'the device to use for training.')

tf.app.flags.DEFINE_bool('generator_use_wad', False, 'Whether to activate wide-and-deep mode for generator.')
tf.app.flags.DEFINE_bool('generator_use_weight', False, 'Whether to weight images/labels in training time.')
tf.app.flags.DEFINE_string('train_label_to_weight_map_path', 'data/train_label_to_weight_map.json', 'train_label_to_weight_map_path.')
tf.app.flags.DEFINE_string('train_labels_count_to_weight_map_path', 'data/train_labels_count_to_weight_map.json', 'train_labels_count_to_weight_map_path.')

tf.app.flags.DEFINE_integer('gru_hidden_size', 20, '')
tf.app.flags.DEFINE_bool('label_emb_trainable', True, '')
tf.app.flags.DEFINE_bool('rnn_concat_all', True, '')

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
    initial_epoch = FLAGS.initial_epoch

    params = {
        'model_dir': FLAGS.model_dir,
        'fine_tune': FLAGS.fine_tune,
        'num_classes': FLAGS.num_classes,
        'image_size': IMG_SIZE,
        'reg': FLAGS.reg,
        'wide_model_dir': FLAGS.wide_model_dir,
        'deep_model_dir': FLAGS.deep_model_dir,
        'drop_out_rate': FLAGS.drop_out_rate,
        
        'use_cudnn': True if FLAGS.gpu_id in ['0', '1'] else False,
        'gru_hidden_size': FLAGS.gru_hidden_size,
        'label_emb_trainable': FLAGS.label_emb_trainable,
        'rnn_concat_all': FLAGS.rnn_concat_all,
    }


    # Get model
#     model = KerasDenseNet(params)
#     model = KerasXception(params)
#     model = WideDeep(params)
    model = KerasXceptionRNN(params)

    ##########################
    ##Prepare data generator##
    ##########################
    generator_params = {
        'generator_use_wad': FLAGS.generator_use_wad,
        'generator_use_weight': False,
        'train_label_to_weight_map_path': '',
        'train_labels_count_to_weight_map_path': '',
    }

    def get_train_generator():
        train_label_map = load_labels(json_path=train_label)

        # Read Data and Augment it: Make sure to select augmentations that are appropriate to your images.
        # To save augmentations un-comment save lines and add to your flow parameters.
        train_generator_params = {
            **generator_params,
            'generator_use_weight': FLAGS.generator_use_weight,
            'train_label_to_weight_map_path': FLAGS.train_label_to_weight_map_path,
            'train_labels_count_to_weight_map_path': FLAGS.train_labels_count_to_weight_map_path,
        }
        train_datagen = ImageDataGenerator(rescale=1. / 255,
                                           horizontal_flip=True,
                                           vertical_flip=True,
                                           is_training=True,
                                           params=train_generator_params)
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
        validation_datagen = ImageDataGenerator(rescale=1. / 255, params=generator_params)
        validation_generator = validation_datagen.flow_from_directory(valid_data_dir,
                                                                      classes=range(num_classes),
                                                                      target_size=(IMG_SIZE, IMG_SIZE),
                                                                      batch_size=batch_size,
                                                                      shuffle=False,
                                                                      class_mode='multilabel',
                                                                      multilabel_classes=valid_label_map)
        return validation_generator

    def get_test_generator():
        test_datagen = ImageDataGenerator(rescale=1./255, params=generator_params)
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
                    initial_epoch=initial_epoch,
                    validation_data=get_validation_generator(),
                    validation_steps=None)
        exit(0)

    # Disable weighting in non-training settings.
    FLAGS.generator_use_weight = False

    if FLAGS.mode.lower() == "eval":
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

    elif FLAGS.mode.lower() in ["debug", "debug_test"]:
        is_test = FLAGS.mode.lower() == "debug_test"
        debug_generator = get_test_generator() if is_test else get_validation_generator()
        total_count = NUM_TEST if is_test else NUM_VALID

        print("Debugging model, output class prediction probablities to %s."%(FLAGS.debug_dump_file))
        with open(FLAGS.debug_dump_file, "w") as f:
            f.write("image_id,label_prob\n")

            probs = model.predict(debug_generator)
            img_id=1

            with tqdm(total=total_count) as progress_bar:
                for prob in probs:
                    labels=" ".join(["%.2f"%(p) for p in prob])
                    f.write("%d,%s\n"%(img_id, labels))
                    img_id += 1
                    progress_bar.update(1)
            print("Processed %d examples. Happy Debugging! :)"%(img_id))

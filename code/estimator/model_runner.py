import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os
import pandas as pd
import re
import input_fn
from tqdm import tqdm
import logging
# import ..ensemble
import time
# from baseline_model import model_fn
# from baseline_model_gru import model_fn
from baseline_model_dense import model_fn
# from baseline_model_dense2 import model_fn
import sys
parent_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
sys.path.append(parent_dir)
import ensemble


tf.app.flags.DEFINE_integer("augment", 0, "")
tf.app.flags.DEFINE_integer("batch_size", 32, "")
tf.app.flags.DEFINE_integer("num_threads", 8, "")

tf.app.flags.DEFINE_string("train_data_dir", '/home/fashion/data/train_processed', "")
tf.app.flags.DEFINE_string("train_label", '/home/fashion/data/train.json', "")
tf.app.flags.DEFINE_string("valid_data_dir", '/home/fashion/data/validation_processed', "")
tf.app.flags.DEFINE_string("valid_label", '/home/fashion/data/validation.json', "")
tf.app.flags.DEFINE_string("test_data_dir", '/home/fashion/data/test_processed', "")
tf.app.flags.DEFINE_string("test_prediction", '/home/shared/cs231n-fashion/submission/test_prediction.csv', "")
tf.app.flags.DEFINE_string("debug_dump_file", 'debug/debug.csv', "")
tf.app.flags.DEFINE_string("model_dir", '/home/shared/cs231n-fashion/model_dir/baseline2/', "")
tf.app.flags.DEFINE_string("ensemble_model_dir", '/home/shared/cs231n-fashion/model_dir/', "")

tf.app.flags.DEFINE_string("train_tfrecord", '/home/shared/cs231n-fashion/data/train_processed.tfrecords', '')
tf.app.flags.DEFINE_string("valid_tfrecord", '/home/shared/cs231n-fashion/data/validation_processed.tfrecords', '')
tf.app.flags.DEFINE_string("test_tfrecord", '/home/shared/cs231n-fashion/data/test_processed.tfrecords', '')

tf.app.flags.DEFINE_integer("hidden_size", 100, "")
tf.app.flags.DEFINE_integer("num_classes", 228, "")
tf.app.flags.DEFINE_float("learning_rate", 3e-4, "")
tf.app.flags.DEFINE_float("reg", 0.1, "")
tf.app.flags.DEFINE_integer("num_train_steps", -1, "")
tf.app.flags.DEFINE_integer("num_epoch", -1, "")
tf.app.flags.DEFINE_integer("num_train_per_eval", 31705, "")
tf.app.flags.DEFINE_string("eval_thresholds", "0.1;0.15;0.2;0.25;0.3;0.4;0.5;0.6;0.7;0.8;0.9", "the thresholds used in eval mode.")
tf.app.flags.DEFINE_bool("module_trainable", False, "whether the pretrained model is trainable or not.")

tf.app.flags.DEFINE_string("mode", "train", "train, eval, or test")
tf.app.flags.DEFINE_string('gpu_id', '0', 'the device to use for training.')
tf.app.flags.DEFINE_string('log_file', '', 'If not empty, will put tf log to the specific file path.')

FLAGS = tf.app.flags.FLAGS

os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu_id #this set the environment, maybe problematic.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # Logging.

NUM_TEST=39706
NUM_VALID=9897

class LossCheckerHook(tf.train.SessionRunHook):
    def begin(self):
        self.loss_collection=tf.get_collection(tf.GraphKeys.LOSSES)


# https://stackoverflow.com/a/44296581/4115411
# WARNING: this function still dumps to stdout.
# Another wordaround is to pipe the command with an output file address.
def set_log_to_file(log_file):
    print('Streaming tf log to file: {}'.format(log_file))
    time.sleep(3)
    log = logging.getLogger('tensorflow')
    log.setLevel('WARN')
    assert len(log.handlers) == 1, 'tf logging handler should contain only stdout.'
    log.removeHandler(log.handlers[0])

    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    log.addHandler(fh)


if __name__ == '__main__':
    if FLAGS.log_file:
        set_log_to_file(FLAGS.log_file)
    
    run_config=tf.estimator.RunConfig(
        session_config=tf.ConfigProto(log_device_placement=False),
        save_checkpoints_secs=30*60,
        keep_checkpoint_max=10,
    )
    
    params={
        'learning_rate': FLAGS.learning_rate,
        'num_classes': FLAGS.num_classes,
        'module_trainable': FLAGS.module_trainable,
        'eval_thresholds': [float(i) for i in FLAGS.eval_thresholds.split(';')],
        'model_dir': FLAGS.model_dir,
        'reg': FLAGS.reg
    }
    
    # Create the estimator.
    classifier = tf.estimator.Estimator(
        config=run_config,
        model_fn=model_fn,
        model_dir=FLAGS.model_dir,
        params=params
    )

    # Train the classifier.
    # Run evaluation every num_train_per_eval training steps.
    # Print the evalutation data and add the data into eval_data_list.
    steps_trained = 0
    
    # input_fn arguments.
    should_augment = FLAGS.augment
    batch_size = FLAGS.batch_size
    num_threads = FLAGS.num_threads
    train_data_dir = FLAGS.train_data_dir
    train_label = FLAGS.train_label
    valid_data_dir = FLAGS.valid_data_dir
    valid_label = FLAGS.valid_label
    
        
    # Define global variables.
    hidden_size = FLAGS.hidden_size
    num_classes = FLAGS.num_classes
    learning_rate = FLAGS.learning_rate
    num_train_steps = FLAGS.num_train_steps
    num_train_per_eval = FLAGS.num_train_per_eval
    num_epoch = FLAGS.num_epoch
        
    train_input_fn = lambda: input_fn.input_fn(
        train_data_dir,
        train_label,
        repeat=True,
        batch_size=batch_size,
        num_threads=num_threads,
    )

    valid_input_fn = lambda: input_fn.input_fn(
        valid_data_dir,
        valid_label,
        repeat=False,
        batch_size=batch_size,
        num_threads=num_threads,
    )
    
    test_input_fn = lambda: input_fn.input_fn(
        FLAGS.test_data_dir,
        None,
        repeat=False,
        test_mode=True,
        batch_size=batch_size,
        num_threads=num_threads,
    )
    
    #########################################
    train_tfr_input_fn = lambda: input_fn.tf_record_input_fn(
        FLAGS.train_tfrecord, 
        batch_size=batch_size,
        repeat=True,
        num_threads=num_threads)
    
    valid_tfr_input_fn = lambda: input_fn.tf_record_input_fn(
        FLAGS.valid_tfrecord,
        shuffle=False,
        repeat=False,
        batch_size=batch_size,
        num_threads=num_threads)
        
    test_tfr_input_fn = lambda: input_fn.tf_record_input_fn(
        FLAGS.test_tfrecord,
        shuffle=False,
        repeat=False,
        batch_size=batch_size, 
        num_threads=num_threads)
    #########################################
    
    tf.logging.set_verbosity(tf.logging.INFO)
    
    if FLAGS.mode.lower() == "train":
        print("Training mode..")
        cur_epoch = 1
        classifier.train(train_tfr_input_fn)

#         while num_epoch == -1 or cur_epoch < num_epoch:
#             print("Epoch %d."%(cur_epoch))
#             classifier.train(train_tfr_input_fn)
#             eval_data=classifier.evaluate(valid_input_fn)
#             print("Eval data: ", eval_data)
#             cur_epoch += 1
        
#         print("Train done: %d epochs since last run."%(cur_epoch))
        
    elif FLAGS.mode.lower() == "eval":
        eval_data=classifier.evaluate(valid_tfr_input_fn)
        print("Eval data: ", eval_data)
        
    elif FLAGS.mode.lower() == "test":
        # KV is 'label': ('model_file', 'exp_name', weight)
        if FLAGS.ensemble_model_dir:
            ensemble_label_to_model_meta = {
                'baseline': ['baseline_model', 'baseline', 0.2],
                'baseline_dense2': ['baseline_model_dense2', 'baseline_dense2', 0.5],
            }
            probs = ensemble.predict(ensemble_label_to_model_meta, run_config, params, test_input_fn)
        else:
            probs = np.array([
                pred['probs']
                for pred in classifier.predict(test_input_fn)
            ])
        ensemble.write_predictions(probs, FLAGS.test_prediction)
        
    elif FLAGS.mode.lower() in ["debug", "debug_test"]:
        is_test = FLAGS.mode.lower() == "debug_test"
        debug_input_fn = test_input_fn if is_test else valid_tfr_input_fn
        total_count = NUM_TEST if is_test else NUM_VALID
        
        print("Debugging model, output class prediction probablities to file.")
        f=open(FLAGS.debug_dump_file, "w")
        f.write("image_id,label_prob\n")
        
        # TODO: Please set the corresponding input_fn for your data set!!
        preds = classifier.predict(debug_input_fn)
        img_id=1
        
        with tqdm(total=total_count) as progress_bar:
            for pred in preds:
                labels=" ".join(["%.2f"%(p) for p in pred['probs']])
                f.write("%d,%s\n"%(img_id, labels))
                img_id += 1
                progress_bar.update(1)
        print("Processed %d examples. Happy Debugging! :)"%(img_id))
        f.close()

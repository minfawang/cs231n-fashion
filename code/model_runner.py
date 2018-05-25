import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os
import input_fn
from tqdm import tqdm
# from baseline_model import model_fn
# from baseline_model_gru import model_fn
# from baseline_model_dense import model_fn
from baseline_model_dense2 import model_fn

tf.app.flags.DEFINE_integer("augment", 0, "")
tf.app.flags.DEFINE_integer("batch_size", 32, "")
tf.app.flags.DEFINE_integer("num_threads", 8, "")

tf.app.flags.DEFINE_string("train_data_dir", '/home/fashion/data/train_processed', "")
tf.app.flags.DEFINE_string("train_label", '/home/fashion/data/train.json', "")
tf.app.flags.DEFINE_string("valid_data_dir", '/home/fashion/data/validation_processed', "")
tf.app.flags.DEFINE_string("valid_label", '/home/fashion/data/validation.json', "")
tf.app.flags.DEFINE_string("test_data_dir", '/home/fashion/data/test_processed', "")
tf.app.flags.DEFINE_string("test_prediction", '/home/shared/cs231n-fashion/submission/test_prediction.csv', "")
tf.app.flags.DEFINE_string("model_dir", '/home/shared/cs231n-fashion/model_dir/baseline2/', "")

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
tf.app.flags.DEFINE_float("pred_threshold", "0.2", "the threshold for prediction")

tf.app.flags.DEFINE_string('gpu_id', '0', 'the device to use for training.')


FLAGS = tf.app.flags.FLAGS

os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu_id #this set the environment, maybe problematic.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # Logging.

NUM_TEST=39706

class LossCheckerHook(tf.train.SessionRunHook):
    def begin(self):
        self.loss_collection=tf.get_collection(tf.GraphKeys.LOSSES)


if __name__ == '__main__':

    run_config=tf.estimator.RunConfig(
        session_config=tf.ConfigProto(log_device_placement=True),
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
        repeat=False,
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
        while num_epoch == -1 or cur_epoch < num_epoch:
            print("Epoch %d."%(cur_epoch))
            classifier.train(train_tfr_input_fn)
            eval_data=classifier.evaluate(valid_input_fn)
            print("Eval data: ", eval_data)
            cur_epoch += 1
        
        print("Train done: %d epochs since last run."%(cur_epoch))
        
    elif FLAGS.mode.lower() == "eval":
        eval_data=classifier.evaluate(valid_tfr_input_fn)
        print("Eval data: ", eval_data)
        
    elif FLAGS.mode.lower() == "test":
        print("Saving test data to: ", FLAGS.test_prediction)
        f=open(FLAGS.test_prediction, "w")
        f.write("image_id,label_id\n")
        test_pred=classifier.predict(test_input_fn)
        img_id=1
        
        with tqdm(total=NUM_TEST) as progress_bar:
            for pred in test_pred:
                labels=" ".join([str(i+1) for i in range(len(pred['probs'])) if pred['probs'][i] >= FLAGS.pred_threshold])
                f.write("%d,%s\n"%(img_id, labels))
                img_id += 1
                progress_bar.update(1)
        print("Processed %d examples. Good Luck! :)"%(img_id))
        f.close()
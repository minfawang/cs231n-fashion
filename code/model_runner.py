import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os
import input_fn
from baseline_model import model_fn

tf.app.flags.DEFINE_integer("augment", 0, "")
tf.app.flags.DEFINE_integer("batch_size", 32, "")
tf.app.flags.DEFINE_integer("num_threads", 8, "")

tf.app.flags.DEFINE_string("train_data_dir", '/home/shared/cs231n-fashion/data/train_processed', "")
tf.app.flags.DEFINE_string("train_label", '/home/shared/cs231n-fashion/data/train.json', "")
tf.app.flags.DEFINE_string("valid_data_dir", '/home/shared/cs231n-fashion/data/validation_processed', "")
tf.app.flags.DEFINE_string("valid_label", '/home/shared/cs231n-fashion/data/validation.json', "")
tf.app.flags.DEFINE_string("model_dir", './model_dir/baseline', "")

tf.app.flags.DEFINE_integer("hidden_size", 100, "")
tf.app.flags.DEFINE_integer("num_classes", 228, "")
tf.app.flags.DEFINE_float("learning_rate", 3e-4, "")
tf.app.flags.DEFINE_integer("num_train_steps", -1, "")
tf.app.flags.DEFINE_integer("num_train_per_eval", 1000, "")
tf.app.flags.DEFINE_integer("num_step_to_eval", 50, ".")
tf.app.flags.DEFINE_integer("num_iter_to_eval_on_valid", 16, "")

tf.app.flags.DEFINE_string("mode", "train", "train, eval, or predict")
    
FLAGS = tf.app.flags.FLAGS
os.environ["CUDA_VISIBLE_DEVICES"] = '0' # Use the first GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # Logging.


class LossCheckerHook(tf.train.SessionRunHook):
    def begin(self):
        self.loss_collection=tf.get_collection(tf.GraphKeys.LOSSES)


if __name__ == '__main__':

    run_config=tf.estimator.RunConfig(
        session_config=tf.ConfigProto(log_device_placement=True),
        save_checkpoints_secs=10*60,
        keep_checkpoint_max=5,
    )
    
    # Create the estimator.
    classifier = tf.estimator.Estimator(
	    config=run_config,
	    model_fn=model_fn,
	    model_dir=FLAGS.model_dir)

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
    num_step_to_eval = FLAGS.num_step_to_eval
    num_iter_to_eval_on_valid = FLAGS.num_iter_to_eval_on_valid
        
    train_input_fn = lambda: input_fn.input_fn(
        train_data_dir,
        train_label,
        batch_size=batch_size)

    valid_input_fn = lambda: input_fn.input_fn(
        valid_data_dir,
        valid_label,
        repeat=False,
        batch_size=batch_size)

    if num_train_steps == -1:
        num_train_steps = None 
    
    if FLAGS.mode.lower() == "train":
        print("Training mode..")
        tf.logging.set_verbosity(tf.logging.INFO)
        logging_hook=tf.train.LoggingTensorHook({
		"loss": "loss",
                #"auc": "auc",
                #"f1_3": "f1_3",
                #"f1_5": "f1_5",
                #"f1_7": "f1_7",
		}, every_n_iter=16)
        classifier.train(train_input_fn, 
#			 hooks=[logging_hook],
                         steps=num_train_steps)
    elif FLAGS.mode.lower() == "eval":
        eval_data=classifier.evaluate(valid_input_fn)
        print("Eval data: ", eval_data)
    else:
        print("not supported yet")
            
#     while num_train_steps < 0 or steps_trained < num_train_steps:
#         print("Training, step: %d..."%steps_trained)
#         classifier.train(train_input_fn, 
#                          steps=num_train_per_eval)
        
#         print("Evaluating on train, step: %d..."%steps_trained)
#         eval_data=classifier.evaluate(train_input_fn, steps=num_step_to_eval)
#         print("Train data: ",eval_data)
        
#         print("Evaluating on validation, step: %d..."%steps_trained)
#         eval_data=classifier.evaluate(valid_input_fn, steps=num_step_to_eval)
#         print("Eval data: ", eval_data)
        
#         steps_trained += num_train_per_eval
        
#         if (steps_trained//num_train_per_eval)%num_iter_to_eval_on_valid == 0:
#             print ("Evaluate on full examples from validation set.")
#             classifier.evaluate(valid_input_fn)

    # Example output for running evaluate function.
    classifier.evaluate(valid_input_fn)
    

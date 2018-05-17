import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os
import input_fn

def model_fn(features, labels, mode):
    """Model function for fashion classes predictions.
    
    Inputs:
        features: dict.
            Required key: "image". The value needs to have shape (batch_size, 299, 299, 3),
                and each value needs to be in range [0, 1].
        labels: shape (batch_size, num_classes)
        mode: tf.estimator.ModeKeys.(PREDICT|TRAIN|EVAL)
    
    Returns:
        estimator_spec.
    """
    # class_prob > threshold will be outputted.
    threshold = 0.5
    
    # Input layer.
    images = features
    module = hub.Module("https://tfhub.dev/google/imagenet/inception_resnet_v2/feature_vector/1")
    features = module(images)  # (batch_size, D)
    
    # Create multi-head sigmoid outputs.
    # It measures the independent probability of a class showing in the image.
    raw_logits = tf.contrib.layers.fully_connected(
        inputs=features,
        num_outputs=num_classes,
        activation_fn=None)  # (batch_size, num_classes)
    
    raw_probs = tf.sigmoid(raw_logits)  # (batch_size, num_classes)
    
#     # RNN layer.
#     gru_cell = tf.nn.rnn_cell.GRUCell(hidden_size)
#     outputs, _ = tf.nn.dynamic_rnn(cell=gru_cell, inputs=)
    
    predictions = {
        'pred_3': (raw_probs > 0.3),
        'pred_5': (raw_probs > 0.5),
        'pred_7': (raw_probs > 0.7),
        'probs': raw_probs,
    }
    
    # PREDICT mode.
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    
    # Calculate loss (for both TRAIN and EVAL mode).
    loss = tf.losses.sigmoid_cross_entropy(
        multi_class_labels=labels,
        logits=raw_logits)
    
    # TRAIN mode.
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)    

    # Add evalutaion metrics (for EVAL mode).
    auc = tf.metrics.auc(
        labels=labels,
        predictions=predictions['probs'])
    
    precisions_3 = tf.metrics.precision(
        labels=labels, predictions=predictions['pred_3']
    recalls_3 = tf.metrics.recall(
        labels=labels, predictions=predictions['pred_3'])
    mean_f1_3 = (2*precisions_3[0]*recalls_3[0]/(precisions_3[0]+recalls_3[0]), precisions_3[1])    
    
    precisions_5 = tf.metrics.precision(
        labels=labels, predictions=predictions['pred_5'])
    recalls_5 = tf.metrics.recall(
        labels=labels, predictions=predictions['pred_5'])   
    mean_f1_5 = (2*precisions_5[0]*recalls_5[0]/(precisions_5[0]+recalls_5[0]), precisions_5[1])
    
    precisions_7 = tf.metrics.precision(
        labels=labels, predictions=predictions['pred_7'])
    recalls_7 = tf.metrics.recall(
        labels=labels, predictions=predictions['pred_7']   
    mean_f1_7 = (2*precisions_7[0]*recalls_7[0]/(precisions_7[0]+recalls_7[0]), precisions_7[1]) 
    
    eval_metric_ops = {
        'precisions_0.3': precisions_3,
        'recalls_0.3': recalls_3,
        'mean_f1_0.3': mean_f1_3,
        'precisions_0.5': precisions_5,
        'recalls_0.5': recalls_5,
        'mean_f1_0.5': mean_f1_5,
        'precisions_0.7': precisions_7,
        'recalls_0.7': recalls_7,
        'mean_f1_0.7': mean_f1_7,
        'auc': auc,
    }
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


class LossCheckerHook(tf.train.SessionRunHook):
    def begin(self):
        self.loss_collection=tf.get_collection(tf.GraphKeys.LOSSES)


if __name__ == '__main__':
    run_config=tf.estimator.RunConfig(session_config=tf.ConfigProto(log_device_placement=True))
    # Create the estimator.
    classifier = tf.estimator.Estimator(
	    config=run_config,
	    model_fn=model_fn,
	    model_dir='../model_dir/baseline')

    # Train the classifier.
    # Run evaluation every num_train_per_eval training steps.
    # Print the evalutation data and add the data into eval_data_list.
    steps_trained = 0
    
    # input_fn arguments.
    should_augment = False
    batch_size = 64
    num_threads = 8
    train_data_dir = '/home/shared/cs231n-fashion/data/train_processed'
    train_label = '/home/shared/cs231n-fashion/data/train.json'
    valid_data_dir = '/home/shared/cs231n-fashion/data/validation_processed'
    valid_label = '/home/shared/cs231n-fashion/data/validation.json'
    
        
    # Define global variables.
    hidden_size = 100
    num_classes = 228
    learning_rate = 3e-4
    num_train_steps = -1
    num_train_per_eval = 1000
    num_step_to_eval = 1000//batch_size
    num_iter_to_eval_on_valid = 16
        
    os.environ["CUDA_VISIBLE_DEVICES"] = '0' # Use the first GPU
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Warning.
    
    train_input_fn = lambda: input_fn.input_fn(
        train_data_dir,
        train_label,
        batch_size=batch_size)

    valid_input_fn = lambda: input_fn.input_fn(
        valid_data_dir,
        valid_label,
        repeat=False,
        batch_size=batch_size)

    loss_hook = LossCheckerHook()
    while num_train_steps < 0 or steps_trained < num_train_steps:
        classifier.train(train_input_fn, 
                         steps=num_train_per_eval,
                         hooks=[loss_hook])
        classifier.evaluate(train_input_fn, steps=num_step_to_eval)
        classifier.evaluate(valid_input_fn, steps=num_step_to_eval)
        steps_trained += num_train_per_eval
        
        if (steps_trained//num_train_per_eval)%num_iter_to_eval_on_valid == 0:
            print ("Evaluate on full examples from validation set.")
            classifier.evaluate(valid_input_fn)

    # Example output for running evaluate function.
    classifier.evaluate(valid_input_fn)
    

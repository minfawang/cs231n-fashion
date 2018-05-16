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
    thresholds =[0.3, 0.5, 0.7]
    threshold = 0.5
#     threshold = tf.get_variable('threshold_unbound', initializer=0.5)
#     threshold = tf.clip_by_value(threshold, 0.1, 0.9, name='threshold')
    
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
        'classes': raw_probs > threshold,
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
    precisions = tf.metrics.precision_at_thresholds(
        labels=labels,
        predictions=predictions['probs'],
        thresholds=thresholds)
    recalls = tf.metrics.recall_at_thresholds(
        labels=labels,
        predictions=predictions['probs'],
        thresholds=thresholds)
    auc = tf.metrics.auc(
        labels=labels,
        predictions=predictions['probs'])
    
    mean_f1 = (2*precisions[0]*recalls[0]/(precisions[0]+recalls[0]), precisions[1])
    eval_metric_ops = {
        'precisions': precisions,
        'recalls': recalls,
        'auc': auc,
        'mean_f1': mean_f1,
    }
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

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

    has_trained_steps = 0
    eval_data_list = []


    # Define global variables.
    hidden_size = 100
    num_classes = 228
    learning_rate = 3e-4
    num_train_steps = -1
    num_train_per_eval = 1000
    batch_to_eval = 100

    train_input_fn = lambda: input_fn.input_fn(
        input_folder,
        label_json_path,
        augment=should_augment,
        batch_size=batch_size,
        num_threads=num_threads,
        images_limit=images_limit)
    
    # input_fn arguments.
    should_augment = False
    images_limit = 1000  # How many images to train.
    batch_size = 32
    num_threads = 8
    input_folder = '/home/shared/cs231n-fashion/data/train_processed'
    label_json_path = '/home/shared/cs231n-fashion/data/train.json'

    # Set up some global variables
    gpu_to_use = 0
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_to_use)

    while num_train_steps < 0 or has_trained_steps < num_train_steps:
        classifier.train(train_input_fn, steps=num_train_per_eval)
        print("Evaluating on train data.")
        # TODO: add eval_input_fn.
        eval_data = classifier.evaluate(train_input_fn, steps=1)
        print(eval_data)
        eval_data_list.append(eval_data)

        has_trained_steps += num_train_per_eval

    # Example output for running evaluate function.
    classifier.evaluate(train_input_fn, steps=batch_to_eval)

